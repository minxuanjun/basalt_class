#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <tbb/blocked_range.h>

#include <basalt/vi_estimator/landmark_database.h>


namespace basalt
{
// 这个类作用是实现最新帧的位姿估计
// 传递的参数有last_frame的3d点和其在当前帧的投影点

class pose_optimizer
{
public:
    explicit pose_optimizer(const Eigen::aligned_unordered_map<int, KeypointPosition> &lm,
                            const OpticalFlowResult::Ptr &opt_flow_meas,
                            const basalt::Calibration<double> &calib)
        : landmarks_(lm),
          optical_flow_meas_(opt_flow_meas),
          calib_(calib)
    {
    }

    virtual ~pose_optimizer() = default;

    Sophus::SE3d optimize(Sophus::SE3d T_cur_ref_init) const
    {

        Sophus::SE3d back_up_state = T_cur_ref_init; // 保存上一时刻的的状态值
        for (size_t iter = 0; iter < num_each_iter_; iter++) {
            auto t1 = std::chrono::high_resolution_clock::now();

            // step1: 计算hessian矩阵
            Sophus::Matrix6d Hpp;
            Sophus::Vector6d bp;
            double error = 0.0;
            computeNormalEquation(back_up_state, Hpp, bp, error);
            // step2: 计算hessian矩阵的对角阵
            Sophus::Vector6d Hdiag = Hpp.diagonal(); //获取hessian矩阵的对角阵

            bool converged = false;
            // step3: LM 迭代一次
            {// Use Levenberg–Marquardt

                bool step = false;
                int max_iter = 10;
                while (!step && max_iter > 0 and !converged) {
                    Sophus::Vector6d Hdiag_lambda = Hdiag * lambda;
                    for (int i = 0; i < Hdiag_lambda.size(); i++) {
                        if (Hdiag_lambda[i] > 0)
                            Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);
                        else
                            Hdiag_lambda[i] = std::min(Hdiag_lambda[i], -min_lambda);
                    }

                    Sophus::Matrix6d HH = Hpp;
                    HH.diagonal() += Hdiag_lambda;
                    Sophus::Vector6d inc = HH.ldlt().solve(-bp);
                    double max_inc = inc.array().abs().maxCoeff();
                    if (max_inc < 1e-4) converged = true; //如果增量很小了,迭代停止

                    // apply increment to poses
                    Sophus::SE3d new_state = back_up_state;
                    new_state.translation() += inc.head<3>();
                    new_state.so3() = Sophus::SO3d::exp(inc.tail<3>()) * new_state.so3();
                    // 判断状态更新后误差是否减小了
                    double new_error;
                    computeError(new_state, new_error); // 计算在新的状态下的误差

                    double f_diff = (error - new_error);

                    if (f_diff < 0) {
                        //                        std::cout << "\t[REJECT lambda]: " << lambda
                        //                                  << " f_diff: " << f_diff
                        //                                  << " max_inc: " << max_inc
                        //                                  << " Error: " << new_error << std::endl;
                        lambda = std::min(max_lambda, lambda_vee * lambda);
                        lambda_vee *= 4;
                    }
                    else {
                        //                        std::cout << "\t[ACCEPTED] lamdba: " << lambda
                        //                                  << " f_diff: " << f_diff
                        //                                  << " max_inc: " << max_inc
                        //                                  << " Error: " << new_error << std::endl;
                        // update the backup state using the new state
                        back_up_state = new_state;
                        lambda = std::max(min_lambda, lambda / 3);
                        lambda_vee = 2;

                        step = true;
                    }
                    max_iter--;
                }//one step of Levenberg–Marquardt is over

            }// LM 迭代结束

            auto t2 = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            // 计算每次迭代之后重投影误差
            double after_update_error = 0;
            computeError(back_up_state, after_update_error);
#if 0
            std::cout << "--------------------" << std::endl;
            std::cout << "iter:  " << iter
                      << "\n before_update_error: vision" << error
                      << std::endl;
            std::cout << "iter: " << iter
                      << "\n after_update_error: vision" << after_update_error
                      << "\n times : " << elapsed.count()
                      << std::endl;
            std::cout << "--------------------" << std::endl;
            if (after_update_error > error)
                std::cout << "increased error after update!!!" << std::endl;
#endif
            //TODO: 到达一定迭代次数时,剔除一些外点
            //            if (iter == filter_iteration)
            //            {
            //
            //            }
            // 如果
            if (converged) break;
        }// LM优化终止

        return back_up_state;
    }
    /**
     * \brief 计算在某个位姿下的重投影误差
     * @param [input] T_cur_ref
     * @param [output] error
     */
    void computeError(Sophus::SE3d T_cur_ref, double &error) const
    {
        error = 0.0;
        std::visit(
            [&](const auto &cam)
            {
                for (const auto &lm: landmarks_) // 遍历公视的landmark
                {
                    // 获得3d点的观测值
                    Eigen::Vector2d
                        obs = optical_flow_meas_->observations[0][lm.first].translation().cast<double>();
                    KeypointObservation lm_obs;
                    lm_obs.kpt_id = lm.first;
                    lm_obs.pos = obs;

                    // 求重投影误差的误差
                    Eigen::Vector2d res;
                    bool valid = linearizePoint(lm_obs, lm.second, T_cur_ref.matrix(), cam, res);
                    if (valid) {
                        double e = res.norm();
                        // 计算huber核权重
                        double huber_weight =
                            e < huber_thresh_ ? 1.0 : huber_thresh_ / e;
                        // 计算huber核权重和测量的信息矩阵的乘ji
                        double obs_weight =
                            huber_weight / (obs_std_dev_ * obs_std_dev_);
                        //                        std::cout << "point: " << lm_obs.kpt_id << " " << "error: " << res.transpose() * res << std::endl;
                        error += (2 - huber_weight) * obs_weight * res.transpose() * res;
                    }
                }// 共视的3d点遍历

            },
            calib_.intrinsics[0].variant
        );
    }

    /**
     * \brief 计算在某个位姿下的重投影误差和Normal Equatation
     * @param [input]  T_cur_ref
     * @param [output] Hpp
     * @param [output] bp
     * @param [output] error
     */
    void computeNormalEquation(Sophus::SE3d T_cur_ref, Sophus::Matrix6d &Hpp, Sophus::Vector6d &bp, double &error) const
    {
        // reset variable
        Hpp.setZero();
        bp.setZero();
        error = 0.0;

        std::visit(
            [&](const auto &cam)
            {
                for (const auto &lm: landmarks_) // 遍历公视的landmark
                {
                    // 获得3d点的观测值
                    Eigen::Vector2d
                        obs = optical_flow_meas_->observations[0][lm.first].translation().cast<double>();
                    KeypointObservation lm_obs;
                    lm_obs.kpt_id = lm.first;
                    lm_obs.pos = obs;

                    // 求重投影误差的误差和雅克比矩阵
                    Eigen::Vector2d res;
                    Eigen::Matrix<double, 2, POSE_SIZE> d_res_d_xi;
                    bool valid = linearizePoint(lm_obs, lm.second, T_cur_ref.matrix(), cam, res, &d_res_d_xi);

                    if (valid) {
                        double e = res.norm();
                        // 计算huber核权重
                        double huber_weight =
                            e < huber_thresh_ ? 1.0 : huber_thresh_ / e;
                        // 计算huber核权重和测量的信息矩阵的乘ji
                        double obs_weight =
                            huber_weight / (obs_std_dev_ * obs_std_dev_);
                        error += (2 - huber_weight) * obs_weight * res.transpose() * res;

                        Hpp += obs_weight * d_res_d_xi.transpose() * d_res_d_xi;
                        bp += obs_weight * d_res_d_xi.transpose() * res;
                    }
                }// 共视的3d点遍历

            },
            calib_.intrinsics[0].variant);
    }
    /**
     * \brief 求解重投影误差,重投影误差对相机位姿的导数,对3d点的导数
     * @tparam CamT
     * @param [in] kpt_obs
     * @param [in] kpt_pos
     * @param [in] T_t_h
     * @param [in] cam
     * @param [out] res
     * @param [out] d_res_d_xi
     * @param [out] d_res_d_p
     * @param [out] proj
     * @return
     */
    template<class CamT>
    static bool linearizePoint(
        const KeypointObservation &kpt_obs, const KeypointPosition &kpt_pos,
        const Eigen::Matrix4d &T_t_h, const CamT &cam, Eigen::Vector2d &res,
        Eigen::Matrix<double, 2, POSE_SIZE> *d_res_d_xi = nullptr,
        Eigen::Matrix<double, 2, 3> *d_res_d_p = nullptr,
        Eigen::Vector4d *proj = nullptr)
    {
        // Todo implement without jacobians
        Eigen::Matrix<double, 4, 2> Jup;
        // product 3d point unit direction
        Eigen::Vector4d p_h_3d;
        p_h_3d = StereographicParam<double>::unproject(kpt_pos.dir, &Jup);
        p_h_3d[3] = kpt_pos.id;

        Eigen::Vector4d p_t_3d = T_t_h * p_h_3d;

        Eigen::Matrix<double, 4, POSE_SIZE> d_point_d_xi;
        d_point_d_xi.topLeftCorner<3, 3>() =
            Eigen::Matrix3d::Identity() * kpt_pos.id;
        d_point_d_xi.topRightCorner<3, 3>() = -Sophus::SO3d::hat(p_t_3d.head<3>());
        d_point_d_xi.row(3).setZero();

        Eigen::Matrix<double, 2, 4> Jp;
        bool valid = cam.project(p_t_3d, res, &Jp);
        valid &= res.array().isFinite().all();

        if (!valid) {
            //      std::cerr << " Invalid projection! kpt_pos.dir "
            //                << kpt_pos.dir.transpose() << " kpt_pos.id " <<
            //                kpt_pos.id
            //                << " idx " << kpt_obs.kpt_id << std::endl;

            //      std::cerr << "T_t_h\n" << T_t_h << std::endl;
            //      std::cerr << "p_h_3d\n" << p_h_3d.transpose() << std::endl;
            //      std::cerr << "p_t_3d\n" << p_t_3d.transpose() << std::endl;

            return false;
        }

        if (proj) {
            proj->head<2>() = res;
            (*proj)[2] = p_t_3d[3] / p_t_3d.head<3>().norm();
        }
        res -= kpt_obs.pos;

        if (d_res_d_xi) {
            *d_res_d_xi = Jp * d_point_d_xi;
        }

        if (d_res_d_p) {
            Eigen::Matrix<double, 4, 3> Jpp;
            Jpp.setZero();
            Jpp.block<3, 2>(0, 0) = T_t_h.topLeftCorner<3, 4>() * Jup;
            Jpp.col(2) = T_t_h.col(3);

            *d_res_d_p = Jp * Jpp;
        }

        return true;
    }


private:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //! landmark
    const Eigen::aligned_unordered_map<int, KeypointPosition> &landmarks_;
    //! 2d observation
    const OpticalFlowResult::Ptr &optical_flow_meas_;
    //! camera
    const basalt::Calibration<double> &calib_;
    //! huber threshold
    const double huber_thresh_ = 1.0;
    //! reprojection error std
    const double obs_std_dev_ = 1.5;
    //! robust optimization的尝试次数
    const size_t num_trails_ = 4;
    //! 每回optimization 的iteration 回数
    const size_t num_each_iter_ = 50;
    //! filter_iteration
    const size_t filter_iteration = 4;
    double outlier_threshold = 3.0;
    // ! Levenberg Marquardt paramters
    mutable double lambda = 1e-32;
    mutable double min_lambda = 1e-32;
    mutable double max_lambda = 1e2;
    mutable double lambda_vee = 2;
};// class pose_optimizer

}// namespace basalt