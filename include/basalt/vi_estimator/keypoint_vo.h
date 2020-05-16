/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <memory>
#include <thread>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <basalt/imu/preintegration.h>
#include <basalt/io/dataset_io.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/test_utils.h>
#include <basalt/camera/generic_camera.hpp>
#include <basalt/camera/stereographic_param.hpp>
#include <basalt/utils/sophus_utils.hpp>

#include <basalt/vi_estimator/ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
namespace basalt
{
// 类的预先声明

class KeypointVoEstimator;

// SLAM 状态的预先声明
enum class SLAMStage
{
    NO_IMAGE_YET,
    NOT_INITIALIZED,
    OK,
    LOST
};

enum class SensorType
{
    Mono,
    Stereo,
    MonoVIO,
    StereoVIO
};

class MonoInitializer
{
public:

    MonoInitializer(const basalt::Calibration<double> &calib,
                    float kRansacProb = 0.999,
                    float kRansacThresholdNormalized = 0.0003
    )
        : calib_(calib),
          kRansacProb_(kRansacProb),
          kRansacThresholdNormalized_(kRansacThresholdNormalized)
    {}
    Sophus::SE3d estimatePose(std::vector<cv::Point2f> &kpn_cur, std::vector<cv::Point2f> &kpn_ref)
    {
        // 计算本质矩阵
        mask_match_.clear();
        cv::Mat E_ref_cur = cv::findEssentialMat(kpn_cur,
                                                 kpn_ref,
                                                 1.0,
                                                 cv::Point2d(0, 0),
                                                 cv::RANSAC,
                                                 kRansacProb_,
                                                 kRansacThresholdNormalized_,
                                                 mask_match_);

        cv::Mat R, t;
        // 恢复位姿
        cv::recoverPose(E_ref_cur, kpn_cur, kpn_ref, R, t, 1.0, cv::Point2d(0, 0));

        Eigen::Matrix3d R_ref_cur;
        R_ref_cur(0, 0) = R.at<double>(0, 0);
        R_ref_cur(0, 1) = R.at<double>(0, 1);
        R_ref_cur(0, 2) = R.at<double>(0, 2);
        R_ref_cur(1, 0) = R.at<double>(1, 0);
        R_ref_cur(1, 1) = R.at<double>(1, 1);
        R_ref_cur(1, 2) = R.at<double>(1, 2);
        R_ref_cur(2, 0) = R.at<double>(2, 0);
        R_ref_cur(2, 1) = R.at<double>(2, 1);
        R_ref_cur(2, 2) = R.at<double>(2, 2);
        Eigen::Vector3d t_ref_cur;
        t_ref_cur << t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0);

        Sophus::SE3d T_ref_cur;
        T_ref_cur.setRotationMatrix(R_ref_cur);
        T_ref_cur.translation() = t_ref_cur;

        return T_ref_cur;
    }
    // 三角化函数
    /// Triangulates the point and returns homogenous representation. First 3
    /// components - unit-length direction vector. Last component inverse
    /// distance.
    template<class Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 1> triangulate(
        const Eigen::MatrixBase<Derived> &f0,
        const Eigen::MatrixBase<Derived> &f1,
        const Sophus::SE3<typename Derived::Scalar> &T_0_1)
    {
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
        //method1：
        // d1*P1 = d2*R*P2 + t
        // P3 = RP2
        // d1*P1.transpose()*P1 = d2*P1.tanspose()*P3 + P1.transpose()*t
        // d1*P3.transpose()*P1 = d2*P3.transpose() + P3.transpose()*t
        // 然后整理成关于d1 d2的二元一次方程
        // method2: Linear triangulate method see chepter 12.2
        // Linear triangulation methods of 'Multiple View Geometry'
        using Scalar = typename Derived::Scalar;
        using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

        Eigen::Matrix<Scalar, 3, 4> P1, P2;
        P1.setIdentity();
        P2 = T_0_1.inverse().matrix3x4();

        Eigen::Matrix<Scalar, 4, 4> A(4, 4);
        A.row(0) = f0[0] * P1.row(2) - f0[2] * P1.row(0);
        A.row(1) = f0[1] * P1.row(2) - f0[2] * P1.row(1);
        A.row(2) = f1[0] * P2.row(2) - f1[2] * P2.row(0);
        A.row(3) = f1[1] * P2.row(2) - f1[2] * P2.row(1);

        Eigen::JacobiSVD<Eigen::Matrix<Scalar, 4, 4>> mySVD(A, Eigen::ComputeFullV);
        Vec4 worldPoint = mySVD.matrixV().col(3);
        worldPoint /= worldPoint.template head<3>().norm();

        // Enforce same direction of bearing vector and initial point
        if (f0.dot(worldPoint.template head<3>()) < 0) worldPoint *= -1;

        return worldPoint;
    }

    // push the first image
    void init(OpticalFlowResult::Ptr f_cur)
    {
        std::cout << "mono init" << std::endl;
        frames_.push_back(f_cur);
    }
    // actually initialize having two available frames
    bool initialize(OpticalFlowResult::Ptr f_cur, LandmarkDatabase &lmdb, int &num_points_added)
    {
        // if too many frames have passed, move the current id_ref forward
        // this is just one possible policy which can be used
        // 等于号判断是必须的
        std::cout << frames_[0] << std::endl;
        if (static_cast<int>(frames_.size() - 1) - id_ref_ >= kMaxIdDistBetweenFrames_
            or f_ref_ == nullptr) {
            id_ref_ = frames_.size() - 1;  //take last frame in the array
            frames_.erase(frames_.begin(), frames_.end()-1); // 剔除没能没能初始化成功的帧
            f_ref_ = frames_.back();       //更新参考帧
        }

        // append current frame
        frames_.push_back(f_cur);
        std::cout << "frames size: " << frames_.size() << std::endl;
        std::cout << "ref id: " << id_ref_ << std::endl;
        std::cout << "num points in cur: " << f_cur->observations[0].size() << std::endl;
        std::cout << "num points in ref: " << f_ref_->observations[0].size() << std::endl;
        // if the current frames do no have enough features exit
        if (static_cast<int>(f_ref_->observations[0].size()) < kNumMinFeatures_ or
            static_cast<int>(f_cur->observations[0].size()) < kNumMinFeatures_) {
            std::cout << "not enough features" << std::endl;
            return false;
        }

        // find image point matches
        std::vector<int> id;
        std::vector<cv::Point2f> features_cur;
        std::vector<cv::Point2f> features_ref;
        Eigen::aligned_vector<Eigen::Vector4d> dir_normal_cur, dir_normal_ref;
        // 匹配特征点,并进行视差的判断
        std::cout << "begin match_frames: " << std::endl;
        bool disparity_valid = match_frames(f_cur, f_ref_,
                                            id,
                                            features_cur, features_ref,
                                            dir_normal_cur, dir_normal_ref);

        if (not disparity_valid)
            return false;

        std::cout << "├────────" << std::endl;
        std::cout << "initializing frames " << f_cur->t_ns << ", " << f_ref_->t_ns << std::endl;
        std::cout << "match points: " << id.size() << std::endl;

        // 获取相对位姿
        Trc = estimatePose(features_cur, features_ref);
        // TODO: 判断匹配点的个数
        int match_points_num = std::accumulate(mask_match_.begin(), mask_match_.end(), 0);
        std::cout << "num of essential match points: " << match_points_num << std::endl;
        if (match_points_num < 100) {
            return false; // 如何匹配的特征点的少于100个特征点,初始化失败
        }
        std::cout << "├────────" << std::endl;
        std::cout << "begin generate landmark " << std::endl;
        // 生成lanmark
        num_points_added = 0; // landmark 的计数器清空
        TimeCamId tcidl(f_ref_->t_ns, 0);  // target frame id
        TimeCamId tcido(f_cur->t_ns, 0); // host frame id

        float sum_inverse_depth = 1e-5;
        for (size_t i = 0; i < id.size(); i++) {
            if (not mask_match_[i]) // 如果一对匹配点不合法,不在进行三角化
                continue;
            //三角化特征是在当前帧坐标系
            Eigen::Vector4d p0_triangulated =
                triangulate(dir_normal_ref[i].head<3>(), dir_normal_cur[i].head<3>(), Trc);
            // step1: 判断3d点的深度值是否为正,如果深度不合法,返回
            if (not(p0_triangulated.array().isFinite().all() and
                p0_triangulated[3] > 0))
                continue;
            // TODO: 判断3d点在参考帧中的深度是否合法(已经实现)
            Eigen::Vector4d p_cur = Trc.inverse().matrix() * p0_triangulated;
            if (not(p_cur.array().isFinite().all() and p_cur[2] > 0))
                continue;

            // TODO: 视差合法判定暂时屏蔽
            // step2: 判断余弦视差是否合法
            Eigen::Vector3d ref_normal = dir_normal_ref[i].head<3>();
            auto ref_norm = ref_normal.norm();
            Eigen::Vector3d cur_normal = Trc.so3() * dir_normal_cur[i].head<3>();
            auto cur_norm = cur_normal.norm();
            auto cos_parallax = ref_normal.dot(cur_normal) / (ref_norm * cur_norm);
            std::cout << "parallax id: " <<id[i] << ", " << cos_parallax << std::endl;
            if (cos_parallax > cos_parallax_thr_)
                continue;

            // step3:
            //!!!TODO: 重投影误差的判断



            KeypointPosition kpt_pos;

            kpt_pos.kf_id = tcidl;
            kpt_pos.dir = StereographicParam<double>::project(p0_triangulated);
            kpt_pos.id = p0_triangulated[3];
            sum_inverse_depth += kpt_pos.id; // 逆深度求和
            num_points_added++;
            lmdb.addLandmark(id[i], kpt_pos);
            // 为landmark 添加观测, ref, host frame
            KeypointObservation kobs;
            kobs.kpt_id = id[i]; // 3d点的id号
            kobs.pos = f_ref_->observations[0][id[i]].translation().cast<double>();
            lmdb.addObservation(tcidl, kobs); // 增加在当前帧的观测
            kobs.pos = f_cur->observations[0][id[i]].translation().cast<double>();
            lmdb.addObservation(tcido, kobs); // 增加在参考帧的观测

        }

        std::cout << "├────────" << std::endl;
        std::cout << "landmark size: " << lmdb.numLandmarks() << std::endl;
        // 如果生成的landmarks 太少,初始化失败
        if (static_cast<int>(lmdb.numLandmarks()) < kNumMinTriangulatedPoints_) {
            return false;
        }
        // 这一部分借鉴DSO初始化3d点逆深度
        // 求得平均逆深度因子
        float rescaleFactor = 1 / (sum_inverse_depth / num_points_added);
        Trc.translation() /= rescaleFactor; // 平移放缩
        lmdb.scaleLandmarkInverseDepth(rescaleFactor);

        return true;
    }

    // match frames f1 and f2
    // out: a vector of match pairs [ ... [match1i, match2i] ... ]
    // return: 返回平均视差是否大于10个像素
    bool match_frames(
        OpticalFlowResult::Ptr f_cur, OpticalFlowResult::Ptr f_ref,
        std::vector<int> &id,
        std::vector<cv::Point2f> &features1,
        std::vector<cv::Point2f> &features2,
        Eigen::aligned_vector<Eigen::Vector4d> &dir_normal1,
        Eigen::aligned_vector<Eigen::Vector4d> &dir_normal2
    )
    {
        id.clear();
        features1.clear();
        features2.clear();
        dir_normal1.clear();
        dir_normal2.clear();

        float disparity = 0.f;

        // 遍历current frame中的观测
        for (auto &obs1: f_cur->observations[0]) {
            // 在refrence frame 查找是否有相同的观测
            auto it = f_ref->observations[0].find(obs1.first);
            // 查找到相同的观测
            if (it != f_ref->observations[0].end()) {

                const Eigen::Vector2d p0 = obs1.second.translation().cast<double>();
                auto obs2 = *it;
                const Eigen::Vector2d p1 = obs2.second.translation().cast<double>();

                Eigen::Vector4d p0_3d, p1_3d;
                bool valid1 = calib_.intrinsics[0].unproject(p0, p0_3d);
                bool valid2 = calib_.intrinsics[0].unproject(p1, p1_3d);
                if (!valid1 || !valid2) continue;
                // 插入新的特征点
                id.emplace_back(obs1.first);
                dir_normal1.emplace_back(p0_3d);
                dir_normal2.emplace_back(p1_3d);
                // 这里需要将点转化为归一化的坐标
                cv::Point2f point1, point2;
                point1.x = p0_3d(0) / p0_3d(2);
                point1.y = p0_3d(1) / p0_3d(2);

                point2.x = p1_3d(0) / p1_3d(2);
                point2.y = p1_3d(1) / p1_3d(2);

                disparity += (std::abs(point1.x - point2.x) + std::abs(point1.y - point2.y));
                // 插入归一化像素平面的点
                features1.emplace_back(point1);
                features2.emplace_back(point2);
            }
        }
        if (id.size() == 0) // 如果匹配的特征点太少,返回视差不合法
            return false;
        std::cout << "disparity: " << disparity * 640.f / id.size() << std::endl;
        return disparity * 640.f / id.size() > 5;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 相机的配置文件
    basalt::Calibration<double> calib_;
    // 单目初始化的一些配置参数
    float kRansacProb_;
    float kRansacThresholdNormalized_ = 0.0003;
    int kMaxIdDistBetweenFrames_ = 5;
    int kNumMinFeatures_ = 300;
    int kNumMinTriangulatedPoints_ = 100;
    int64_t id_ref_ = 0;

    float cos_parallax_thr_ = 0.9999619230641713; // = cos(0.5deg)
    // 初始化输入的队列
    std::vector<OpticalFlowResult::Ptr> frames_;
    // 参考帧
    OpticalFlowResult::Ptr f_ref_ = nullptr;
    // ransac之后匹配的mask
    std::vector<uchar> mask_match_;

    Sophus::SE3d Trc;

};
class KeypointVoEstimator: public VioEstimatorBase,
                           public BundleAdjustmentBase
{
public:
    typedef std::shared_ptr<KeypointVoEstimator> Ptr;

    static const int N = 9;
    typedef Eigen::Matrix<double, N, 1> VecN;
    typedef Eigen::Matrix<double, N, N> MatNN;
    typedef Eigen::Matrix<double, N, 3> MatN3;

    KeypointVoEstimator(const basalt::Calibration<double> &calib,
                        const VioConfig &config);

    void initialize(int64_t t_ns, const Sophus::SE3d &T_w_i,
                    const Eigen::Vector3d &vel_w_i, const Eigen::Vector3d &bg,
                    const Eigen::Vector3d &ba);

    void initialize(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba);

    virtual ~KeypointVoEstimator()
    { processing_thread->join(); }

    void addIMUToQueue(const ImuData::Ptr &data);
    void addVisionToQueue(const OpticalFlowResult::Ptr &data);

    //-------track module--------//
    Sophus::SE3d trackNewframeCorse(const OpticalFlowResult::Ptr &opt_flow_meas);
    //---------------------------//

    //------backend module-------//
    // 双目版后端优化
    bool measure(const OpticalFlowResult::Ptr &data, bool add_frame);
    // 单目版的后端优化
    bool measure(const OpticalFlowResult::Ptr &data, Sophus::SE3d T_w_i);
    //---------------------------//
    // int64_t propagate();
    // void addNewState(int64_t data_t_ns);

    void marginalize(const std::map<int64_t, int> &num_points_connected);

    void optimize();

    void checkMargNullspace() const;

    int64_t get_t_ns() const
    {
        return frame_states.at(last_state_t_ns).getState().t_ns;
    }
    const Sophus::SE3d &get_T_w_i() const
    {
        return frame_states.at(last_state_t_ns).getState().T_w_i;
    }
    const Eigen::Vector3d &get_vel_w_i() const
    {
        return frame_states.at(last_state_t_ns).getState().vel_w_i;
    }

    const PoseVelBiasState &get_state() const
    {
        return frame_states.at(last_state_t_ns).getState();
    }
    PoseVelBiasState get_state(int64_t t_ns) const
    {
        PoseVelBiasState state;

        auto it = frame_states.find(t_ns);

        if (it != frame_states.end()) {
            return it->second.getState();
        }

        auto it2 = frame_poses.find(t_ns);
        if (it2 != frame_poses.end()) {
            state.T_w_i = it2->second.getPose();
        }

        return state;
    }
    // const MatNN get_cov() const { return cov.bottomRightCorner<N, N>(); }

    void computeProjections(
        std::vector<Eigen::aligned_vector<Eigen::Vector4d>> &res) const;

    inline void setMaxStates(size_t val)
    { max_states = val; }
    inline void setMaxKfs(size_t val)
    { max_kfs = val; }

    Eigen::aligned_vector<Sophus::SE3d> getFrameStates() const
    {
        Eigen::aligned_vector<Sophus::SE3d> res;

        for (const auto &kv : frame_states) {
            res.push_back(kv.second.getState().T_w_i);
        }

        return res;
    }

    Eigen::aligned_vector<Sophus::SE3d> getFramePoses() const
    {
        Eigen::aligned_vector<Sophus::SE3d> res;

        for (const auto &kv : frame_poses) {
            res.push_back(kv.second.getPose());
        }

        return res;
    }

    Eigen::aligned_map<int64_t, Sophus::SE3d> getAllPosesMap() const
    {
        Eigen::aligned_map<int64_t, Sophus::SE3d> res;

        for (const auto &kv : frame_poses) {
            res[kv.first] = kv.second.getPose();
        }

        for (const auto &kv : frame_states) {
            res[kv.first] = kv.second.getState().T_w_i;
        }

        return res;
    }

    const Sophus::SE3d &getT_w_i_init()
    { return T_w_i_init; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    bool take_kf;
    int frames_after_kf;
    std::set<int64_t> kf_ids;

    int64_t last_state_t_ns;
    int64_t llast_state_t_ns;
    // monocular initializer
    MonoInitializer mono_initializer;
    // Input

    Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;

    std::map<int64_t, int> num_points_kf;

    // Marginalization
    AbsOrderMap marg_order;
    Eigen::MatrixXd marg_H;
    Eigen::VectorXd marg_b;

    Eigen::Vector3d gyro_bias_weight, accel_bias_weight;

    size_t max_states;
    size_t max_kfs;

    Sophus::SE3d T_w_i_init;
    // 滑动窗口的数据结构
    // KF1, KF2, KF3, KF4, I2, I1, I0
    Sophus::SE3d T_w_llast; // new add, 用于存储I2帧的位姿,用于运动模型的假设
    bool initialized;
    SLAMStage stage = SLAMStage::NO_IMAGE_YET;
    SensorType sensor = SensorType::Mono;
    VioConfig config;

    double lambda, min_lambda, max_lambda, lambda_vee;

    int64_t msckf_kf_id;

    std::shared_ptr<std::thread> processing_thread;
};
}  // namespace basalt
