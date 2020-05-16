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

#include <basalt/utils/assert.h>
#include <basalt/vi_estimator/keypoint_vo.h>
#include <basalt/vi_estimator/pose_optimize.h>
#include <basalt/optimization/accumulator.h>


#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <chrono>

namespace basalt
{

KeypointVoEstimator::KeypointVoEstimator(
    const basalt::Calibration<double> &calib, const VioConfig &config)
    : take_kf(true),
      frames_after_kf(0),
      mono_initializer(calib), // 单目初始化class
      initialized(false),
      config(config),
      lambda(config.vio_lm_lambda_min),
      min_lambda(config.vio_lm_lambda_min),
      max_lambda(config.vio_lm_lambda_max),
      lambda_vee(2)
{
    if (calib.intrinsics.size() > 1)
        sensor = SensorType::Stereo;
    else
        sensor = SensorType::Mono;
    this->obs_std_dev = config.vio_obs_std_dev;
    this->huber_thresh = config.vio_obs_huber_thresh;
    this->calib = calib;

    // Setup marginalization
    marg_H.setZero(POSE_SIZE, POSE_SIZE);
    marg_b.setZero(POSE_SIZE);

    // prior on pose
    marg_H.diagonal().setConstant(config.vio_init_pose_weight);

    std::cout << "marg_H\n" << marg_H << std::endl;

    gyro_bias_weight = calib.gyro_bias_std.array().square().inverse();
    accel_bias_weight = calib.accel_bias_std.array().square().inverse();

    max_states = config.vio_max_states;
    max_kfs = config.vio_max_kfs;

    vision_data_queue.set_capacity(10);
    imu_data_queue.set_capacity(300);
}

void KeypointVoEstimator::initialize(int64_t t_ns, const Sophus::SE3d &T_w_i,
                                     const Eigen::Vector3d &vel_w_i,
                                     const Eigen::Vector3d &bg,
                                     const Eigen::Vector3d &ba)
{
    UNUSED(vel_w_i);
    UNUSED(bg);
    UNUSED(ba);

    initialized = true;
    T_w_i_init = T_w_i;

    llast_state_t_ns = last_state_t_ns = t_ns;
    frame_poses[t_ns] = PoseStateWithLin(t_ns, T_w_i, true);

    marg_order.abs_order_map[t_ns] = std::make_pair(0, POSE_SIZE);
    marg_order.total_size = POSE_SIZE;
    marg_order.items = 1;

    initialize(bg, ba);
}

void KeypointVoEstimator::initialize(const Eigen::Vector3d &bg,
                                     const Eigen::Vector3d &ba)
{
    auto proc_func = [&, bg, ba]
    {
        // 初始化函数
        OpticalFlowResult::Ptr prev_frame, curr_frame;
        bool add_pose = false;
        while (true) {
            vision_data_queue.pop(curr_frame);

            if (config.vio_enforce_realtime) {
                // drop current frame if another frame is already in the queue.
                while (!vision_data_queue.empty()) vision_data_queue.pop(curr_frame);
            }

            if (!curr_frame.get()) {
                break;
            }

            // Correct camera time offset
            // curr_frame->t_ns += calib.cam_time_offset_ns;

            while (!imu_data_queue.empty()) {
                ImuData::Ptr d;
                imu_data_queue.pop(d);
            }

            if (!initialized) {
                // 双目初始化
                if (sensor == SensorType::Stereo) {
                    Eigen::Vector3d vel_w_i_init;

                    // 将当前帧加入到frame_poses中,同时将其边缘化
                    llast_state_t_ns = last_state_t_ns = curr_frame->t_ns;

                    frame_poses[last_state_t_ns] =
                        PoseStateWithLin(last_state_t_ns, T_w_i_init, true);

                    marg_order.abs_order_map[last_state_t_ns] =
                        std::make_pair(0, POSE_SIZE);
                    marg_order.total_size = POSE_SIZE;
                    marg_order.items = 1;

                    std::cout << "Setting up filter: t_ns " << last_state_t_ns << std::endl;
                    std::cout << "T_w_i\n" << T_w_i_init.matrix() << std::endl;

                    initialized = true;
                }// 单目初始化
                else {
                    if (stage == SLAMStage::NO_IMAGE_YET) {
                        std::cout << "mono first frame " << std::endl;
                        mono_initializer.init(curr_frame);
                        stage = SLAMStage::NOT_INITIALIZED;
                        continue;
                    }
                    if (stage == SLAMStage::NOT_INITIALIZED) {
                        // 三角化生成地图点
                        std::cout << "mono not initialized " << std::endl;
                        LandmarkDatabase lmdb; // landmark database
                        int num_points_added;
                        bool initial_valid = mono_initializer.initialize(curr_frame, lmdb, num_points_added);
                        if (initial_valid) {
                            stage = SLAMStage::OK;
                            std::cout << "mono initialized success" << std::endl;

                            //TODO:1 这一块需要修改,我们需要将参考帧加入到frame_poses,同时将其边缘化
                            llast_state_t_ns = last_state_t_ns = mono_initializer.f_ref_->t_ns;
                            frame_poses[last_state_t_ns] =
                                PoseStateWithLin(last_state_t_ns, T_w_i_init, true);

                            prev_opt_flow_res[last_state_t_ns] = mono_initializer.f_ref_;

                            marg_order.abs_order_map[last_state_t_ns] =
                                std::make_pair(0, POSE_SIZE);
                            marg_order.total_size = POSE_SIZE;
                            marg_order.items = 1;

                            std::cout << "initialize first frame: t_ns " << last_state_t_ns << std::endl;
                            std::cout << "T_w_ref\n" << T_w_i_init.matrix() << std::endl;

                            // TODO:2 将当前帧加入到frame_poses中
                            last_state_t_ns = curr_frame->t_ns;
                            prev_opt_flow_res[last_state_t_ns] = curr_frame;
                            frame_poses[last_state_t_ns] =
                                PoseStateWithLin(last_state_t_ns, mono_initializer.Trc, false);
                            std::cout << "initialize second frame: t_ns " << last_state_t_ns << std::endl;
                            std::cout << "T_w_cur\n" << mono_initializer.Trc.matrix() << std::endl;
                            // TODO:3 将landmark 插入数据数据中
                            this->lmdb = lmdb;

                            // TODO:4 update num_points_kf
                            num_points_kf[llast_state_t_ns] = this->lmdb.numLandmarks();
                            kf_ids.emplace(mono_initializer.f_ref_->t_ns);
                            frames_after_kf = 1;
                            initialized = true;
                            // TODO: 将运行的结果输出到可视化队列
                            if (out_vis_queue) {
                                //
                                VioVisualizationData::Ptr data(new VioVisualizationData);

                                data->t_ns = last_state_t_ns;

                                BASALT_ASSERT(frame_states.empty());

                                for (const auto &kv : frame_poses) {
                                    data->frames.emplace_back(kv.second.getPose());
                                }

                                get_current_points(data->points, data->point_ids);

                                data->projections.resize(curr_frame->observations.size());
                                computeProjections(data->projections);

                                data->opt_flow_res = prev_opt_flow_res[last_state_t_ns];

                                out_vis_queue->push(data);
                            }// out_vis_queue over

                            std::cout << "out_vis_queue success " << std::endl;
                        }// monocular initial success
                    }// condition: NOT_INITIALIZED
                }//Condition: SensorType Mono
            } // initialize success

            //如果初始化成功,开始追踪和优化
            if (initialized) {
                if (sensor == SensorType::Stereo) {
                    if (prev_frame) {
                        add_pose = true;
                    }
                    if(frame_poses.size()>1)
                    {
                        std::cout << "track module" << std::endl;
                        Sophus::SE3d T_w_new = trackNewframeCorse(curr_frame);
                        std::cout << "backend modlue" << std::endl;
                        measure(curr_frame, T_w_new);
                    }
                    else{
                        std::cout << "backend modlue" << std::endl;
                        measure(curr_frame, add_pose);
                    }
                }
                else {
                    // condition: 为了简化程序的流程,单目初始化完,不需要进行追踪和后端优化
                    if (last_state_t_ns != curr_frame->t_ns) {
                        std::cout << "track module" << std::endl;
                        Sophus::SE3d T_w_new = trackNewframeCorse(curr_frame);
                        std::cout << "backend modlue" << std::endl;
                        measure(curr_frame, T_w_new);
                    }

                }

            }

            prev_frame = curr_frame; //update last frame
        }

        if (out_vis_queue) out_vis_queue->push(nullptr);
        if (out_marg_queue) out_marg_queue->push(nullptr);
        if (out_state_queue) out_state_queue->push(nullptr);

        finished = true;

        std::cout << "Finished VIOFilter " << std::endl;
    };

    processing_thread.reset(new std::thread(proc_func));
}

void KeypointVoEstimator::addVisionToQueue(const OpticalFlowResult::Ptr &data)
{
    vision_data_queue.push(data);
}

/**
 * \biref 追踪上一帧来获得当前帧的初始位姿
 * @param opt_flow_meas
 * @return
 */
Sophus::SE3d KeypointVoEstimator::trackNewframeCorse(const OpticalFlowResult::Ptr &opt_flow_meas)
{

    //step1: 追踪上一帧,
    /*****method1*******/
    //step1.1 获得上一帧的的位姿
    std::cout << "last frame id: " << last_state_t_ns << std::endl;
    Sophus::SE3d state_t = frame_poses.at(last_state_t_ns).getPose(); //上一时刻的状态
    TimeCamId tcid_t(last_state_t_ns, 0); // 上一帧的id
    // recent frame I0, I1之间共视的3d点
    Eigen::aligned_unordered_map<int, KeypointPosition> convisible_landmark;

    // TODO: 检索共视的3d点的代码应该没有问题
    // 遍历当前帧cam0 的观测

    for (const auto &kv_obs: opt_flow_meas->observations[0]) {
        int kpt_id = kv_obs.first; // get feature id

        // 确定特征点对应的地图点是否存在
        if (lmdb.landmarkExists(kpt_id)) {
            // 确定特征点是否在上一帧是否有观测
            auto last_obs_it = prev_opt_flow_res[last_state_t_ns]->observations[0].find(kpt_id);
            if (last_obs_it != prev_opt_flow_res[last_state_t_ns]->observations[0].end()) {
                const auto &lm = lmdb.getLandmark(kpt_id);
                if (lm.kf_id.frame_id == last_state_t_ns) // 如果当前帧的id等于上一帧图片, 即上一帧是关键帧
                {
                    convisible_landmark[kpt_id] = lm;
                }
                else {
                    //获得3d点主导帧的位姿
                    Sophus::SE3d state_h = frame_poses.at(lm.kf_id.frame_id).getPose();
                    KeypointPosition lm_last_frame;
                    //step1 更换landmark的id
                    lm_last_frame.kf_id.frame_id = last_state_t_ns;
                    lm_last_frame.kf_id.cam_id = 0;
                    //step2 更换landmarkid 和dir
                    //step2.1 将landmark的dir转化为3d的方位向量
                    Eigen::Vector4d p_h_3d;
                    p_h_3d = StereographicParam<double>::unproject(lm.dir);
                    // 求得3dhomogeneous
                    p_h_3d[3] = lm.id;

                    Sophus::SE3d T_t_h_sophus;
                    T_t_h_sophus.so3() = state_t.so3().inverse() * state_h.so3();
                    T_t_h_sophus.translation() =
                        state_t.so3().inverse() * (state_h.translation() - state_t.translation());
                    // 将3d点变换到I1帧
                    Eigen::Vector4d p_t_3d = T_t_h_sophus.matrix() * p_h_3d;
                    // 3d点的归一化
                    p_t_3d /= p_t_3d.template head<3>().norm();
                    // 将3d点变换为用dir和id表示
                    lm_last_frame.dir = StereographicParam<double>::project(p_t_3d);
                    lm_last_frame.id = p_t_3d[3];
                    convisible_landmark[kpt_id] = lm_last_frame;
                }
            }
        }
    }
    //condition1: 如果当前帧和上一帧共视的3d点太少
    if (convisible_landmark.size() < 50) {
        std::cout << "the convisible landmarks between current frame \n "
                     "and last frame is too less: " << convisible_landmark.size() << std::endl;
    }
    std::cout << "convisible_landmark size: " << convisible_landmark.size() << std::endl;
    //TODO: 这里需要参考dso运动模型假设
    //step2.1: 构造两帧之间的优化器
    pose_optimizer Optimizer(convisible_landmark, opt_flow_meas, calib);
    //step2.2 尝试不同的运动模型假设, 选择其中重投影误差最小的模型,KITTI 好像是匀速运动模型
    Eigen::aligned_vector<Sophus::SE3d> motion_model;
    motion_model.push_back(Sophus::SE3d());                                // asssume zero motion model
    if (llast_state_t_ns != last_state_t_ns) {
        //        Sophus::SE3d llstate_t = frame_poses.at(llast_state_t_ns).getPose(); //上上一时刻的状态
        Sophus::SE3d llstate_t = T_w_llast;                                  //上上一时刻的状态
        Sophus::SE3d velocity = state_t.inverse() * llstate_t;               // T_l_ll
        motion_model.push_back(velocity);                                    // assume constant motion
        motion_model.push_back(velocity * velocity);                         // assume double motion (frame skipped)
        motion_model.push_back(Sophus::SE3d::exp(velocity.log() * 0.5));     // assume half motion model
        for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta += 0.01) {
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, 0, 0),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, 0, rotDelta, 0),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, 0, 0, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, 0, 0),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, 0, -rotDelta, 0),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, 0, 0, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, rotDelta, 0),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, 0, rotDelta, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, 0, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, rotDelta, 0),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, 0, -rotDelta, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, 0, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, -rotDelta, 0),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, 0, rotDelta, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, 0, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, -rotDelta, 0),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, 0, -rotDelta, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, 0, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
            motion_model.push_back(Sophus::SE3d(Eigen::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                                                Eigen::Vector3d(0, 0, 0)) * velocity);
        }
    }

    int best_motionmodel_index = -1;
    double min_error = std::numeric_limits<double>::max();
    for (size_t i = 0; i < motion_model.size(); i++) {
        double error = 0;
        Optimizer.computeError(motion_model[i], error);
        if (error < min_error) {
            min_error = error;
            best_motionmodel_index = i;
        }
    }
    switch (best_motionmodel_index) {
    case 0:std::cout << "zero motion" << std::endl;
        break;
    case 1:std::cout << "constant motion" << std::endl;
        break;
    case 2:std::cout << "double motion" << std::endl;
        break;
    case 3:std::cout << "half motion" << std::endl;
        break;
    default:std::cout << "other motion model" << std::endl;
        break;
    }
    //step2.3 选择最好的一个初值,开始迭代求解
    Sophus::SE3d T_cur_ref = Optimizer.optimize(motion_model[best_motionmodel_index]);
    double error = 0.;
    Optimizer.computeError(T_cur_ref, error);
    std::cout << "tracking error: " << error << std::endl;
    //step2.4 将结果赋值给最新帧的位姿
    /********method1**********/
    /********method2**********/
    //PNP
    /********method2**********/
    // TODO: bug
    return state_t * T_cur_ref.inverse();
}
bool KeypointVoEstimator::measure(const OpticalFlowResult::Ptr &opt_flow_meas,
                                  const bool add_pose)
{
    // TODO: 这一块需要修改,需要需改为使用PnP求解当前帧的初始位姿
    if (add_pose) {
        const PoseStateWithLin &curr_state = frame_poses.at(last_state_t_ns); //上一时刻的状态

        llast_state_t_ns = last_state_t_ns;  // 更新 ll frame 的id
        last_state_t_ns = opt_flow_meas->t_ns; //更新 last_frame id

        // 用上一帧的位姿初始化下一帧的位姿
        PoseStateWithLin next_state(opt_flow_meas->t_ns, curr_state.getPose());
        frame_poses[last_state_t_ns] = next_state;
    }

    for (auto frame: frame_poses) {
        std::cout << "frame id: " << frame.first << std::endl;
    }
    std::cout << "llast frame: " << llast_state_t_ns
              << "\n last frame: " << last_state_t_ns << std::endl;

    // save results
    prev_opt_flow_res[opt_flow_meas->t_ns] = opt_flow_meas;

    // Make new residual for existing keypoints
    int connected0 = 0;
    std::map<int64_t, int> num_points_connected;
    std::unordered_set<int> unconnected_obs0;
    for (size_t i = 0; i < opt_flow_meas->observations.size(); i++) {
        TimeCamId tcid_target(opt_flow_meas->t_ns, i);

        for (const auto &kv_obs : opt_flow_meas->observations[i]) {
            int kpt_id = kv_obs.first;

            if (lmdb.landmarkExists(kpt_id)) {
                const TimeCamId &tcid_host = lmdb.getLandmark(kpt_id).kf_id;

                KeypointObservation kobs;
                kobs.kpt_id = kpt_id;
                kobs.pos = kv_obs.second.translation().cast<double>();

                lmdb.addObservation(tcid_target, kobs);
                // obs[tcid_host][tcid_target].push_back(kobs);

                if (num_points_connected.count(tcid_host.frame_id) == 0) {
                    num_points_connected[tcid_host.frame_id] = 0;
                }
                num_points_connected[tcid_host.frame_id]++;

                if (i == 0) connected0++;
            }
            else {
                if (i == 0) {
                    unconnected_obs0.emplace(kpt_id);
                }
            }
        }
    }
    //TODO: 利用connected0 来求解当前帧的位姿

    if (double(connected0) / (connected0 + unconnected_obs0.size()) <
        config.vio_new_kf_keypoints_thresh &&
        frames_after_kf > config.vio_min_frames_after_kf)
        take_kf = true;

    if (config.vio_debug) {
        std::cout << "connected0 " << connected0 << " unconnected0 "
                  << unconnected_obs0.size() << std::endl;
    }

    if (take_kf) {
        // Triangulate new points from stereo and make keyframe for camera 0
        take_kf = false;
        frames_after_kf = 0;
        kf_ids.emplace(last_state_t_ns);

        TimeCamId tcidl(opt_flow_meas->t_ns, 0);

        int num_points_added = 0;
        for (int lm_id : unconnected_obs0) {
            // Find all observations
            std::map<TimeCamId, KeypointObservation> kp_obs;

            for (const auto &kv : prev_opt_flow_res) {
                for (size_t k = 0; k < kv.second->observations.size(); k++) {
                    auto it = kv.second->observations[k].find(lm_id);
                    if (it != kv.second->observations[k].end()) {
                        TimeCamId tcido(kv.first, k);

                        KeypointObservation kobs;
                        kobs.kpt_id = lm_id;
                        kobs.pos = it->second.translation().cast<double>();

                        // obs[tcidl][tcido].push_back(kobs);
                        kp_obs[tcido] = kobs;
                    }
                }
            }

            // triangulate
            bool valid_kp = false;
            const double min_triang_distance2 =
                config.vio_min_triangulation_dist * config.vio_min_triangulation_dist;
            for (const auto &kv_obs : kp_obs) {
                if (valid_kp) break;
                TimeCamId tcido = kv_obs.first;

                const Eigen::Vector2d p0 = opt_flow_meas->observations.at(0)
                    .at(lm_id)
                    .translation()
                    .cast<double>();
                const Eigen::Vector2d p1 = prev_opt_flow_res[tcido.frame_id]
                    ->observations[tcido.cam_id]
                    .at(lm_id)
                    .translation()
                    .cast<double>();

                Eigen::Vector4d p0_3d, p1_3d;
                bool valid1 = calib.intrinsics[0].unproject(p0, p0_3d);
                bool valid2 = calib.intrinsics[tcido.cam_id].unproject(p1, p1_3d);
                if (!valid1 || !valid2) continue;

                Sophus::SE3d T_i0_i1 =
                    getPoseStateWithLin(tcidl.frame_id).getPose().inverse() *
                        getPoseStateWithLin(tcido.frame_id).getPose();
                Sophus::SE3d T_0_1 =
                    calib.T_i_c[0].inverse() * T_i0_i1 * calib.T_i_c[tcido.cam_id];

                if (T_0_1.translation().squaredNorm() < min_triang_distance2) continue;

                Eigen::Vector4d p0_triangulated =
                    triangulate(p0_3d.head<3>(), p1_3d.head<3>(), T_0_1);
                //TODO: 需要增加视差判定
                constexpr double cos_parallax_thr_ = 0.9999619230641713;
                Eigen::Vector3d ref_normal = T_0_1.so3() * p1_3d.head<3>();
                auto ref_norm = ref_normal.norm();
                Eigen::Vector3d cur_normal = p0_3d.head<3>();
                auto cur_norm = cur_normal.norm();
                auto cos_parallax = ref_normal.dot(cur_normal) / (ref_norm * cur_norm);
                if (cos_parallax > cos_parallax_thr_)
                    continue;
                //TODO: 需要增加远深度的判定
                if (p0_triangulated.array().isFinite().all() &&
                    p0_triangulated[3] > 0 && p0_triangulated[3] < 2.50) {
                    KeypointPosition kpt_pos;
                    kpt_pos.kf_id = tcidl;
                    kpt_pos.dir = StereographicParam<double>::project(p0_triangulated);
                    kpt_pos.id = p0_triangulated[3];
                    lmdb.addLandmark(lm_id, kpt_pos);

                    num_points_added++;
                    valid_kp = true;
                }
            }

            if (valid_kp) {
                for (const auto &kv_obs : kp_obs) {
                    lmdb.addObservation(kv_obs.first, kv_obs.second);
                }
            }
        }

        num_points_kf[opt_flow_meas->t_ns] = num_points_added;
    }
    else {
        frames_after_kf++;
    }

    optimize();
    marginalize(num_points_connected);

    if (out_state_queue) {
        const PoseStateWithLin &p = frame_poses.at(last_state_t_ns);

        PoseVelBiasState::Ptr data(
            new PoseVelBiasState(p.getT_ns(), p.getPose(), Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));

        out_state_queue->push(data);
    }

    if (out_vis_queue) {
        VioVisualizationData::Ptr data(new VioVisualizationData);

        data->t_ns = last_state_t_ns;

        BASALT_ASSERT(frame_states.empty());

        for (const auto &kv : frame_poses) {
            data->frames.emplace_back(kv.second.getPose());
        }

        get_current_points(data->points, data->point_ids);

        data->projections.resize(opt_flow_meas->observations.size());
        computeProjections(data->projections);

        data->opt_flow_res = prev_opt_flow_res[last_state_t_ns];

        out_vis_queue->push(data);
    }

    last_processed_t_ns = last_state_t_ns;

    return true;
}
/**
 * \brief monocular backend
 * 包含3个部分:
 *     1 生成关键帧(三角化一些特征点)
 *     2 优化(重投影因子和边缘化因子)
 *     3 边缘化
 * @param opt_flow_meas, 当前帧的测量
 * @param T_w_i 给当前帧设置的初始位姿
 * @return always return true
 */
bool KeypointVoEstimator::measure(const OpticalFlowResult::Ptr &opt_flow_meas, Sophus::SE3d T_w_i)
{
    // TODO: 这一块需要修改, 需要需改为使用PnP求解当前帧的初始位姿,或者使用优化的方式获得当前帧的初始位姿

    llast_state_t_ns = last_state_t_ns;  // 更新 last last frame 的id
    last_state_t_ns = opt_flow_meas->t_ns; // 更新last last frame的 id
    // 使用T_w_i 初始化最新帧的位姿
    std::cout << "new frame: " << last_state_t_ns << std::endl;
    std::cout << "last frame: " << llast_state_t_ns << std::endl;

    PoseStateWithLin next_state(opt_flow_meas->t_ns, T_w_i);
    frame_poses[last_state_t_ns] = next_state;


    // save results
    prev_opt_flow_res[opt_flow_meas->t_ns] = opt_flow_meas;

    // Make new residual for existing keypoints
    int connected0 = 0;
    std::map<int64_t, int> num_points_connected;
    std::unordered_set<int> unconnected_obs0;
    for (size_t i = 0; i < opt_flow_meas->observations.size(); i++) {
        TimeCamId tcid_target(opt_flow_meas->t_ns, i);

        for (const auto &kv_obs : opt_flow_meas->observations[i]) {
            int kpt_id = kv_obs.first;

            if (lmdb.landmarkExists(kpt_id)) {
                const TimeCamId &tcid_host = lmdb.getLandmark(kpt_id).kf_id;

                KeypointObservation kobs;
                kobs.kpt_id = kpt_id;
                kobs.pos = kv_obs.second.translation().cast<double>();

                lmdb.addObservation(tcid_target, kobs);
                // obs[tcid_host][tcid_target].push_back(kobs);

                if (num_points_connected.count(tcid_host.frame_id) == 0) {
                    num_points_connected[tcid_host.frame_id] = 0;
                }
                num_points_connected[tcid_host.frame_id]++;

                if (i == 0) connected0++;
            }
            else {
                if (i == 0) {
                    unconnected_obs0.emplace(kpt_id);
                }
            }
        }
    }
    //TODO: 利用connected0 来求解当前帧的位姿

    if (double(connected0) / (connected0 + unconnected_obs0.size()) <
        config.vio_new_kf_keypoints_thresh &&
        frames_after_kf > config.vio_min_frames_after_kf)
        take_kf = true;

    if (config.vio_debug) {
        std::cout << "connected0 " << connected0 << " unconnected0 "
                  << unconnected_obs0.size() << std::endl;
    }

    if (take_kf) {
        // Triangulate new points from stereo and make keyframe for camera 0
        take_kf = false;
        frames_after_kf = 0;
        kf_ids.emplace(last_state_t_ns);

        TimeCamId tcidl(opt_flow_meas->t_ns, 0);

        int num_points_added = 0;
        for (int lm_id : unconnected_obs0) {
            // Find all observations
            std::map<TimeCamId, KeypointObservation> kp_obs;

            for (const auto &kv : prev_opt_flow_res) {
                for (size_t k = 0; k < kv.second->observations.size(); k++) {
                    auto it = kv.second->observations[k].find(lm_id);
                    if (it != kv.second->observations[k].end()) {
                        TimeCamId tcido(kv.first, k);

                        KeypointObservation kobs;
                        kobs.kpt_id = lm_id;
                        kobs.pos = it->second.translation().cast<double>();

                        // obs[tcidl][tcido].push_back(kobs);
                        kp_obs[tcido] = kobs;
                    }
                }
            }

            // triangulate
            bool valid_kp = false;
            const double min_triang_distance2 =
                config.vio_min_triangulation_dist * config.vio_min_triangulation_dist;
            for (const auto &kv_obs : kp_obs) {
                if (valid_kp) break;
                TimeCamId tcido = kv_obs.first;

                const Eigen::Vector2d p0 = opt_flow_meas->observations.at(0)
                    .at(lm_id)
                    .translation()
                    .cast<double>();
                const Eigen::Vector2d p1 = prev_opt_flow_res[tcido.frame_id]
                    ->observations[tcido.cam_id]
                    .at(lm_id)
                    .translation()
                    .cast<double>();

                Eigen::Vector4d p0_3d, p1_3d;
                bool valid1 = calib.intrinsics[0].unproject(p0, p0_3d);
                bool valid2 = calib.intrinsics[tcido.cam_id].unproject(p1, p1_3d);
                if (!valid1 || !valid2) continue;

                Sophus::SE3d T_i0_i1 =
                    getPoseStateWithLin(tcidl.frame_id).getPose().inverse() *
                        getPoseStateWithLin(tcido.frame_id).getPose();
                Sophus::SE3d T_0_1 =
                    calib.T_i_c[0].inverse() * T_i0_i1 * calib.T_i_c[tcido.cam_id];

                if (T_0_1.translation().squaredNorm() < min_triang_distance2) continue;

                Eigen::Vector4d p0_triangulated =
                    triangulate(p0_3d.head<3>(), p1_3d.head<3>(), T_0_1);

                if (p0_triangulated.array().isFinite().all() &&
                    p0_triangulated[3] > 0 && p0_triangulated[3] < 3.0) {
                    KeypointPosition kpt_pos;
                    kpt_pos.kf_id = tcidl;
                    kpt_pos.dir = StereographicParam<double>::project(p0_triangulated);
                    kpt_pos.id = p0_triangulated[3];
                    lmdb.addLandmark(lm_id, kpt_pos);

                    num_points_added++;
                    valid_kp = true;
                }
            }

            if (valid_kp) {
                for (const auto &kv_obs : kp_obs) {
                    lmdb.addObservation(kv_obs.first, kv_obs.second);
                }
            }
        }

        num_points_kf[opt_flow_meas->t_ns] = num_points_added;
    }
    else {
        frames_after_kf++;
    }

    optimize();
    marginalize(num_points_connected);

    if (out_state_queue) {
        const PoseStateWithLin &p = frame_poses.at(last_state_t_ns);

        PoseVelBiasState::Ptr data(
            new PoseVelBiasState(p.getT_ns(), p.getPose(), Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));

        out_state_queue->push(data);
    }

    if (out_vis_queue) {
        VioVisualizationData::Ptr data(new VioVisualizationData);

        data->t_ns = last_state_t_ns;

        BASALT_ASSERT(frame_states.empty());

        for (const auto &kv : frame_poses) {
            data->frames.emplace_back(kv.second.getPose());
        }

        get_current_points(data->points, data->point_ids);

        data->projections.resize(opt_flow_meas->observations.size());
        computeProjections(data->projections);

        data->opt_flow_res = prev_opt_flow_res[last_state_t_ns];

        out_vis_queue->push(data);
    }

    last_processed_t_ns = last_state_t_ns;

    return true;
}

void KeypointVoEstimator::checkMargNullspace() const
{
    checkNullspace(marg_H, marg_b, marg_order, frame_states, frame_poses);
}

void KeypointVoEstimator::marginalize(
    const std::map<int64_t, int> &num_points_connected)
{
    BASALT_ASSERT(frame_states.empty());
    // Marginalize
    AbsOrderMap aom;
    // TODO:
    T_w_llast = frame_poses.at(llast_state_t_ns).getPose();
    //remove all frames_poses that are not kfs and not the current frame
    std::set<int64_t> non_kf_poses;
    for (const auto &kv : frame_poses) {
        if (kf_ids.count(kv.first) == 0 and kv.first != last_state_t_ns) {
            non_kf_poses.emplace(kv.first);
        }
        else {
            aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);
            // Check that we have the same order as marginalization
            if (marg_order.abs_order_map.count(kv.first) > 0) {
                BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                    aom.abs_order_map.at(kv.first));
            }
            aom.total_size += POSE_SIZE;
            aom.items++;
        }
    }

    // 次新帧不是关键帧直接扔掉
    for (int64_t id : non_kf_poses) {
        frame_poses.erase(id); // 剔除非关键帧的位姿
        lmdb.removeFrame(id);  // 剔除landmark在非关键帧的观测
        prev_opt_flow_res.erase(id);
    }

    auto kf_ids_all = kf_ids;
    std::set<int64_t> kfs_to_marg;
    // 如果关键帧的数量大多, 需要选择需要被边缘化的关键帧, 最新帧是关键帧
    while (kf_ids.size() > max_kfs) {
        int64_t id_to_marg = -1;
        //method1: 通过共视关系剔除一些关键帧
        {
            std::vector<int64_t> ids;
            for (int64_t id : kf_ids) {
                ids.push_back(id);
            }
            // 与最新帧共视关系最差的帧被选择边缘化
            for (size_t i = 0; i < ids.size() - 2; i++) {
                if (num_points_connected.count(ids[i]) == 0 or
                    (num_points_connected.at(ids[i]) / num_points_kf.at(ids[i]) < 0.05)) {
                    id_to_marg = ids[i];
                    break;
                }
            }
        }//method1
        //method2: distance schedule
        if (id_to_marg < 0) {
            std::vector<int64_t> ids;
            for (int64_t id: kf_ids) {
                ids.push_back(id);
            }

            int64_t last_kf = *kf_ids.crbegin();
            double min_score = std::numeric_limits<double>::max();
            int64_t min_score_id = -1;
            // 最新的两个关键帧不会被剔除
            for (size_t i = 0; i < ids.size() - 2; i++) {
                double denom = 0;
                for (size_t j = 0; j < ids.size() - 2; j++) {
                    if (i == j)
                        continue;
                    denom += 1 / ((frame_poses.at(ids[i]).getPose().translation() -
                        frame_poses.at(ids[j]).getPose().translation()).norm() +
                        1e-5);
                }

                double score =
                    std::sqrt((frame_poses.at(ids[i]).getPose().translation() -
                        frame_poses.at(last_kf).getPose().translation()).norm()) * denom;
                if (score < min_score) {
                    min_score_id = ids[i];
                    min_score = score;
                }
            }
            id_to_marg = min_score_id;
        }// method2

        kfs_to_marg.emplace(id_to_marg);
        non_kf_poses.emplace(id_to_marg);

        kf_ids.erase(id_to_marg);
    }
//    std::cout << "marg order" << std::endl;
//    aom.print_order();
//
//    std::cout << "marg prior order" << std::endl;
//    marg_order.print_order();
    if (config.vio_debug) {
        std::cout << "non_kf_poses.size() " << non_kf_poses.size() << std::endl;
        for (const auto &v: non_kf_poses) std::cout << v << ' ';
        std::cout << std::endl;

        std::cout << "kfs_to_marg.size() " << kfs_to_marg.size() << std::endl;
        for (const auto &v: kfs_to_marg) std::cout << v << ' ';
        std::cout << std::endl;

        std::cout << "latest_state_t_ns: " << last_state_t_ns << std::endl;

        for (const auto &v: frame_poses) std::cout << v.first << ' ';
        std::cout << std::endl;
    }
    //condition: 如果关键帧的数目太多,那么需要边缘化操作
    if (!kfs_to_marg.empty()) {
        //Marginalize only if latest state is keyframe
        //边缘化关键帧时,一定是因为关键帧的数目太多导致的
        BASALT_ASSERT(kf_ids_all.count(last_state_t_ns) > 0);

        size_t asize = aom.total_size;
        double marg_prior_error;

        DenseAccumulator accum;
        accum.reset(asize);

        // visual projection error factor
        {
            // Linearize points
            Eigen::aligned_map<
                TimeCamId,
                Eigen::aligned_map<TimeCamId,
                                   Eigen::aligned_vector<KeypointObservation>>>
                obs_to_lin;

            for (auto it = lmdb.getObservations().cbegin();
                 it != lmdb.getObservations().cend();) {
                if (kfs_to_marg.count(it->first.frame_id) > 0) {
                    for (auto it2 = it->second.cbegin(); it2 != it->second.cend(); ++it2) {
                        obs_to_lin[it->first].emplace(*it2);
                    }
                }
                ++it;
            }

            double rld_error;
            Eigen::aligned_vector<RelLinData> rld_vec;
            linearizeHelper(rld_vec, obs_to_lin, rld_error);

            for (auto &rld : rld_vec) {
                rld.invert_keypoint_hessians();

                Eigen::MatrixXd rel_H;
                Eigen::VectorXd rel_b;

                linearizeRel(rld, rel_H, rel_b);
                linearizeAbs(rel_H, rel_b, rld, aom, accum);
            }
        }// visual projection error factor
        // add marginalization prior
        linearizeMargPrior(marg_order, marg_H, marg_b, aom,
                           accum.getH(), accum.getB(), marg_prior_error);
        // add marginalization prior end
        if (out_marg_queue and not kfs_to_marg.empty()) {
            MargData::Ptr m(new MargData);
            m->aom = aom;
            m->abs_H = accum.getH();
            m->abs_b = accum.getB();
            m->frame_poses = frame_poses;
            m->frame_states = frame_states;
            m->kfs_all = kf_ids_all;
            m->kfs_to_marg = kfs_to_marg;
            m->use_imu = false;

            for (int64_t t: m->kfs_all) {
                m->opt_flow_res.emplace_back(prev_opt_flow_res.at(t));
            }

            out_marg_queue->push(m);
        }
        //
        std::set<int> idx_to_keep, idx_to_marg;
        for (const auto &kv : aom.abs_order_map) {
            if (kv.second.second == POSE_SIZE) {
                int start_idx = kv.second.first;
                if (kfs_to_marg.count(kv.first) == 0) {
                    for (size_t i = 0; i < POSE_SIZE; i++) {
                        idx_to_keep.emplace(start_idx + i);
                    }
                }
                else {
                    for (size_t i = 0; i < POSE_SIZE; i++) {
                        idx_to_marg.emplace(start_idx + i);
                    }
                }
            }
            else {
                BASALT_ASSERT(false);
            }
        }

        if (config.vio_debug) {
            std::cout << "keeping " << idx_to_keep.size() << " marg "
                      << idx_to_marg.size() << " total " << asize << std::endl;
            std::cout << "frame_poses: " << frame_poses.size() << "frame_sates "
                      << frame_states.size() << std::endl;
        }

        Eigen::MatrixXd marg_H_new;
        Eigen::VectorXd marg_b_new;

        marginalizeHelper(accum.getH(), accum.getB(), idx_to_keep, idx_to_marg,
                          marg_H_new, marg_b_new);

        // set linear point
        for (auto &kv: frame_poses) {
            if (!kv.second.isLinearized())
                kv.second.setLinTrue();
        }
        // erase keyframe state and measurement
        for (const int64_t id: kfs_to_marg) {
            frame_poses.erase(id);
            prev_opt_flow_res.erase(id);
        }
        // 扔掉主导帧主导的3d点, 剔除3d点在被边缘化关键帧的观测
        lmdb.removeKeyframes(kfs_to_marg, kfs_to_marg, kfs_to_marg);
        AbsOrderMap marg_order_new;
        for (const auto &kv : frame_poses) {
            marg_order_new.abs_order_map[kv.first] =
                std::make_pair(marg_order_new.total_size, POSE_SIZE);

            marg_order_new.total_size += POSE_SIZE;
            marg_order_new.items++;
        }

        marg_H = marg_H_new;
        marg_b = marg_b_new;
        marg_order = marg_order_new;
        BASALT_ASSERT(size_t(marg_H.cols()) == marg_order.total_size);

        Eigen::VectorXd delta;
        computeDelta(marg_order, delta);
        marg_b -= marg_H * delta; //计算线性点处能量函数的偏置

        if (config.vio_debug) {
            std::cout << "marginalizaon done !!" << std::endl;

            std::cout << "========= Marg nullspace =========" << std::endl;
            checkMargNullspace();
            std::cout << "==================================" << std::endl;
        }

    }// kfs_to_marg is not empty()

//    std::cout << "new marg prior order" << std::endl;
//    marg_order.print_order();
}

void KeypointVoEstimator::optimize()
{
    if (config.vio_debug) {
        std::cout << "=================================" << std::endl;
    }

    if (true) {
        // Optimize

        AbsOrderMap aom;

        for (const auto &kv : frame_poses) {
            aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

            // Check that we have the same order as marginalization
            if (marg_order.abs_order_map.count(kv.first) > 0)
                BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                    aom.abs_order_map.at(kv.first));

            aom.total_size += POSE_SIZE;
            aom.items++;
        }

        BASALT_ASSERT(frame_states.empty());

        //    std::cout << "opt order" << std::endl;
        //    aom.print_order();

        //    std::cout << "marg prior order" << std::endl;
        //    marg_order.print_order();

        for (int iter = 0; iter < config.vio_max_iterations; iter++) {
            auto t1 = std::chrono::high_resolution_clock::now();

            double rld_error;
            Eigen::aligned_vector<RelLinData> rld_vec;
            linearizeHelper(rld_vec, lmdb.getObservations(), rld_error);

            BundleAdjustmentBase::LinearizeAbsReduce<DenseAccumulator<double>> lopt(
                aom);

            tbb::blocked_range<Eigen::aligned_vector<RelLinData>::iterator> range(
                rld_vec.begin(), rld_vec.end());

            tbb::parallel_reduce(range, lopt);

            double marg_prior_error = 0;
            linearizeMargPrior(marg_order, marg_H, marg_b, aom, lopt.accum.getH(),
                               lopt.accum.getB(), marg_prior_error);

            double error_total = rld_error + marg_prior_error;

            if (config.vio_debug)
                std::cout << "[LINEARIZE] Error: " << error_total << " num points "
                          << std::endl;

            lopt.accum.setup_solver();
            Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();

            bool converged = false;

            if (config.vio_use_lm) {  // Use Levenberg–Marquardt
                bool step = false;
                int max_iter = 10;

                while (!step && max_iter > 0 && !converged) {
                    Eigen::VectorXd Hdiag_lambda = Hdiag * lambda;
                    for (int i = 0; i < Hdiag_lambda.size(); i++)
                        Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

                    const Eigen::VectorXd inc = lopt.accum.solve(&Hdiag_lambda);
                    double max_inc = inc.array().abs().maxCoeff();
                    if (max_inc < 1e-4) converged = true;

                    backup();

                    // apply increment to poses
                    for (auto &kv : frame_poses) {
                        int idx = aom.abs_order_map.at(kv.first).first;
                        kv.second.applyInc(-inc.segment<POSE_SIZE>(idx));
                    }

                    BASALT_ASSERT(frame_states.empty());

                    // Update points
                    tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
                    auto update_points_func = [&](const tbb::blocked_range<size_t> &r)
                    {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                            const auto &rld = rld_vec[i];
                            updatePoints(aom, rld, inc);
                        }
                    };
                    tbb::parallel_for(keys_range, update_points_func);

                    double after_update_marg_prior_error = 0;
                    double after_update_vision_error = 0;

                    computeError(after_update_vision_error);

                    computeMargPriorError(marg_order, marg_H, marg_b,
                                          after_update_marg_prior_error);

                    double after_error_total =
                        after_update_vision_error + after_update_marg_prior_error;

                    double f_diff = (error_total - after_error_total);

                    if (f_diff < 0) {
                        if (config.vio_debug)
                            std::cout << "\t[REJECTED] lambda:" << lambda
                                      << " f_diff: " << f_diff << " max_inc: " << max_inc
                                      << " Error: " << after_error_total << std::endl;
                        lambda = std::min(max_lambda, lambda_vee * lambda);
                        lambda_vee *= 2;

                        restore();
                    }
                    else {
                        if (config.vio_debug)
                            std::cout << "\t[ACCEPTED] lambda:" << lambda
                                      << " f_diff: " << f_diff << " max_inc: " << max_inc
                                      << " Error: " << after_error_total << std::endl;

                        lambda = std::max(min_lambda, lambda / 3);
                        lambda_vee = 2;

                        step = true;
                    }
                    max_iter--;
                }

                if (config.vio_debug && converged) {
                    std::cout << "[CONVERGED]" << std::endl;
                }
            }
            else {  // Use Gauss-Newton
                Eigen::VectorXd Hdiag_lambda = Hdiag * min_lambda;
                for (int i = 0; i < Hdiag_lambda.size(); i++)
                    Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

                const Eigen::VectorXd inc = lopt.accum.solve(&Hdiag_lambda);
                double max_inc = inc.array().abs().maxCoeff();
                if (max_inc < 1e-4) converged = true;

                // apply increment to poses
                for (auto &kv : frame_poses) {
                    int idx = aom.abs_order_map.at(kv.first).first;
                    kv.second.applyInc(-inc.segment<POSE_SIZE>(idx));
                }

                // apply increment to states
                for (auto &kv : frame_states) {
                    int idx = aom.abs_order_map.at(kv.first).first;
                    kv.second.applyInc(-inc.segment<POSE_VEL_BIAS_SIZE>(idx));
                }

                // Update points
                tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
                auto update_points_func = [&](const tbb::blocked_range<size_t> &r)
                {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        const auto &rld = rld_vec[i];
                        updatePoints(aom, rld, inc);
                    }
                };
                tbb::parallel_for(keys_range, update_points_func);
            }

            if (config.vio_debug) {
                double after_update_marg_prior_error = 0;
                double after_update_vision_error = 0;

                computeError(after_update_vision_error);

                computeMargPriorError(marg_order, marg_H, marg_b,
                                      after_update_marg_prior_error);

                double after_error_total =
                    after_update_vision_error + after_update_marg_prior_error;

                double error_diff = error_total - after_error_total;

                auto t2 = std::chrono::high_resolution_clock::now();

                auto elapsed =
                    std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

                std::cout << "iter " << iter
                          << " before_update_error: vision: " << rld_error
                          << " marg_prior: " << marg_prior_error
                          << " total: " << error_total << std::endl;

                std::cout << "iter " << iter << "  after_update_error: vision: "
                          << after_update_vision_error
                          << " marg prior: " << after_update_marg_prior_error
                          << " total: " << after_error_total << " error_diff "
                          << error_diff << " time : " << elapsed.count()
                          << "(us),  num_states " << frame_states.size()
                          << " num_poses " << frame_poses.size() << std::endl;

                if (after_error_total > error_total) {
                    std::cout << "increased error after update!!!" << std::endl;
                }
            }

            if (iter == config.vio_filter_iteration) {
                filterOutliers(config.vio_outlier_threshold, 4);
            }

            if (converged) break;

            // std::cerr << "LT\n" << LT << std::endl;
            // std::cerr << "z_p\n" << z_p.transpose() << std::endl;
            // std::cerr << "inc\n" << inc.transpose() << std::endl;
        }
    }

    if (config.vio_debug) {
        std::cout << "=================================" << std::endl;
    }
}  // namespace basalt

void KeypointVoEstimator::computeProjections(
    std::vector<Eigen::aligned_vector<Eigen::Vector4d>> &data) const
{
    for (const auto &kv : lmdb.getObservations()) {
        const TimeCamId &tcid_h = kv.first;

        for (const auto &obs_kv : kv.second) {
            const TimeCamId &tcid_t = obs_kv.first;

            if (tcid_t.frame_id != last_state_t_ns) continue;

            if (tcid_h != tcid_t) {
                PoseStateWithLin state_h = getPoseStateWithLin(tcid_h.frame_id);
                PoseStateWithLin state_t = getPoseStateWithLin(tcid_t.frame_id);

                Sophus::SE3d T_t_h_sophus =
                    computeRelPose(state_h.getPose(), calib.T_i_c[tcid_h.cam_id],
                                   state_t.getPose(), calib.T_i_c[tcid_t.cam_id]);

                Eigen::Matrix4d T_t_h = T_t_h_sophus.matrix();

                FrameRelLinData rld;

                std::visit(
                    [&](const auto &cam)
                    {
                        for (size_t i = 0; i < obs_kv.second.size(); i++) {
                            const KeypointObservation &kpt_obs = obs_kv.second[i];
                            const KeypointPosition &kpt_pos =
                                lmdb.getLandmark(kpt_obs.kpt_id);

                            Eigen::Vector2d res;
                            Eigen::Vector4d proj;

                            linearizePoint(kpt_obs, kpt_pos, T_t_h, cam, res, nullptr,
                                           nullptr, &proj);

                            proj[3] = kpt_obs.kpt_id;
                            data[tcid_t.cam_id].emplace_back(proj);
                        }
                    },
                    calib.intrinsics[tcid_t.cam_id].variant);

            }
            else {
                // target and host are the same
                // residual does not depend on the pose
                // it just depends on the point

                std::visit(
                    [&](const auto &cam)
                    {
                        for (size_t i = 0; i < obs_kv.second.size(); i++) {
                            const KeypointObservation &kpt_obs = obs_kv.second[i];
                            const KeypointPosition &kpt_pos =
                                lmdb.getLandmark(kpt_obs.kpt_id);

                            Eigen::Vector2d res;
                            Eigen::Vector4d proj;

                            linearizePoint(kpt_obs, kpt_pos, Eigen::Matrix4d::Identity(),
                                           cam, res, nullptr, nullptr, &proj);

                            proj[3] = kpt_obs.kpt_id;
                            data[tcid_t.cam_id].emplace_back(proj);
                        }
                    },
                    calib.intrinsics[tcid_t.cam_id].variant);
            }
        }
    }
}

}  // namespace basalt
