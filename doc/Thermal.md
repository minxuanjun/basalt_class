## Thermal Inertial Odometry

1. frontend

 A. 参考LDSO的特征提取, 部分特征点是角点, 其他点是DSO特征提取方法
 B. 初始化过程中, 使用光流追踪特征点,进行IMU偏置的初始化,然后使用DSO的初始化方法,只优化特征点的逆深度
 C. 当滑动窗口中激活点点,初始化成功  
[![teaser](/doc/img/kitti_video.png)](https://www.youtube.com/watch?v=M_ZcNgExUNc)

We demonstrate the usage of the system with the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) as an example.

**Note:** The path to calibration and configuration files used here works for the APT installation. If you compile from source specify the appropriate path to the files in [data folder](/data/).

Download the sequences (`data_odometry_gray.zip`) from the dataset and extract it. 
```
# We assume you have extracted the sequences in ~/dataset_gray/sequences/
# Convert calibration to the basalt format
basalt_convert_kitti_calib.py ~/dataset_gray/sequences/00/

# If you want to convert calibrations for all sequences use the following command
for i in {00..21}; do basalt_convert_kitti_calib.py ~/dataset_gray/sequences/$i/; done
```
Optionally you can also copy the provided ground-truth poses to `poses.txt` in the corresponding sequence.

### Visual odometry
To run the visual odometry execute the following command.
```
basalt_vio --dataset-path ~/dataset_gray/sequences/00/ --cam-calib /work/kitti/dataset_gray/sequences/00/basalt_calib.json --dataset-type kitti --config-path /usr/etc/basalt/kitti_config.json --show-gui 1 --use-imu 0
```
![magistrale1_vio](/doc/img/kitti.png)

## Mono Visual odometry

### System Module
1．单目初始化

- [x] 使用本质矩阵恢复相机的位姿
- [x] 完成3d点的三角化
- [x] 单目相机的恢复的尺度进行归一化 
- [ ] 3d点三角化的进行重投影误差的判定
- [ ] 将初始化的frame和landmark插入数据库

问题:
三角化特征点加入余弦值判断,好像初始化不容易成功,可能需要借鉴
VINS-Mono的SFM初始化部分的代码

2．track 模块(自己手写两帧优化器)
- [x] 追踪last frame （使用优化库，还是自己手写优化器）
- [x] 追踪last keyframe（使用优化库，还是自己手写优化器）
- [x] 优化器的设计,参考DSO,和openvslam,basalt,只需要求对$T_{rc}$的jacobian
- [ ] 需要保存last_frame, ll_frame的id,来计算速度估计值


问题:
这一块可能需要借鉴DSO corseTrackr 部分完成位姿追踪,如果上一帧不是关键帧,而不需要放入滑窗中

3 ．mapping 模块
- [x] 三角化特征点(basalt 已经实现)
这一块可能需要修改,可能需要遍历滑窗的中的所有为三角化的特征点,将其三角化
而不是像basalt只三角化当前帧对应的3d点

4．backend optimization 模块
好像不太需要怎么修改


### 思考问题

1. 我们应该将初始化的3d点主导帧选择为**参考帧**还是**当前帧**?
主导帧必须选择是参考帧, 因为如果当前帧被选择为主导帧,那么参考帧将没有主导的3d点,当参考帧被边缘化时,
其给滑窗中的其他帧的先验为0, 没有先验,系统变成了一个无约束的问题.系统将会在零空间飘逸.


## Stereo Visual Odometry

1. 双目需要写一个初始化函数吗?
 好像不需要

2. 第一帧能被边缘化吗?
可以,由于是双目相机,第一帧可以三角化一些特征点,即其主导一些3d点,边缘化其主导的3d点
 可以给滑窗中的其他帧留下先验
 
 
 
 ***Notice***
 纯视觉里程计滑动窗口好像只能维护关键帧,非关键帧好像应该直接删除,last_frame, llast frame 好像应该存放在CorseTrackr 中,用来计算速度
 
 
 
 ***bug***
track 模块的CorseTrackr 模块将上一帧存储在estimator 中

mapping模块需要每次运行,将为三角化的特征点进行三角化

marginalize 操作,直接将非关键帧直接删除


