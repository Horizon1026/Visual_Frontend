# Visual_Frontend
A general and simple visual frontend consulting VINS-Mono.

# Components
- [x] Simple mono visual frontend based on klt.
    - [x] Pipeline.
    - [x] Result visualization.
    - [x] Log record.
- [x] Simple stereo visual frontend based on klt.
    - [x] Pipeline.
    - [x] Result visualization.
    - [x] Log record.
- [ ] Simple multi-view visual frontend based on klt.
    - [ ] Pipeline.
    - [ ] Result visualization.
    - [ ] Log record.

# Dependence
- Slam_Utility
- Feature_Detector
- Feature_Tracker
- Sensor_Model
- Vision_Geometry
- Image_Processor
- Visualizor2D
- Binary_Data_Log

# Compile and Run
- 第三方仓库的话需要自行 apt-get install 安装
- 拉取 Dependence 中的源码，在当前 repo 中创建 build 文件夹，执行标准 cmake 过程即可
```bash
mkdir build
cmake ..
make -j
```
- 编译成功的可执行文件就在 build 中，具体有哪些可执行文件可参考 run.sh 中的列举。可以直接运行 run.sh 来依次执行所有可执行文件

```bash
sh run.sh
```

# Tips
- 欢迎一起交流学习，不同意商用；
