```
---
tags: tensorflow
categories: 深度学习
---
```



## 1. 编译环境搭建

> 这里使用docker搭建 

### 1.1 拉取docker镜像

```bash
docker pull tensorflow/tensorflow:latest-devel-gpu-py3
```

具体镜像可以去 https://hub.docker.com/r/tensorflow/tensorflow/tags?page=1&name=devel  选择想要的拉取

### 1.2 创建docker实例

```bash
docker run --gpus all --name work3 -it -v /data:/data tensorflow/tensorflow:latest-devel-gpu-py3 /bin/bash
```

### 1.3 拉取最新的`tensorflow`源码

```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r2.2
```

### 1.4 配置build

```bash
./configure
```

### 1.5 编译

```bash
###编译
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
###生成whl
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./
###pip安装生成的whl即可
pip install ...
```

这个会很慢,几个小时以后再来看...

## 2. tflite编译benchmark工具

### 2.1 安装NDK

https://developer.android.google.cn/ndk/downloads/ 下载对应平台的NDK

```bash
unzip android-ndk-r21b-linux-x86_64.zip
```

在`~/.bashrc`中设置NDK环境变量

```
### NDK
export ANDROID_NDK_HOME=/data/work_ubuntu/android-ndk-r21b
export PATH=${PATH}:$ANDROID_NDK_HOME
```

**必须先设置好NDK**,不然会出现类似于下面的错误

```
toolchain' does not contain a toolchain for cpu 'arm64-v8a'
```

### 2.2 编译

```bash
bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark:benchmark_model
```

以下来自tf-github, 参考 https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark

```bash
(1) Build for your specific platform, e.g.:
bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark:benchmark_model

(2) Connect your phone. Push the binary to your phone with adb push (make the directory if required):
adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp

(3) Make the binary executable.
adb shell chmod +x /data/local/tmp/benchmark_model

(4) Push the compute graph that you need to test. For example:
adb push mobilenet_quant_v1_224.tflite /data/local/tmp

(5) Optionally, install Hexagon libraries on device.
That step is only needed when using the Hexagon delegate.
### tf2.2以前的hexagon在tensorflow/lite/experimental/delegates/hexagon/hexagon_nn下
bazel build --config=android_arm64 \
  tensorflow/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so
adb push bazel-bin/tensorflow/lite/delegates/hexagon/hexagon_nn/libhexagon_interface.so /data/local/tmp
### 下面的得去 https://www.tensorflow.org/lite/performance/hexagon_delegate#example_usage 下载最新版本
### 具体的路径 https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run
### 执行./hexagon_nn_skel_v1.17.0.0.run 得到下面的要用到的三个so
adb push libhexagon_nn_skel*.so /data/local/tmp

(6) Run the benchmark. For example:

adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --num_threads=4
```

更多用法,访问 https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark