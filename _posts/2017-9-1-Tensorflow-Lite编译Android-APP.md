> 主要介绍Tensorflow Lite编译Android APP

# 1. bazel安装
官网安装教程
推荐使用教程里面的 Install using binary installer 方法，apt方法会报Java9之类的错误。

# 2. 下载tensorflow源码

```git clone https://github.com/tensorflow/tensorflow.git```

# 3. 安装SDK和NDK

首先安装的Android studio，然后在file->setting->system setting->Android SDK里面，安装需要的sdk,ndk,sdk-tools等等这些

然后进入tensorflow的根目录，更改WORKSPACE文件，打开注释，对sdk,ndk进行设置。下面是我的设置。
```
android_sdk_repository(
    name = "androidsdk",
    api_level = 25,
    # Ensure that you have the build_tools_version below installed in the
    # SDK manager as it updates periodically.
    build_tools_version = "27.0.1",
    # Replace with path to Android SDK on your system
    path = "/home/dynasty/Android/Sdk",
)

android_ndk_repository(
    name="androidndk",
    path="/home/dynasty/android-ndk-r14b",
    # This needs to be 14 or higher to compile TensorFlow.
    # Please specify API level to >= 21 to build for 64-bit
    # archtectures or the Android NDK will automatically select biggest
    # API level that it supports without notice.
    # Note that the NDK version is not the API level.
    api_level=14)
```
由于Android studio安装的ndk版本为16，在编译的时候，报下面的错误

```external/androidndk/ndk/sources/cxx-stl/gnu-libstdc++/4.9/include/cstdlib:72:10: fatal error: 'stdlib.h' file not found```

网上说，这是32位软件和64位系统不兼容什么什么的，另外我看到编译的时候，系统提示```WARNING: The major revision of the Android NDK referenced by android_ndk_repository rule 'androidndk' is 16. The major revisions supported by Bazel are [10, 11, 12, 13, 14]. Defaulting to revision 14.```这样的警告，于是我下了一个ndk14，这也是为什么我上面的ndk的设置路径不一样。

# 4. 编译
```bazel build -c opt --cxxopt='--std=c++11' //tensorflow/contrib/lite/java/demo/app/src/main:TfLiteCameraDemo```

成功运行。
这里使用的是官方的例子，我把相对应的tflite文件和labels.txt换成了我自己的。
然后就可以在tensorflow根目录的bazel-bin/tensorflow/contrib/lite/java/demo/app/src/main/TfLiteCameraDemo.apk找到你的apk了。

参考
lite官方文档
lite编译Android APP
不会安卓，sdk,ndk安装设置简直让人欲仙欲死。。。。
