---
tags: object detection
categories: 深度学习
---



## 1. tensorflow object detection API 

> tensorflow object detection API 将很多模型封装好了，如faster-RCNN、SSD等，可以自由切换自己想要的模型，使用很方便。

[tensorflow object detection api 页面](https://github.com/tensorflow/models/tree/master/research/object_detection)

### 1.1 安装

[安装](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

### 1.2 数据集制作

[VOC和 Oxford-IIIT Pet 数据集tfrecord制作](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md)

[自定义数据集制作](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)

### 1.3   model zoo

[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

| Model name                                                   | Speed (ms) | COCO mAP | Outputs |
| ------------------------------------------------------------ | ---------- | ------------ | ------- |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 30         | 21           | Boxes   |
| [ssd_mobilenet_v1_0.75_depth_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz) | 26         | 18           | Boxes   |
| [ssd_mobilenet_v1_quantized_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz) | 29         | 18           | Boxes   |
| [ssd_mobilenet_v1_0.75_depth_quantized_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz) | 29         | 16           | Boxes   |
| [ssd_mobilenet_v1_ppn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz) | 26         | 20           | Boxes   |
| [ssd_mobilenet_v1_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) | 56         | 32           | Boxes   |
| [ssd_resnet_50_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) | 76         | 35           | Boxes   |
| [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | 31         | 22           | Boxes   |
| [ssd_mobilenet_v2_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2018_09_14.tar.gz) | 29         | 22           | Boxes   |
| [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) | 27         | 22           | Boxes   |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) | 42         | 24           | Boxes   |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 58         | 28           | Boxes   |
| [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) | 89         | 30           | Boxes   |
| [faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz) | 64         |              | Boxes   |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz) | 92         | 30           | Boxes   |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz) | 106        | 32           | Boxes   |
| [faster_rcnn_resnet101_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz) | 82         |              | Boxes   |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz) | 620        | 37           | Boxes   |
| [faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz) | 241        |              | Boxes   |
| [faster_rcnn_nas](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz) | 1833       | 43           | Boxes   |
| [faster_rcnn_nas_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz) | 540        |              | Boxes   |



人脸

| Model name                                                   | Speed (ms) | Open Images mAP@0.5 | Outputs |
| ------------------------------------------------------------ | ---------- | ----------------------- | ------- |
| [facessd_mobilenet_v2_quantized_open_image_v4](http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz) | 20         | 73 (faces)              | Boxes   |

### 1.4 训练

[Oxford-IIIT Pets训练](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md)

[本地训练](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)

[云端训练](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md)

---

步骤
* 准备数据集，转化成tfrecord格式
* 准备pipline文件 -》各种参数、路径等等的定义
* 训练，可以直接使用model zoo里面的finetune
* 测试
* [转换成lite上设备](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)

---

## 2. FaceBook Detectron框架

详情参考： [传送门](https://github.com/facebookresearch/Detectron)

caffe2框架（应该是并入Pytorch了）

支持下面的相关论文

- [Mask R-CNN](https://arxiv.org/abs/1703.06870) -- *Marr Prize at ICCV 2017*
- [RetinaNet](https://arxiv.org/abs/1708.02002) -- *Best Student Paper Award at ICCV 2017*
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [RPN](https://arxiv.org/abs/1506.01497)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [R-FCN](https://arxiv.org/abs/1605.06409)

## 3. static image object detection： Pelee

### 3.1  PeleeNet

Pytorch训练的一个特征提取框架，类似于VGG、resnet。

优点： 

* 速度快，输入直接接一个stride为2的卷积层，然后又缩小一半，类似于resnet，直接将输入变为1/4大小。另外，channel会动态调整，降低计算量。
* 跨层连接，除了前面的的原始层。中间是两个层concat出来，再加上原始层，可以使结果更精确。Resblock也是concat两个卷积层的输出，类似于inception。

结果：

比mobilenet_v1 计算量略小，速度快，准确率比mobilenet_v1高

### 3.2 object detection 框架 -- Pelee

基于SSD框架的，源码caffe，准确率比ssd+mobilenet高，速度也稍快。

**iPhone8   23.6FPS   图片分辨率304x304**

[论文地址](https://papers.nips.cc/paper/7466-pelee-a-real-time-object-detection-system-on-mobile-devices.pdf)

[源码](https://github.com/Robert-JunWang/Pelee)

## 4. video  object detection

### 4.1 使用convLSTM

* [Mobile Video Object Detection with Temporally-Aware Feature Maps](https://arxiv.org/pdf/1711.06368.pdf)
* [Video Object Detection with an Aligned Spatial-Temporal Memory](http://openaccess.thecvf.com/content_ECCV_2018/papers/Fanyi_Xiao_Object_Detection_with_ECCV_2018_paper.pdf)           [项目主页](http://fanyix.cs.ucdavis.edu/project/stmn/project.html)           [代码(torch)](https://github.com/fanyix/STMN) 

### 4.2 使用optical flow

* 

### 4.3  reinforcement learning

* [End-to-end Active Object Tracking via Reinforcement Learning](http://proceedings.mlr.press/v80/luo18a/luo18a.pdf)
* [Deep Reinforcement Learning of Region Proposal Networks for Object Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pirinen_Deep_Reinforcement_Learning_CVPR_2018_paper.pdf)
* [Reinforcement Cutting-Agent Learning for Video Object Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Han_Reinforcement_Cutting-Agent_Learning_CVPR_2018_paper.pdf)
* 

