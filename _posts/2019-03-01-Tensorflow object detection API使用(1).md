---
tags: tensorflow
categories: 深度学习
---


## 1. 数据集制作(这个一个人做了就通用的了，可跳过)

### 1.1 数据集按照VOC格式放置

 比如我创建一个`VOC2007`的数据，首先创建一个`VOC2007`的文件夹

* 建立`Annotations`文件夹  ->  将`xml`文件全部放到该文件夹里

* 建立`ImageSets`文件夹
        在`ImageSets`文件夹下再建立`Main`文件夹，生成四个`txt`文件，`test.txt`是测试集，`train.txt`是训练集，`val.txt`是验证集，`trainval.txt`是训练和验证集。`txt`里的内容是即图片名字（无后缀）。

* 建立`JPEGImages`文件夹  ->  所有的训练图片放到该文件夹里(JPG格式)

### 1.2 一些必要的安装包(lxml等)和编译
`protoc object_detection/protos/*.proto --python_out=.`

如有编译失败，参考[链接](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#manual-protobuf-compiler-installation-and-usage)

### 1.3 建立pbtxt，也就是你的类
可以去`object_detection/data`目录下去找一个修改
这里因为是使用的是`VOC`，所以复制一个`pascal_label_map.pbtxt`修改

```pascal
item {
  id: 1
  name: 'cloud'
}
item {
  id: 2
  name: 'hair'
}
item {
  id: 3
  name: 'cloth'
}
item {
  id: 4
  name: 'flower'
}
```


### 1.4 修改`dataset_tools/create_pascal_tf_record.py`文件

* `pbtxt`路径修改
```python
#flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
#                    'Path to label map proto')
flags.DEFINE_string('label_map_path', '/mnt/data2/caodai/VOC2007/pascal_label_map.pbtxt',                                'Path to label map proto')
```
* 
```python
#img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
img_path = os.path.join(FLAGS.year, image_subdirectory, data['filename'])
```
* 
```python
#examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
#                            'aeroplane_' + FLAGS.set + '.txt')
examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                 FLAGS.set + '.txt')
```



### 1.4 生成数据集
```bash
python object_detection/dataset_tools/create_pascal_tf_record.py \
  --data_dir=/mnt/data2/caodai \ ##刚刚建立的VOC2007文件夹的路径
  --year=VOC2007 \ ##定义的就是VOC2007
  --output_path=/mnt/data2/caodai/VOC2007/train.record ## 数据集保存路径
```

## 2.  训练

### 2.1 下载`config`文件对应的`ckpt`

[官方model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models)

解压后的里面有`ckpt`，还有个`pipline.config`（这个和`object_detection/samples/configs`对应的`config`文件是一样的，也可以直接修改这里的`config`），`model`的`ckpt`路径就是上面 `config` 问题 `fine_tune_checkpoint`后面要修改的路径

### 2.2 修改`config`文件

`config`文件路径 `object_detection/samples/configs`

复制一份修改，`ssd_mobilenet_v1_coco.config` 前面`ssd`是算法，`mobilenet_v1`是提取特征框架，`coco`是代表是在`coco`数据上训练的模型

然后将`PATH_TO_BE_CONFIGURED`换成实际的路径，`num_classes`换成所需要的类

其它参数也可以调整，后面看自己需求。

### 2.3 安装一些包和插件

安装`pycocotools`

```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
将库加入到`pythonpath`

```bash
# 在 models/research/ 目录下运行
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
或者写在.bashrc里面，用绝对路径
export PYTHONPATH=$PYTHONPATH:/mnt/data1/newhome/caodai/models/research:/mnt/data1/newhome/caodai/models/research/slim
```

### 2.4 运行 


```bash
CUDA_VISIBLE_DEVICES=0 \
python object_detection/model_main.py \
    --pipeline_config_path=/mnt/data2/caodai/VOC2007/ssd_mobilenet_v1_coco.config \
    --model_dir=/mnt/data2/caodai/VOC2007/modelss \
    --num_train_steps=10000 \
    --sample_1_of_n_eval_examples=1 \
    --alsologtostderr
```
