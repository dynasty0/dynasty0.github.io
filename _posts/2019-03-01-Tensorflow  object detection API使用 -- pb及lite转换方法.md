---
tags: tensorflow
categories: 深度学习
---





## 1. 转换成pb

```bash
CUDA_VISIBLE_DEVICES=0 \
python ~/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix /mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/model.ckpt-100000 \
    --output_directory /mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/
```

会在`output_directory`目录下生成`frozen_inference_graph.pb`

## 2. 转换成tflite

### 2.1 错误的方式

**WARNING: 不能使用上面生成的那个pb去转换lite**

如果直接使用下面的转换方式

```bash
CUDA_VISIBLE_DEVICES=0 \
tflite_convert \
  --output_file=/mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/a.tflite \
  --input_arrays=image_tensor \
  --input_shapes=1,300,300,3 \
  --graph_def_file=/mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/frozen_inference_graph.pb \
  --output_arrays=detection_boxes,num_detections
```

将会得到下面的信息

```bash
RuntimeError: TOCO failed see console for info.
2019-03-13 14:56:11.664004: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: TensorArrayV3
2019-03-13 14:56:11.664096: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1127] Op node missing output type attribute: Preprocessor/map/TensorArray
2019-03-13 14:56:11.664159: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: TensorArrayScatterV3
2019-03-13 14:56:11.664185: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1127] Op node missing output type attribute: Preprocessor/map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3
2019-03-13 14:56:11.664217: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: TensorArrayV3
2019-03-13 14:56:11.664231: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1127] Op node missing output type attribute: Preprocessor/map/TensorArray_1
2019-03-13 14:56:11.664252: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: TensorArrayV3
2019-03-13 14:56:11.664262: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1127] Op node missing output type attribute: Preprocessor/map/TensorArray_2
2019-03-13 14:56:11.664285: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1080] Converting unsupported operation: Enter
...
...
...
2019-03-13 14:56:11.841215: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before Removing unused ops: 1549 operators, 2700 arrays (0 quantized)
2019-03-13 14:56:11.962505: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After Removing unused ops pass 1: 1424 operators, 2479 arrays (0 quantized)
2019-03-13 14:56:12.131439: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before general graph transformations: 1424 operators, 2479 arrays (0 quantized)
2019-03-13 14:56:12.288594: F tensorflow/contrib/lite/toco/graph_transformations/resolve_constant_slice.cc:59] Check failed: dim_size >= 1 (0 vs. 1)
Aborted (core dumped)
```

这是因为原始的pb里面包含大量的预处理和后处理，里面很多OP是不支持的

### 2.2 正确打开方式

使用`export_tflite_ssd_graph.py`脚本生成`tflite_graph.pb`和`tflite_graph.pbtxt`

```bash
CUDA_VISIBLE_DEVICES=9 \
python ~/models/research/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path /mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix /mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/model.ckpt-100000 \
    --output_directory /mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/
```

再使用生成的`tflite_graph.pb`转换成lite

`tflite_graph.pb` 里面最后一个OP是`TFLite_Detection_PostProcess `，这个OP是不支持的，它的作用是使用前面计算的`raw_outputs/box_encodings` , `raw_outputs/class_predictions` 和`anchors` 再做一些后处理

所以在转lite的时候，使用上面3个输出作为output，再使用这3个output做后处理，两者分开（前面在网络中跑，后面的另外再实现）。

```bash
CUDA_VISIBLE_DEVICES=9 \
tflite_convert \
  --output_file=/mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/a2.tflite \
  --input_arrays=normalized_input_image_tensor \
  --input_shapes=1,300,300,3 \
  --graph_def_file=/mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/tflite_graph.pb \
  --output_arrays=anchors,raw_outputs/box_encodings,raw_outputs/class_predictions
```

### 2.3 量化

```bash
CUDA_VISIBLE_DEVICES=9 \
tflite_convert \
  --output_file=/mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/a2.tflite \
  --graph_def_file=/mnt/data2/caodai/object_detection_train/ssd_mobilenet_v1_record/record0305/tflite_graph.pb \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=normalized_input_image_tensor \
  --input_shapes=1,300,300,3 \
  --output_arrays=anchors,raw_outputs/box_encodings,raw_outputs/class_predictions \
  --default_ranges_min=0 \
  --default_ranges_max=6 \
  --mean_values=128 \
  --std_dev_values=127 
```

量化前 *21.22M* -->  量化后*5.37M*

上面的有问题

---

## 3. 官方guide

[guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)

[github上面一个很好的object detection论文整理总结](https://github.com/hoya012/deep_learning_object_detection)
