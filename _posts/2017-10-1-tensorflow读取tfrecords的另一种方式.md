> 此版本需要在1.2以后的版本中使用，在这里使用的1.4，具体的因版本有些许差别，主要是`tf.data`定义的位置不同。

```
#coding=utf-8
import tensorflow  as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' ###指定日志输出级别，3：只显示error
#file = '/home/caodai/VOC2012/voc_train.tfrecords'

def dataset_input_fn():
  filenames = '/home/caodai/VOC2012/voc_train.tfrecords'
  dataset = tf.data.TFRecordDataset(filenames)

  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        'img_orginal': tf.FixedLenFeature([], tf.string),
        'img_segmentation': tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.decode_raw(parsed["img_orginal"],tf.uint8)
    image = tf.reshape(image, [500, 500, 3])
    label = tf.decode_raw(parsed["img_segmentation"], tf.int32)
    label = tf.reshape(label, [500, 500, 1])
    return image, label

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(10)
  iterator = dataset.make_one_shot_iterator()
  return iterator
  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  # features, labels = iterator.get_next()
  # return features, labels

iterator = dataset_input_fn()
a1, b1 = iterator.get_next()
# a2, b2 = iterator.get_next()
sess = tf.Session()
print a1
### 两次显示的结果不同，a1是个迭代器，自动更新。
print sess.run(a1[0,0:10,0:10,0])
print sess.run(a1[0,0:10,0:10,0])
# print sess.run(a2[0,0:10,0:10,0])
# print a1[0,0:10,0:10,0].eval(session=sess)
```

输出结果如下：
```
python test.py
Tensor("IteratorGetNext:0", shape=(?, 500, 500, 3), dtype=uint8)
[[14 13 14 14 13 13 16 17 15 14]
 [14 13 14 14 13 14 15 14 14 15]
 [14 13 14 14 13 15 15 11 13 15]
 [14 14 15 14 14 16 15 11 14 15]
 [13 14 15 14 13 15 16 13 15 16]
 [13 12 14 14 13 13 15 15 16 17]
 [14 12 13 15 14 13 15 16 16 18]
 [15 12 13 16 15 14 15 16 14 17]
 [15 14 14 15 15 14 14 17 16 16]
 [15 14 14 16 15 13 14 16 17 16]]
[[ 81  82  86  83  80  94  95  95  86  93]
 [ 75  80  85  83  80  89  90  90  91  85]
 [ 82  85  87  87  83  88  91  94  97  94]
 [ 87  86  90  96  92  94  94  94  95  98]
 [ 97  87  85  88  85  93  97  95  83  81]
 [ 98  87  77  79  79  84  90  91  79  76]
 [ 57  54  37  36  40  42  48  59  49  55]
 [ 81  92  82  83  92  89  87 100 112 121]
 [133 132 147 125 119 126 127 119 134 143]
 [ 98 105 126 103 100 112 112  98 108 109]]

```

参考：
官方文档：[importing data(需翻墙)](https://www.tensorflow.org/programmers_guide/datasets#batching_dataset_elements)
