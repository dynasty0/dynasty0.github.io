> Tensorflow实现自己的分类模型,
> 这里我们采用tensorflow源码自带的retrain.py进行训练

# 1. 首先建立数据集，将不同类的图片放入不同的文件夹里面，最后应该是{path}/data/class1,{path}/data/calss2这样的
# 2. 调用retrain.py重新训练
```python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos/   --architecture mobilenet_1.0_224_quantized```
或者

```python
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/ml/blogs/road-not-road/data/ \
    --learning_rate=0.0001 \
    --testing_percentage=20 \
    --validation_percentage=20 \
    --train_batch_size=32 \
    --validation_batch_size=-1 \
    --flip_left_right True \
    --random_scale=30 \
    --random_brightness=30 \
    --eval_step_interval=100 \
    --how_many_training_steps=600 \
    --architecture mobilenet_1.0_224
```
具体的参数解释请参考：官方源码

注：直接python xxx这样的方式调用是需要先pip安装好tensorflow的，如果没有预先安装，则需要bazel。

```
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/flower_photos
```

训练的结果应该如下显示：

```
INFO:tensorflow:2017-11-19 15:15:47.347557: Step 3999: Train accuracy = 100.0%
INFO:tensorflow:2017-11-19 15:15:47.347650: Step 3999: Cross entropy = 0.000604
INFO:tensorflow:2017-11-19 15:15:47.367126: Step 3999: Validation accuracy = 99.0% (N=100)
INFO:tensorflow:Final test accuracy = 99.4% (N=165)
INFO:tensorflow:Froze 20 variables.
Converted 20 variables to const ops
```

# 3. 使用得到的pb文件进行inference测试
训练完成之后，保存的模型文件，labels.txt包括一些中间生成，下载的预训练好的模型（上面--architecture所对应的模型文件），都会保存在/tmp/目录下。
最重要的两个文件是：/tmp/output_graph.pb和/tmp/output_labels.txt。
有了pb文件，能不能跑起来呢，我们来测试一下。

```python
import tensorflow as tf
import numpy as np
import cv2

img = cv2.imread('img.jpg')
img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
img = img.astype(np.float32)*2/255 - 1

output_graph_path = './output_graph.pb'
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    input = sess.graph.get_tensor_by_name("input:0")
    print input
    output = sess.graph.get_tensor_by_name("final_result:0")
    print output
    res = sess.run(output,feed_dict={input:[img]})
    print res
```

结果显示为：

```python
Tensor("input:0", shape=(1, 224, 224, 3), dtype=float32)
Tensor("final_result:0", shape=(?, 2), dtype=float32)
[[ 0.99773538  0.00226463]]
TODO: 使用quantized的--architecture会遇到
Traceback (most recent call last):
  File "test.py", line 16, in <module>
    _ = tf.import_graph_def(output_graph_def, name="")
  File "/home/dynasty/miniconda2/lib/python2.7/site-packages/tensorflow/python/util/deprecation.py", line 316, in new_func
    return func(*args, **kwargs)
  File "/home/dynasty/miniconda2/lib/python2.7/site-packages/tensorflow/python/framework/importer.py", line 443, in import_graph_def
    node, 'Input tensor %r %s' % (input_name, te)))
ValueError: graph_def is invalid at node u'final_training_ops/weights/MovingAvgQuantize/AssignMinEma/MovingAvgQuantize/min/MovingAvgQuantize/MovingAvgQuantize/min': Input tensor 'MovingAvgQuantize/MovingAvgQuantize/min/biased:0' Cannot convert a tensor of type float32 to an input of type float32_ref.
```

这个问题还未解决。

# 4. 将pb文件转换成lite文件

```python
import tensorflow as tf

pb_file_path = './output_graph.pb'

def canonical_name(x):
  return x.name.split(":")[0]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  output_graph_def = tf.GraphDef()
  with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")

  out = sess.graph.get_tensor_by_name("final_result:0")
  out_tensors = [out]
  input = sess.graph.get_tensor_by_name("input:0")
  frozen_graphdef = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph_def, map(canonical_name, out_tensors))
  tflite_model = tf.contrib.lite.toco_convert(
      frozen_graphdef, [input], out_tensors)
  open("converted_model.tflite", "wb").write(tflite_model)
```

# 5. tflite文件建立Android APP
。。。
