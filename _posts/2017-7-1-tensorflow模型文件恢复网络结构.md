---
tags: tensorflow
categories: 深度学习
---


> tensorflow模型文件恢复网络结构的几种方法介绍

## 有tf的model文件恢复网络结构

```python
import tensorflow  as tf

file_name = 'fcn.tfmodel'

reader = tf.train.NewCheckpointReader(file_name)

print(reader.debug_string().decode("utf-8"))
```

输出为：
```python
Variable (DT_INT32) []
W6 (DT_FLOAT) [7,7,512,4096]
W6/Adam (DT_FLOAT) [7,7,512,4096]
W6/Adam_1 (DT_FLOAT) [7,7,512,4096]
W7 (DT_FLOAT) [1,1,4096,4096]
W7/Adam (DT_FLOAT) [1,1,4096,4096]
W7/Adam_1 (DT_FLOAT) [1,1,4096,4096]
W8 (DT_FLOAT) [1,1,4096,3]
W8/Adam (DT_FLOAT) [1,1,4096,3]
W8/Adam_1 (DT_FLOAT) [1,1,4096,3]
W_t1 (DT_FLOAT) [4,4,512,3]
W_t1/Adam (DT_FLOAT) [4,4,512,3]
W_t1/Adam_1 (DT_FLOAT) [4,4,512,3]
W_t2 (DT_FLOAT) [4,4,256,512]
W_t2/Adam (DT_FLOAT) [4,4,256,512]
W_t2/Adam_1 (DT_FLOAT) [4,4,256,512]
W_t3 (DT_FLOAT) [16,16,3,256]
W_t3/Adam (DT_FLOAT) [16,16,3,256]
W_t3/Adam_1 (DT_FLOAT) [16,16,3,256]
b6 (DT_FLOAT) [4096]
b6/Adam (DT_FLOAT) [4096]
b6/Adam_1 (DT_FLOAT) [4096]
b7 (DT_FLOAT) [4096]
b7/Adam (DT_FLOAT) [4096]
b7/Adam_1 (DT_FLOAT) [4096]
b8 (DT_FLOAT) [3]
b8/Adam (DT_FLOAT) [3]
b8/Adam_1 (DT_FLOAT) [3]
b_t1 (DT_FLOAT) [512]
b_t1/Adam (DT_FLOAT) [512]
b_t1/Adam_1 (DT_FLOAT) [512]
b_t2 (DT_FLOAT) [256]
b_t2/Adam (DT_FLOAT) [256]
b_t2/Adam_1 (DT_FLOAT) [256]
....
....
```

## pb文件查看网络结构

```python
import tensorflow as tf

output_graph_path = './output_graph.pb'
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
    for node in output_graph_def.node:
        print node
```

## 使用nohup

```nohup python xxx.py```

然后```cat nohup.out | grep name```:
这只是个例子，具体按照实际需要来。

## 另外还有从meta恢复网络结构的

```python
## 某个例子
import tensorflow as tf
import cv2
import sys
import numpy as np
img = cv2.imread(sys.argv[1])
msk = cv2.imread(sys.argv[2])
sess = tf.Session()
## 从meta恢复网络
saver = tf.train.import_meta_graph('./model.meta')
saver.restore(sess,'./model')

graph = tf.get_default_graph()

input_img = graph.get_tensor_by_name('image:0')
input_mask = graph.get_tensor_by_name('mask:0')

output = graph.get_tensor_by_name('output:0')

out = sess.run(output,feed_dict={input_img: [img],input_mask: [msk]})

out = (out + 1)*127.5 
out = out.astype(np.uint8)

cv2.imwrite('out.png',out[0])
print(' ok !')
```
