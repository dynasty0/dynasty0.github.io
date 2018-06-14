---
tags: tensorflow
categories: 深度学习
---

> GPU版本tensorflow强制使用CPU方法


在训练好的模型进行测试时，往往需要在各种模型下进行测试，看看运行时间，内存，显存占用等等这些参数，如果你需要做模型移植到板子上手机上，在CPU下的测试数据，是一个很重要的参考。

# 使用tensorflow的with tf.device('/cpu:0'):函数。简单操作就是把所有命令都放在前面所述的域里面。

有点麻烦，要改代码。。

# 使用tensorflow声明Session时的参数：
在声明Session的时候加入device_count={'GPU':0}即可，代码如下：

```python
import tensorflow as tf

#### ‘GPU’这个一定要大写，至少在我用的1.2.1版本里需要大写

sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}))
```
# 第三种是使用CUDA_VISIBLE_DEVICES命令行参数，代码如下：
CUDA_VISIBLE_DEVICES="" python train.py (可以跑，但是越跑越慢)
