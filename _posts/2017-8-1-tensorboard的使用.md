---
tags: 
  - tensorflow
  - tensorboard
categories: 深度学习
---


> 这篇笔记主要介绍tensorflow中tensorboard的使用


tensorboard的作用不言而喻

首先将需要显示在tensorboard里面的变量放置在summary中
这个变量可以是：

* 标量Scalars
* 图片Images
* 音频Audio
* 计算图Graph
* 数据分布Distribution
* 直方图Histograms
* 嵌入向量Embeddings

分别使用tf.summary.scalar记录标量，tf.summary.histogram记录数据的直方图，tf.summary.distribution记录数据的分布图，tf.summary.image记录图像数据等

调用tf.summary.merge_all去将所有summary节点合并成一个节点，或者tf.merge_all_summaries，具体看tf的版本，有些函数已经更新。
将汇总数据写入磁盘，需要将汇总的protobuf对象传递给tf.train.Summarywriter
SummaryWriter的构造函数中包含了参数 logdir。这个logdir 非常重要，所有事件都会写到它所指的目录下。此外，SummaryWriter中还包含了一个可选择的参数 GraphDef。如果输入了该参数，那么 TensorBoard 也会显示你的图像。

在session中run前面merge_all的summary
来自官方的代码：

```python
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
total_step = 0
while training:
  total_step += 1
  session.run(training_op)
  if total_step % 100 == 0:
    summary_str = session.run(merged_summary_op)
    summary_writer.add_summary(summary_str, total_step)
```

最后启动tensorboard

logdir 就是前面SummaryWriter里面log的路径，具体操作方式如下：

```python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory```

最后就可以在输出信息里面找到地址在浏览器中观察训练的具体情况了。

如果pip安装tensorboard之后可以用下面这种方式

```tensorboard --logdir=/path/to/log-directory```
