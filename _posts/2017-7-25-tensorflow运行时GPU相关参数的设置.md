> tensorflow运行时GPU相关参数的设置

使用tensorflow时，设置GPU的一些参数，如可以多人共享使用显卡训练网络。

```python
config = tf.ConfigProto()
# 使用一半的显存
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
```
具体参考tf.ConfigProto()及gpu_options相关参数的定义。

