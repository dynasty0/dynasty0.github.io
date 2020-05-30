---
tags: tensorflow
categories: 深度学习
---


# tensorflow 2.x 使用预训练模型


## 1. keras.applications

具体模型会下载保存在`~/.keras/models`下



```python
model = tf.keras.applications.ResNet101V2(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000
)
model.summary()

for layer in model.layers:
	print(layer.name,)

```

也可通过`layer = model.get_layer(name = "name",index = "index")`获取具体的层

`layer`的属性:

`input`

`output`

`trainable` 

...

具体通过`dir(layer)`查看

获取具体的层再向下传递



**缺点：**

不能函数式向下传递、、、、



## 2. tensorflow_hub

```python
#-----------------------#
### in TF 1.x
path = "https://..."
model = hub.Module(path)
### 但是不支持eage模式
resnet = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3")
### 内置__call__方法可获取中间特征
outputs = resnet(np.random.rand(1,224,224,3), signature="image_feature_vector", as_dict=True)
for intermediate_output in outputs.keys():
    print(intermediate_output)


```

tf2有两种方式

```python
### 获取某些中间层特征，需要保证signature模型中有定义
hub.KerasLayer(
    handle, trainable=False, arguments=None, _sentinel=None, tags=None,
    signature=None, signature_outputs_as_dict=None, output_key=None,
    output_shape=None, **kwargs
)

### 示例
path = "https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_101/feature_vector/4"
resnet101 = hub.KerasLayer(,
                   trainable=False,
                   signature="image_feature_vector",
                   signature_outputs_as_dict = True,
                   )
feature_names = (
    "resnet_v2_101/block1/unit_1/bottleneck_v2/shortcut",
    "resnet_v2_101/block2/unit_1/bottleneck_v2/shortcut",
    "resnet_v2_101/block3/unit_1/bottleneck_v2/shortcut",
    "resnet_v2_101/block4/unit_1/bottleneck_v2/shortcut",
    )
input = tf.random.uniform([1,512,512,3])

result_dict = resnet101(input)

for name in feature_names:
    if name in result_dict.keys():
        print(name,result_dict[name])
```

另一种，

```python
###同样需要模型中有signature
path = "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_100_224/feature_vector/3"

model = hub.load(path)

input = tf.random.uniform([1,512,512,3])
model = model.signatures["image_feature_vector"](input)

feature_names = [
    "resnet_v2_101/block1/unit_1/bottleneck_v2/shortcut",
    "resnet_v2_101/block2/unit_1/bottleneck_v2/shortcut",
    "resnet_v2_101/block3/unit_1/bottleneck_v2/shortcut",
    "resnet_v2_101/block4/unit_1/bottleneck_v2/shortcut",
     ]
for key in feature_names:
    print(key, model[key].shape)
```

但是，有个坑的地方是

```python
hub.KerasLayer(
    handle, trainable=False, arguments=None, _sentinel=None, tags=None,
    signature=None, signature_outputs_as_dict=None, output_key=None,
    output_shape=None, **kwargs
)
```

里面的`trainable`如果是在1.x的模型中是不能设为`True`的，而2.x的模型貌似都没有`signature`。估计还是试验阶段吧。。。。。



**更新**

```python
###使用tf2.x模型，添加 arguments=dict(return_endpoints=True) 以字典的形式，获取所有输出
l = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4",
    trainable=True,
    arguments=dict(return_endpoints=True))  ### 《----- here
images = tf.keras.layers.Input((224, 224, 3))
outputs = l(images)
for k, v in sorted(outputs.items()):
  print(k, v.shape)

```

终于可以愉快的玩耍了。
