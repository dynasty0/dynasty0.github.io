> 主要记录CRFasRNN及FCN训练细节

# 1. 数据集的制作
LFW数据集包含了原图，以及分割好的图片，其中背景（蓝色）、人脸（绿色）、头发（红色）。数据层Data部分就是原图，label是分割的图片，将带颜色的分割图片替换成0->背景，人脸->1，头发->2，保存为h5文件

H5文件大小有限制，且caffe的HDF5Data层不支持对数据的变换，如归一化操作。

之后又做了Lmdb的数据层，data和label分开作为两个数据层输入。

# 2. 网络的修改
将train.prototxt的数据层替换成相对应的层，在最后几层的卷积和反卷积num_output替换为我们的类数3.

# 3. 训练的细节
Caffe一般的训练方式

```../../caffe train –solver solver.prototxt –weights xxx.caffemodel```

FCN网络中使用的反卷积使用的是双线性插值初始化，不能直接使用类似于上面的方式进行训练。

FCN源代码中提供了solve.py文件，调用自己定义的surgery实现了对反卷积层的初始化。

FCN训练首先训练FCN32，这一步需要使用训练好的vgg16网络，且必须使用下面的方式将vgg16中的参数移植过来。

```python
##原作者自己定义的
vgg_weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'

vgg_proto = '../ilsvrc-nets/VGG_16_deploy.prototxt'

weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'

solver = caffe.SGDSolver('solver.prototxt')

vgg_net=caffe.Net(vgg_proto,vgg_weights,caffe.TRAIN)

surgery.transplant(solver.net,vgg_net)
```

上面这个是直接训练，还有一种是拿训练好的fcn32进行微调，这时，直接

```python
weights = 'fcn32s-heavy-pascal.caffemodel'

solver = caffe.SGDSolver('solver.prototxt')

solver.net.copy_from(weights)
```

这时不需要移植，然后再使用训练好的fcn32对fcn16进行微调，再使用训练好的fcn16微调fcn8，学习率分别为1e-9，1e-12，1e-13。

# 4. 主要问题及解决

主要问题是因为数据的问题，由于lfw数据集里面给的分割图片是ppm格式的，原图片是背景蓝色(0,0,255),头发(255,0,0),脸部(0,255,0),最开始转成了jpg格式，由于jpg格式会压缩，导致图片中有很多杂点，像(254,1,0)、(255,0,1)之类的

因为做数据的时候需要按照像素点打标签，如果像素点不准的话，直接影响后面训练。

另外，lfw数据图片大小是250x250的，需要放大到500x500训练，一般采用双线性插值的方式放大图片，这同样会引入很多杂点。

解决方案：直接在横向、纵向复制像素点或者resize时采用最近邻方式

```img_new = np.repeat(np.repeat(np.uint8(img),2,0),2,1)```

同样，训练的时候使用fcn作者提供的solve.py，batch_size=1,在原作者在voc上训练的caffemodel上进行fine-tune,迭代100000次。
