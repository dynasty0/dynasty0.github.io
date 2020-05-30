---
tags: object detection
categories: 深度学习
---





## 1. RCNN

[传送门](https://arxiv.org/pdf/1311.2524v5.pdf)





## 2. Fast-RCNN

[传送门](https://arxiv.org/pdf/1504.08083.pdf)





## 3. Faster-RCNN

[传送门](https://arxiv.org/pdf/1506.01497.pdf)


## 4. R-FCN
[传送门](https://arxiv.org/pdf/1605.06409.pdf)




## 5.  yolo_v1
[传送门](https://arxiv.org/pdf/1506.02640.pdf)

### 5.1实现

[网络结构图]

前面是一个正常的卷积+pool的提取特征的网络，接两个全连接层使得最后输出是1470（7x7x30）的向量，再reshape成（7,7,30），7x7与前面网络里面最小的feature map大小一致，文章中称这7x7为grid，也就是说每一个grid小格子包含30个channel，而这30个通道包含2个框（每个框有4个值代表框的位置信息，另1个值代表confidence信息），和20个分类得分（因为是VOC）

每一个grid单元输出只有一组分类得分相关的信息，即两个框共享这些分类信息，所以grid上的两个框只能表示同一种物体

### 5.2 loss函数

[loss函数图片]

loc loss部分：

采用(x,y,w,h)的方式计算loc的L2 loss, 这里对w和h做了sqrt操作再计算loss，这是因为小框对变化更敏感，这个很好理解，大框变化一个像素，基本没有影响，但是小框，也许一个像素就是30%-50%的变化。个人觉得$[(1-W_{i}/\hat{W_{i}})^2 + (1-H_{i}/\hat{H_{i}})^2]​$更合适

confidence loss部分：

confidence的ground truth 是0或者1，如果grid单元中包含物体中心，则为1，否则为0

分类loss部分：

这个分类的loss写的很随意啊，而不写成softmax loss之类的

因为grid中包含有物体的单元数量比不包含的数量要少，为了平衡，需要加大有框的loss，减小无框的loss，所以这里取了不同的权重系数，$\lambda_{coord}=5, \lambda_{noobj}=0.5​$。

### 5.3 优缺点

优点：快，最早的one-stage方法

缺点：

* 因为算法本身每个grid单元只能有一个物体，所以在物体比较密集的情况下（448->7x7，在某个grid单元对应原图64像素中出现2个以上物体），则必定会出现部分物体无法检测到的情况
* 同样的，448->7x7，缩小的64倍，小物体特征基本上全部被抹掉了，所以小物体检测很差
* YOLO虽然可以降低将背景检测为物体的概率，但同时导致召回率较低

## 6. ssd

[传送门](https://arxiv.org/pdf/1512.02325.pdf)

### 6.1  实现

[网络结构图]

输入图片是300x300分辨率的，基础网络是VGG16，把最后的全连接层全部改成了卷积层，在VGG后面又接了一些卷积层，输出loc，confidence等信息也都是通过卷积层计算的。

另外一个很重要的是，ssd使用了muti feature map，feature比较大的去检测小物体，feature map较小的检测大物体，这样就可以有效弥补yolo_v1里面提到的缺点。

网络总共使用了6个feature map，分别是conv4_3，fc7，conv6_2，conv7_2，conv8_2，conv9_2。然后分别对这些feature map进行卷积操作，得到4,6,6,6,4,4个框的信息，其中每一个过程都有3个分支，第一个分支是计算loc，得到$4*N_{框}$通道的特征，然后对后面3个维度进行reshape操作，变成$N*(H*W*C)$，然后把6个reshape之后的concat到一起，得到mbox_loc；第二个分支，是得到$N_{类}*N_{框}$（如果是VOC，$N_{类}$就是21）通道的特征，同样操作，concat成mbox_conf；第三个分支是计算priorbox的，需要输入的原始图像和前面6个feature map，同样concat到一起，组成mbox_priorbox。

每个priorbox的最小最大值分别为[30,60]，[60,111]，[111,162]，[162,213]，[213,264]，[264,315]

这些值是怎么来的呢？

在默认情况下，对conv4_3设置的scale $S_0$为0.1，$S_{min}=0.2,\ S_{max} = 0.9$，

然后根据公式  
$$
S_{k} = S_{min}+\frac{S_{max}-S_{min}}{m-1}*(k-1), \ k\in[1,m]
$$
conv4_3的scale设置了，还有5个没有设置，所以$m=5​$
$$
\frac{S_{max}-S_{min}}{m-1} = \frac{0.7}{4} = 0.175\approx{0.17}
$$
所以， $S_1=0.2,\ S_2 = 0.37,\ S_3 = 0.54,\ S_4 = 0.71,\ S_5 = 0.88$ 这些都是归一化的值，转换到300分辨率上就是$[60,111,162,213,264]​$，这里使用非归一化的只是为了容易理解，在算法内部都是归一化的。

改变conv4_3的scale $S_0​$ 或者$S_{min}​$及m的个数可以调整不同的预设框数值

预设框的大小计算规则：

| 比例 |                         公式                          | 例子1 [30,60] | 例子2 [60,111] |
| :--: | :---------------------------------------------------: | :-----------: | :------------: |
| 1:1  |                  $[S_{min},S_{min}]$                  |    [30,30]    |    [60,60]     |
| 2:1  |         $[S_{min}*\sqrt{2},S_{min}/\sqrt{2}]$         | [42.43,21.21] | [84.85,42.43]  |
| 1:2  |         $[S_{min}/\sqrt{2},S_{min}*\sqrt{2}]$         | [21.21,42.43] | [42.43,84.85]  |
| 3:1  |        $[[S_{min}*\sqrt{3},S_{min}/\sqrt{3}]]$        |     None      | [103.92,34.64] |
| 1:3  |         $[S_{min}/\sqrt{3},S_{min}*\sqrt{3}]$         |     None      | [34.64,103.92] |
| 1:1  | $[\sqrt { S_{min}*S_{max}},\sqrt { S_{min}*S_{max}}]$ | [42.43,42.43] | [81.61,81.61]  |

4个框时不计算3:1和1:3的框

### 6.2  loss函数

[loss公式图片]

loc loss：

采用fast_RCNN里面的smooth_L1计算
$$
smooth_{L_{1}} =\begin{cases}0.5*x^2, \ if \ \ |x|\lt1 \\
|x|-0.5, \ otherwise\end{cases}
$$
confidence loss：

使用softmax计算，正常的分类使用的loss。

### 6.3  优缺点



## 7. yolo_v2

[传送门](https://arxiv.org/pdf/1612.08242.pdf)





## 8. RetinaNet

[传送门](https://arxiv.org/pdf/1708.02002.pdf)



## 9. yolo_v3
[传送门](https://arxiv.org/pdf/1804.02767.pdf)
