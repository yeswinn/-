# 机器学习

## 关于我对这一段马拉松的经历的分享
---

### 我的一个自己介绍吧
首先我先介绍一下自己吧，我叫郭俊熙，0903班的新生，我感觉我高考是机缘巧合才上来的，我可能没有像其他人那么优秀，并且在电脑使用这个方面我也是一个纯小白，我感觉相当于零基础了

### 选择这个软件工程的一个原因
  我觉得我之所以选择软件工程的原因呢，主要是源于我小学和初中的一个经历吧，我小学就开始一点点的接触这个电脑，小时候我记得我们上信息课的时候我很喜欢玩[编程猫](https://cn.bing.com/search?pglt=43&q=%E7%BC%96%E7%A8%8B%E7%8C%AB&cvid=49ac6bbba0b041139fa68cb24d59f3c3&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIGCAEQABhAMgYIAhAAGEAyBggDEAAYQDIGCAQQABhAMgYIBRAAGEAyBggGEAAYQDIGCAcQABhAMgYICBAAGEDSAQg0OTA3ajBqMagCCLACAQ&FORM=ANNTA1&ucpdpc=UCPD&adppc=EdgeStart&PC=NMTS)，当时很自豪自己还真的编出个小游戏，就是一只黄色小鸟在两个柱子中间穿过的那个，我也忘记叫什么名字了。
  到了初中，我开始参加那个创客cca，每周就一节课，我当时学什么我自己都忘记了，但我当时第一次听说python这个词在课堂上，但没有讲，他让我们自学，当然我在家都是玩游戏，也没有听他说的话去自学，到初一下半年，我就去参加了一个机器人的寻路比赛，当然我什么都不会，我就是挺水的，我就负责3d打印，负责拼装那个小车，其他都是初二的学长带我飞的，当时还拿了一等奖（我是被拉着去的，主要没参加过比赛，想去观摩一下）
  当然我到此还是对电脑几乎是一窍不通，可能我比较蠢吧，当然也是这一次次经历把我带向了软件工程（第一志愿）这个专业（我是自愿来的，不是被调剂的）
### 选择机器学习的原因
  我是10月8号才开始决定选择机器学习，因为我想着本来学校就要求学习c语言，我打算先进行需要c语言方向的学习，例如cs之类的，但在我对每一个方向进行一个粗略的了解之后我发现我对机器学习比较感兴趣，所以我还是选择了机器学习这个方向，主要是我比较懒，想着自己也可以做出一个机器能模仿人类的行为，帮我做事，这样我就可以躺平了（当然我主要感觉比较有趣）
### 顺带吐槽一下这个我的自学的一个过程
10.20
自学过程可谓是跌宕起伏，我感觉我开始学习的时间比较晚了，所以我就一直在图书馆熬夜奋战了，但我感觉自己是真的很蠢，就比如我配置环境真的是足足配了2个小时，就是我是先配置了pycharm和python，这两个比较简单，但我配置anaconda的时候我明明跟着教程走，我也不懂为什么会错，还有我搭建pytorch框架的时候我是想用gpu安装的，但我跟着教程走，他就是不能安装，搞了好久，真不知道英伟达是不是针对我，感觉真的太崩溃了，最后还是cpu安装了pytorch
还有我发现了那个人工智能的数学基础b站下面的真的是参次不齐，大多数都是需要我有一定的高数的基础，然后再进行一个人工智能下面拓展的应用，我通常听着听着就要求助ai了，后面我听了2天我真的直接放弃了，还有[廖雪峰的python教学](https://www.bilibili.com/video/BV1zE41137Nm/?spm_id_from=333.337.search-card.all.click&vd_source=c53d227101d827f11dfe7ff94ff55878)，我听了6个小时，我感觉他讲的挺好，很生动，但他不给我举例子呀，就是没有实战的经历，真的很难听下去，就是挺多理论灌进我的大脑，到最后也没学会python，还有[吴恩达的机器学习](https://www.bilibili.com/video/BV1Bq421A74G?spm_id_from=333.788.videopod.episodes&vd_source=c53d227101d827f11dfe7ff94ff55878&p=67)，我看了很多的机器学习的视频，他讲的是最生动的，但是那个字幕到20集就没了，我用b站的字幕，经常给我一堆奇奇怪怪的东西，什么乙状结肠函数，把我都整不会了，我还以为我在医学院呢，还有他的教程都是Tensorflow，就同时学两个框架感觉有点乱，我是真的绷不住，然后学习[pytorch框架](https://www.bilibili.com/video/BV13a411u7G5/?spm_id_from=333.337.search-card.all.click&vd_source=c53d227101d827f11dfe7ff94ff55878)的时候,他直接实战演练，说草履虫都能学会，我是15分钟的视频，40分钟都在ai上面查每一段代码的意思，有时候我都怀疑我不如草履虫了
### 学习的收获
虽然这段时间很忙，学的东西也不是很多，但我至少也是涨经验了，知道怎么去学习了，也变得比较自律，至少没有像暑假一样摆烂了，还有即使学的不算多但也是勉强的入门了

---
## 机器学习过程中知识的总括
### 偏导数、链式法则、梯度、矩阵等数学概念在机器学习中的作用
#### 偏导数
偏导数和告诉我们损失函数在每个参数方向上的变化率。根据这些变化率，我们可以调整参数，使得损失函数朝着下降的方向变化。
还有对于一个具有多个层和多个神经元的神经网络，我们可以通过计算损失函数对每个参数的偏导数，我们可以确定每个参数对损失的贡献，从而更新参数以优化模型。
#### 链式法则
当我们要计算损失函数关于权重的导数，就需要使用链式法则，因为损失函数是通过y间接依赖于权重的。链式法则允许我们将复杂的导数计算分解为一系列简单的导数计算，从而计算出损失函数对每个参数的导数，进而更新参数。
我认为用于反向传播算法，反向传播通过链式法则从输出层开始，逐层计算损失函数对每个神经元的权重和偏置的导数，从而有效地更新网络参数。
#### 梯度
比较典型的就是梯度下降，在优化模型参数时，我们通常沿着梯度的反方向更新参数，以最小化损失函数，他的公式是[如图]
$$
\theta = \theta - \alpha \nabla_\theta J(\theta)
$$
其中阿尔法是学习率的意思
#### 矩阵
在一个具有个样本和个特征的数据集，我们可以用一个的矩阵来表示特征数据，这样可以是我们计算更加快捷
通过利用矩阵的性质，如并行计算等特点，可以在 GPU 等硬件上高效地实现模型的训练和预测。例如，在深度学习框架中，对矩阵乘法的优化可以大幅提高模型的训练速度。
### 监督学习和非监督学习
![监督学习](https://pic.imgdb.cn/item/67163dd9d29ded1a8c513e4d.png)
![非监督学习](https://pic.imgdb.cn/item/67163e75d29ded1a8c52afec.png)
#### 总述
监督学习和非监督学习就是一个是我输出了一个x会有一个具体的y值会输出，一个只是会把我x值进行处理，而不会产生y值
#### 监督学习
例如图中的监督学习中的例子 
回归模型中我是根据我给出的数据集来构建一个函数，从而我可以输入一个x时预测相对应的一个y值，例如最经典的一个买房问题，当我给出100个房子的特征和他所对应的价钱后，我会通过这100个房子的特征与价钱来构建一个函数，从而我可以在下次输入另外一个房子的特征时可以输出价格y，这个y是一个具体的值
像分类模型中，图中只是一个简单的逻辑回归，就例如图中，他给出这么多的数据集，我们要做的就是把他们进行绿色和红色的分类，在电脑中就是1和0的分类，故相对应的，用人类的语言表示就是，当我在图上再点一个点的时候，电脑就会输出y就是他是绿色还是红色的，也就是1和0
#### 非监督学习
这个我就不过多的描述了，像图中就是很明显的例子，机器要学习的就是如何将我i给出的数据集进行处理，例如第二个的降维，我输入很多的数字，机器要做的要让数字接近的颜色相近，我从而能更好的观察与分类，例如第三个的密度估计，根据我给上面标的带你来进行密度的估计，密度大的用深色来表示，在这整个过程中我只对x进行处理，但我至始至终都不会输出y

---
### 机器学习和深度学习的区别
#### 机器学习
![机器学习](https://pic.imgdb.cn/item/67163e0ad29ded1a8c51b543.png)
#### 深度学习
如图所示在机器学习中，我们需要凭靠经验和手工的数据处理，像我们平常的线性回归，逻辑回归还有决策树之类的它通常只需要比较简单的算法就可以执行
![深度学习](https://pic.imgdb.cn/item/67164485d29ded1a8c5f47a8.png)
如图所示，在表示学习中能够自动从原始数据中学习复杂的特征表示，减少了特征工程的需求，并且深度学习会更加复杂，他是很多个神经层来构成的，依赖于深层神经网络，这些网络由多个非线性的变换层组成。
简单来说深度学习是机器学习中的特例子，他更加的复杂，需要的算力会越大，需要读取的数据集需要很多，当然我觉得也可以说机器学习是深度学习比较底层的一个逻辑吧

### 常见的激活函数
![激活函数](https://pic.imgdb.cn/item/6716499ad29ded1a8c6386c4.png)
最常见的激活函数就是这三种
当然还有很多的激活函数：线性回归，逻辑回归，relu
#### softwax
例如像逻辑回归的进阶版softmax函数
$$
\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j}e^{x_j}}
$$
怎么理解呢？就相当于逻辑回归中我只能表示1和0，那要是我要表示不止表示一个的分类时候我们就可以用softmax函数来表示
![softmax与逻辑回归的区](https://pic.imgdb.cn/item/67163e10d29ded1a8c51c110.png)
如下就是两个函数的公式的表达的相似和区别了

#### Tanh神经元

### 神经网络的基本结构

简单来说就是分为输入层，隐藏层和输出层
而隐藏层中包含着神经元
![基本架构](https://pic.imgdb.cn/item/67171118d29ded1a8ca21437.png)

##### 权重和偏置
权重：连接神经元的边，决定输入信号的重要性。每个连接都有一个权重值。
偏置：为每个神经元添加的常数项，允许激活函数在不同的点上移动。
简单来说就是这样
![权重](https://pic.imgdb.cn/item/67171bb9d29ded1a8cb4104c.png)
![](https://pic.imgdb.cn/item/67171264d29ded1a8ca3396b.jpg)
如图，像w就是权重决定着这个数据的重要程度，越重要他的权重占比就会越大，偏置就是图中的b

##### 激活函数
激活函数就不多说了，常见的函数可见上面
主要是讲一下在神经网络中如何的去选择激活函数
**先说结论一般在隐藏层中，都会选择relu函数
在输出层中根据实际情况来选择函数**
例如你要是要输出真假值的话就用逻辑回归，要是是一个线性回归的一个问题的话，就使用线性回归函数
拿线性回归函数来举例子，要是隐藏层使用线性回归方程，可以想象一下，到最后他只是一个线性回归的一个嵌套，我们根本没有必要设置隐藏层，我们可以直接使用一个比较复杂的线性回归就可以解决
要是使用逻辑回归来当隐藏层的激活函数的话，他就会有下面的结果![](https://pic.imgdb.cn/item/67163e17d29ded1a8c51d19c.png)
因为他前后都趋近于零，当我们使用梯度下降的时候，他就会有很多的局部最小点，那我们就很难得到一个比较小的一个成本函数


#### 损失函数损失函数
（Loss Function），也称为代价函数（Cost Function），是机器学习中用来评估模型预测值与真实值之间差异的函数。损失函数的目的是量化模型的预测误差
**简单来说，损失函数就是评价你的模型好坏的一个标准**
对于线性回归来说他的成本函数是
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
对于逻辑回归来说
$$
\text{Cross-Entropy Loss (Binary)} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$
对于softmax来说
$$
\text{Cross-Entropy Loss (Categorical)} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{m} y_{ic} \log(\hat{y}_{ic})
$$
![](https://pic.imgdb.cn/item/67171a05d29ded1a8cb0280c.png)

#### 优化器
优化器（Optimizer）是用于在训练过程中调整模型参数以最小化损失函数的算法
以下是常用的优化器（暂时我只学了这几种）

##### 随机梯度下降
与批量梯度下降每次更新参数时使用所有训练样本不同，SGD每次只随机选取一个样本或一对样本来更新模型的参数
$$
\theta = \theta - \alpha \nabla_\theta L(\theta; x^{(i)}, y^{(i)})
$$
每次参数更新只需要处理一个样本，因此计算和内存需求较低，运算会比较快，
但由于每次更新只基于一个样本，参数更新的步长可能会很大，导致收敛路径不稳定。
并且很依赖学习率，一般会与其他优化器一同使用

##### adam
简单来说，它可以根据实际情况改变学习率，例如要是成本函数一直沿着一个方向下降但速度较慢，那么他会增大学习率来加快收敛，要是学习率太大，离成本函数越来越远，那么他就会减少学习率，是函数收敛

#### 训练过程
**前向传播：输入数据通过各层进行计算，生成输出。
反向传播：通过计算损失函数的梯度，更新权重和偏置，以减少预测误差。**
#### 训练集 测试集 交叉验证测试集
训练集就是用来训练模型的一个数据集，机器会通过这个数据集来构造一个模型
交叉验证测试集就是我在已经得出一个模型中实验我是否会产生过拟合和欠拟合的问题，就是我再给出一个x，我通过这个模型来推测出y，来和我的真实值y比较，来算出损失函数
我会通过这个结果来改进我的模型
测试集就是最终用来测试最终的结果，他是不受干扰的
### 机器学习中的数据处理
#### 数据清洗：

**处理缺失值**：删除或填充缺失的数据。
**移除异常值**：识别并处理异常值，以减少它们对模型的影响。
#### 特征工程：
**特征选择**：从现有数据中选择最相关的特征。
**特征提取**：创建新的特征或转换现有特征以提供更多信息

#### 数据转换：

**归一化/标准化**：调整特征的尺度，使其对模型的影响一致。
**降维**：减少特征的数量，如使用主成分分析（PCA）

#### 数据划分：

**将数据集划分为训练集、验证集和测试集**


### 过拟合和欠拟合和正则化
关于这几个概念我将引用几幅图来表示
![过拟合和欠拟合](https://pic.imgdb.cn/item/67177d5bd29ded1a8c4f0a61.jpg)
正如同他的字面意思意义欠拟合就是在训练集中都不能大概预测y值，也就是我们说的代价函数太大，不够拟合
过拟合就是在训练集上拟合的程度太高，反而可能在预测其他非训练集的时候会产生较大的误差，这就叫做过拟合
![正则化](https://pic.imgdb.cn/item/67177caad29ded1a8c4d90c5.jpg)
![正则化](https://pic.imgdb.cn/item/67177d39d29ded1a8c4ebf41.jpg)
正如同图中写出的任何损害优化的方法都是正则化，为什么要损害优化？因为过拟合了，就好像第二幅图一样，我要防止他的图像特别陡峭，我要使图像变得平滑一点，那么我要降低他的权重，这样我x在变化的时候他的变化就不会那么大，当然肯定不止这种方法
像图一的把训练集扩大一点，训练的更多，最后预测的会更加地准确，提前停止呢就是像图中所示，当我的验证集的错误率上升的时候，说明他就过拟合了，这个时候我们就可以停止了
### 多标签分类和多类分类


简单的来说多标签问题就是我在一张图片中，有车人和红绿灯，我要做的就是检测这张图片中是否有这些东西，它可以同时包含车人红绿灯，通常使用逻辑回归
但对于多类问题，他的每个样本是互斥的，就比如一张动物图片中要不就是猫要不就是狗要不就是其他，不可能同时存在狗和猫的混合体


## 最后的话
今天是10.23，很感慨，从10月9号到今天14天，还有几天发烧，感觉还是有点累的，兼顾课业和招新，感觉还是有点后悔的，后悔我怎么不在9月份的时候就开始学习机器学习，总体的感觉还是自己不够时间，其实我当时学到15号的时候我都有点想放弃了，那个时候我才刚刚装好环境，刚看了一下数学基础和python学习，但我不喜欢半途而废，所以就坚持下来了，真的是每天呆在图书馆里面到10点多，听那个音乐响起，但我还是很开心的，毕竟还有我的两个舍友陪我受苦，在这么辛苦的过程中我收获了知识，更收获了友谊，认识了我们班在机器学习方向挣扎的人，但他们是大佬，我是小菜鸡，跟在他们身边纯纯长见识，还有最后一点，我发现机器学习的魅力了，感觉真的很奇妙，也是坚定了我在这条路上不断前行，今天交了这个招新题我就不赶进度了，我想花几天重头系统地学习，我感觉我有点急于求成了
