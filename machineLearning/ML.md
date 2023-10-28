# ML

## 绪论

### ML框架

![image-20230913084005731](img/1.png)

### types of learning problems

* 监督学习
* 无监督学习
  * 自监督学习
* 半监督学习
* 迁移学习
* 主动学习
* 强化学习
* 元学习

## Regression

### 单变量线性回归

* **输入特征**

  $x^{(i)} \in R^{n+1},i=1,2,···,m$

* **输出**

  $y^{(i)} \in R$

* **参数**

  $\theta = R^{n+1}$

* 假设$h_{\theta}(x):R^{n+1} \to R$

* **损失函数**

  $\ell :R \times R \to R_+$

  满足

  * 非负：不存在负损失
  * 如果预测结果$h_{\theta}(x)$与给定的y差别小，则损失小，反之则损失大

  平方损失：

  $\ell(h_{\theta}(x),y)=(h_{\theta}(x)-y)^2$

#### **三要素**

  * 假设：$h_{\theta}(x)= \theta_0+\theta_1x$，其中参数为$\theta_0,\theta_1$

  * 目标函数：

    $J\left(\theta_0, \theta_1\right)=\frac{1}{2 m} \sum_{i=1}^m \ell\left(h_\theta\left(x^{(i)}\right), y^{(i)}\right)=\frac{1}{2 m} \sum_{i=1}^m\left(h_\theta\left(x^{(i)}\right)-y^{(i)}\right)^2$

  * 优化算法：给定训练集，如何找到最优的参数$\theta$使得
    $$
    \min _{\theta_0, \theta_1} J\left(\theta_0, \theta_1\right)
    $$

* **参数优化**

  找到最优的参数$\theta^*=arg~ \min_{\theta} J(\theta)$

  * 穷举所有$\theta$
  * 随机搜索
  * 梯度下降
  

#### **梯度下降**

  repeat until convergence{

  $\theta_j:=\theta_j-\alpha \frac{\partial}{\partial \theta_j} J\left(\theta_0, \theta_1\right) \quad($ for $j=0$ and $j=1)$

  }

* **梯度**：
  $$
  \nabla_\theta f(\theta) \in \mathbb{R}^n=\left[\begin{array}{c}
  \frac{\partial f(\theta)}{\partial \theta_1} \\
  \frac{\partial f(\theta)}{\partial \theta_2} \\
  \vdots \\
  \frac{\partial f(\theta)}{\partial \theta_n}
  \end{array}\right]
  $$
  梯度下降算法的另一种表述：
  $$
  Repeat:~\theta=\theta-\alpha \nabla_{\theta}f(\theta)
  $$

* 单变量线性回归模型的梯度下降
  $$
  repeat~until~convergence:\\
  \theta_0=\theta_0-\alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})\\
  \theta_1=\theta_1-\alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)}) \cdot x^{(i)}
  $$

### 多特征（变量）

#### 三要素

* 假设：$h_\theta(x)=\theta_0 x_0+\theta_1 x_1+\theta_2 x_2+\cdots+\theta_n x_n, x_0=1$
* 参数：$\theta_0,\theta_1,\dots,\theta_n$
* 目标函数：$J\left(\theta_0, \theta_1, \cdots, \theta_n\right)=\frac{1}{2 m} \sum_{i=1}^m \ell\left(h_\theta\left(x^{(i)}\right), y^{(i)}\right)=\frac{1}{2 m} \sum_{i=1}^m\left(h_\theta\left(x^{(i)}\right)-y^{(i)}\right)^2$

#### 梯度下降

$$
repeat~until~convergence:\\
\theta_j:=\theta_j-\alpha \frac{\partial}{\partial \theta_j} J\left(\theta_0, \theta_1,\dots,\theta_n\right)~~j=0,1,\dots,n
$$

#### 特征尺度归一化

* 范围归一化：使得每个特征尽量接近某个范围，如$0 \le x_i \le 1$

* 零均值归一化：用$x_i-\mu_i$替代$x_i$，即$x_i- \mu_i \to x_i$，其中$\mu_i=\frac{1}{m} \sum_{i=1}^{m} x_i$为均值

* 零均值+范围归一化

* 零均值单位方差归一化：
  $$
  \frac{x_i- \mu_i}{\sigma_i} \to x_i
  $$

#### 学习率

梯度下降$\theta_j:=\theta_j-\alpha \frac{\partial}{\partial \theta_j} J\left(\theta \right)$

收敛条件$\Delta J(\theta) \le 10^{-3}$

自动收敛测试

* 对于足够小的$\alpha,J(\theta)$应该在每一次迭代中减小
* 如果$\alpha$太小，梯度下降算法收敛速度慢
* 反之，梯度下降算法不会收敛、发散或者震荡

#### 正规方程

* 对于求函数极小值问题，除了迭代方法之外，可以**令函数的微分为零，然后求解方程**

$$
\theta \in \mathbb{R}^{n+1},\\
J\left(\theta_0, \theta_1, \cdots, \theta_n\right)=\frac{1}{2 m} \sum_{i=1}^m\left(h_\theta\left(x^{(i)}\right)-y^{(i)}\right)^2\\
\nabla_\theta J(\theta)=0
$$

解出$\theta_0,\theta_1,\dots,\theta_n$

考虑到求和需要进行循环，使用矩阵运算会有更小的时间复杂度

得到$J(\theta)=\frac{1}{2m}(X\theta-y)^T(X\theta-y)$

则$\nabla_\theta J(\theta)=\frac{1}{m}(X^TX\theta-X^Ty)=0$

解出$\theta=(X^TX)^{-1}X^Ty$

#### 梯度下降和正规方程的比较

m训练样本，n个特征

* 梯度下降
  * 需要选择合适的$\alpha$
  * 需要多次迭代
  * 即使n很大效果也很好
* 正规方程
  * 不需要选择$\alpha$
  * 不需要迭代
  * 需要计算$(X^TX)^{-1}$
  * n很大会导致求解很慢
  * 矩阵不可逆时需要删减一些特征，或者进行正则化

## classification

### Logistic Regression

目标：$0 \le h_\theta(x) \le 1$
$$
h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$

#### Sigmoid函数

![image-20230927092917418](img/2.png)

#### 概率解释

$$
h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$

$h_\theta(x)$对于输入x，输出y=1的可能性

给出x，估计y=1的可能性，$\theta$为参数
$$
P(y=0|x;\theta)+P(y=1|x;\theta)=1
$$

#### 分类边界

数形结合可知，$h_{\theta}(x)$形成的一个闭合曲线/曲面作为一个分类边界
$$
h_{\theta}(x) \ge c \to y=1
$$

$$
h_{\theta}(x) \lt c \to y=0
$$

#### 损失函数

$$
P(y=0|x;\theta)=h_{\theta}(x)
$$

$$
P(y=1|x;\theta)=1-h_{\theta}(x)
$$

$$
p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$

$$
L(\theta)=p(y|\mathbf{X};\theta)=\prod_{i=1}^m p(y^{(i)}|x^{(i)};\theta)=\prod_{i=1}^m (h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
$$

**Logistic损失函数**
$$
\ell(\theta)=-\log L(\theta)=-\left[\sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)})+(1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]
$$
**cross entropy 交叉熵**
$$
H(p,q)=-\sum_x p(x) \log q(x)
$$
通俗来说，$p(x)$是真实样本的分布，$q(x)$是预测样本的分布，cross entropy代表的是两个分布之间的距离，越小说明分布越接近，迭代过程中需要将ce降下来

**分类问题中常用**
$$
\operatorname{Cost}\left(h_\theta(x), y\right)=\left\{\begin{aligned}
-\log \left(h_\theta(x)\right) & \text { if } y=1 \\
-\log \left(1-h_\theta(x)\right) & \text { if } y=0
\end{aligned}\right.
$$

#### 梯度下降

$$
J(\theta)=-\left[\sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)})+(1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]
$$

找到合适的参数$\theta$使得$\min_\theta J(\theta)$

repeat{
$$
\theta_j=\theta_j-\alpha \frac{\partial}{\partial \theta_j}J(\theta)
$$
}

$\frac{\partial}{\partial \theta_j}J(\theta)=(h_\theta(x)-y)x_j$

不可以直接使用线性回归中的平方损失函数

![image-20231008163324478](img/3.png)

黑色的是cross entropy，红色的是square error

#### Multi-class Classification

$$
h_{\theta_i}=P(y=i|x;\theta_i),i=1,2,3,...
$$

一对多

* 为每类训练一个逻辑回归分类器$h_{\theta_i}(x)$用来预测$y=i$的可能性

* 对于新输入$x$，做一个预测，选择一个类别$i^*$使得：
  $$
  i^*=arg \max_i h_{\theta_i}(x)
  $$

#### softmax Regression

$$
p(y=i|x;\theta)=h_{\theta_i}(x)=\frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}},z_j=(\theta_i)^Tx
$$

对数似然为
$$
L(\theta)=\sum_{i=1}^{m} \log p(y^{(i)}|x^{(i)};\theta)=\sum_{i=1}^{m} \log (\frac{e^zy^{(i)}}{\sum_{j=1}^K e^{z_j}})
$$
总损失为
$$
\ell(\theta)=-L(\theta)=-\sum_{i=1}^{m} \log (\frac{e^zy^{(i)}}{\sum_{j=1}^K e^{z_j}})=\sum_{i=1}^m \left[\log(\sum_{j=1}^K e^{z_j})-z_{y^(i)}  \right]
$$

$$
\hat{y_i}=h_{\theta_i}(x)
$$

## 模型选择与正则化

### 偏差与方差

偏差bias，方差variance
$$
bias(h(x))=E[h(x)-y(x)]
$$

$$
var(h(x))=E \lbrace h(x)-E[h(x)] \rbrace
$$

### 过拟合

如果多项式阶数较大，训练得到的模型对于训练集能正确拟合$J(\theta)=\frac{1}{2m}[h_\theta(x^{(i)}-y^{(i)})^2] \approx  0$，但是对于新的样本预测效果却不好

实际应用中容易出现过拟合

绘制这个模型的学习曲线

通过学习曲线的形态来判断

所谓学习曲线就是训练集得分和验证集得分随着训练样本数的增大而变化的曲线

**欠拟合情况**：随着训练样本数增大，训练集得分和验证集得分收敛，并且两者的收敛值很接近

**过拟合情况**：随着训练样本数增大，训练集得分和验证集得分相差还是很大

### 模型选择

将训练集随机分成两部分：用于训练参数的训练集和用于模型选择的验证集

![image-20231015142857582](img/4.png)

![image-20231015142932617](img/5.png)

![image-20231015142956584](img/6.png)

### 诊断误差和方差

训练误差：$L_{train}(\theta)=\frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2$

验证误差：$L_{val}(\theta)=\frac{1}{2m_{val}} \sum_{i=1}^{m_{val}} (h_\theta(x_{val}^{(i)})-y_{val}^{(i)})^2$

#### 偏差大（underfit欠拟合）

训练误差：大

训练误差与验证误差差别较小

#### 方差大（overfit过拟合）

训练误差：小

验证误差远大于训练误差

![image-20231015143732091](img/7.png)

从图像上可以知道，靠左侧是欠拟合，靠右侧是过拟合

### 解决欠拟合和过拟合问题

#### 欠拟合

核心：增加模型的复杂度

* 收集新的特征
* 增加多项式组合特征
* ...

#### 过拟合

* 增加数据
* 降低模型的复杂度
  * 减少特征（人为筛选）
  * 正则化，可降低方差提高偏差

### 正则化线性回归

#### Regularized Linear Regression

$$
\min_\theta J(\theta)
$$

$$
J(\theta)=\frac{1}{2m} \left[ \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2 \right]
$$

$$
L(\theta)=\frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2
$$

$$
J(\theta)=L(\theta)+\lambda R(\theta)
$$

gradient descent

repeat{
$$
\theta_j=\theta_j(1-\alpha \frac{\lambda}{m})-\alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$
}

**正则化参数$\lambda$的选择**
$$
J(\theta)=\frac{1}{2m} \left[ \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2 \right]
$$

$$
L(\theta)=\frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2
$$

#### RegularizedNormal equation

$$
\theta=\left(X^T X+\lambda\left[\begin{array}{lllll}
0 & & & & \\
& 1 & & & \\
& & 1 & & \\
& & & \ddots & \\
& & & & 1
\end{array}\right]\right) X^{-1} X^T y
$$

### Regularized Logistic Regression

$$
J(\theta)=\left[-\frac{1}{m} \sum_{i=1}^m y^{(i)} \log \left(h_\theta\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log (1-h_\theta\left(x^{(i)}\right))\right]+\frac{\lambda}{2 m} \sum_{j=1}^n \theta_j^2\right.
$$

梯度下降同上

$$
J(\theta)=L(\theta)+\lambda R(\theta)
$$

$$
L(\theta)=\left[-\frac{1}{m} \sum_{i=1}^m y^{(i)} \log (h_\theta(x^{(i)})+(1-y^{(i)}) \log (1-h_\theta(x^{(i)}))\right]
$$

### 学习曲线

![image-20231016190009923](img/8.png)

如果一个模型测试结果是high bias，使用更多的训练数据并不能改进模型

![image-20231016190147599](img/9.png)

如果一个模型测试结果是high variance，使用更多的训练数据会有效改进模型

### 模型性能评估

* 用训练集训练参数

$$
\theta^*=a\arg \min_\theta \frac{1}{m} \sum_{i=1}^m \ell(h_\theta(x^{(i)}),y^{(i)})
$$

* 用验证集选择模型，用于调参（正则化参数、多项式阶数、特征选择）
* 测试集仅用于性能评估

#### 验证集和测试集

* 验证集和测试集应具有同分布
* 验证集和测试集的大小
  * 验证集：1000-10000；应当足够大
  * 测试集：中小30%；大数据足够大

#### 交叉验证 k-fold cross validation

* 数据集规模较小情况下采用
* 把数据随机划分为k等份，每次用其中的(k - 1)份做训练，剩下的做验证
* 计算平均误差（和方差）

## 神经网络与深度学习

### 非线性分类

### The "one learning algorithm" hypothesis

### 神经元模型：Logistic unit

![image-20231016192226609](img/10.png)

### 全连接前馈网络 Fully Connect Feedforward Network

![image-20231016192336825](img/11.png)

![image-20231016192436942](img/12.png)

如果在第j层有$s_j$个units，第j+1层有$s_{j+1}$个units，则$\theta^{(j)}$的维数是$s_j \times s_{j+1}$

FCFN的执行过程

激活函数$g(z)=\frac{1}{1+e^{-z}}$

![image-20231016193208513](img/13.png)

### 前向传播：矩阵表示

![image-20231016193446007](img/14.png)
$$
h_\theta(x)=g(\theta^{(2)}g(\theta^{(1)}x))
$$

### 特征学习

![image-20231016193745129](img/15.png)

### 多层神经网络

![image-20231016193817513](img/16.png)

#### 用神经网络求解XOR/XNOR问题

![image-20231017090005632](img/18.png)

![image-20231017085931923](img/17.png)

![image-20231017090721937](img/19.png)

![image-20231017090803876](img/20.png)

#### 处理多分类问题

![image-20231017091225042](img/21.png)

#### 手写数字识别

![image-20231017091338145](img/22.png)

### 网络结构

$$
h_\theta(x)=g_L(\theta^{L-1}g_{L-1}(\dots g_2(\theta^{(2)}g_1(\theta^{(1)}x))))
$$

### DeepLearning:Many hidden layers

### 激活函数

$$
z=\sum \theta_ix_i +\theta_0
$$

$$
\frac{dz}{d\theta_i}=x_i
$$

$$
\frac{dJ}{d\theta_i}=\frac{dJ}{dz} \frac{dz}{d \theta_i}=\frac{dJ}{dz}x_i
$$

#### sigmoid

![image-20231020160853644](img/23.png)
$$
sigmoid:g(x)=\frac{1}{1+e^{-x}}
$$

1. Saturated neurons “kill” the gradients
2. Sigmoid outputs are not zerocentered
3. exp() is a bit compute expensive

#### tanh

![image-20231020161036525](img/24.png)
$$
tanh:g(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

1. Zero centered (nice)

2. “kill”the gradients
3. exp() is a bit compute expensive

#### ReLU

![image-20231020161629731](img/25.png)
$$
ReLU:g(x)=\max(0,x)
$$

1. Does not saturate (in +region)

2. Very computationally efficient
3. Converges much faster than sigmoid/tanh in practice (e.g. 6x)
4. Actually more biologically plausible than sigmoid
5. Not zero-centered output
6. x<0: dead ReLU will never activate => never update

#### Leaky ReLU

![image-20231020161851458](img/26.png)
$$
Leaky ReLU:g(x)=\max(0.1x,x)
$$

1. Does not saturate (in +region)

2. Very computationally efficient
3. Converges much faster than sigmoid/tanh in practice (e.g. 6x)
4. Actually more biologically plausible than sigmoid
5. Not zero-centered output
6. Will not “die”

$$
Parametric\ Rectifier\ Linear\ Unit(PReLU):g(x)=\max(\alpha x,x)
$$

### 损失函数

$$
l(y,\hat{y})=-\sum_{i=1}^n y_i \log(\hat{y_i})
$$

### 目标函数

$$
J(\theta)=L(\theta)+\lambda R(\theta)
$$

$$
h_\theta(x) \in \mathbb{R}^K,(h_\theta(x))_k=k^{th}\ output
$$

$$
\begin{aligned}
J(\Theta) & =-\frac{1}{m}\left[\sum_{i=1}^m \log p\left(y^{(i)} \mid x^{(i)} ; \Theta\right)\right]+\frac{\lambda}{2 m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}}\left(\Theta_{j i}^{(l)}\right)^2 \\
& =-\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log \left(h_{\Theta}\left(x^{(i)}\right)\right)_k+\frac{\lambda}{2 m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}}\left(\Theta_{j i}^{(l)}\right)^2
\end{aligned}
$$

### 梯度下降法

$$
\min_\theta J(\theta)
$$

### 梯度计算：反向传播

BackPropagation，BP

#### 链式法则

![image-20231020170049734](img/27.png)

#### 梯度矢量化表示

![image-20231020170507628](img/28.png)

#### Batch gradient descent vs. Stochastic gradient descent

![image-20231020171456260](img/29.png)

![image-20231020171603203](img/30.png)

**Mini-batch gradient descent**

![image-20231020171644520](img/31.png)

* Batch gradient descent

  ```python
  for i in range(nb_epochs):
  	params_grad=evaluate_gradient(loss_function,data,params)
  	params=params-learning_rate*params_grad
  ```

* Stochastic gradient descent

  ```python
  for i in range(nb_epochs):
  	np.random.shuffle(data)
  	for example in data:
  		params_grad=evaluate_gradient(loss_function,example,params)
  		params=params-learning_rate*params_grad
  ```

* Mini-batch gradient descent

  ```python
  for i in range(nb_epochs):
  	np.random.shuffle(data)
  	for batch in get_batches(data,batch_size=50):
  		params_grad=evaluate_gradient(loss_function,batch,params)
  		params=params-learning_rate*params_grad
  ```

### Stepsize vs. Gradient

![image-20231020172336633](img/32.png)

### Adaptive 学习率

![image-20231020172436170](img/33.png)

#### AdaGrad

$$
\alpha^t=\frac{\alpha}{\sqrt{t+1}},g_t=\frac{\partial J(\theta^t)}{\partial \theta}
$$

$$
\theta^{t+1}=\theta^t-\frac{\alpha^t}{\sigma^t}g_t,\sigma^t=\sqrt{\frac{1}{t+1}\sum_{i=0}^{t}g_i^2}
$$

$$
\theta^{t+1}=\theta^t-\frac{\alpha^t}{\sqrt{\sum_{i=0}^tg_i^2}}g_t
$$

#### RMSProp

$$
u^0=0
$$

$$
u^t=\rho u^{t-1}+(1-\rho)g_t^2
$$

$$
\theta^{t+1}=\theta^t-\frac{\alpha}{\sqrt{u^t}}g_t
$$

#### AdaDelta

![image-20231021145518518](img/34.png)

#### Momentum

**GD+Momentum**
$$
v^0=0
$$

$$
v^t=\rho v^{t-1}+\alpha g_t
$$

$$
\theta^{t+1}=\theta^t-v^t
$$

引入速度（前面梯度的加权均值）

![image-20231021145743467](img\35.png)

**Nesterov Momentum**

![image-20231021145908869](img/36.png)
$$
v^t=\rho v^{t-1}+\alpha \frac{\partial J(\theta^t+\rho v^{t-1})}{\partial \theta}
$$

$$
\theta^{t+1}=\theta^t-v^t
$$

#### Adam

$$
m^0=0,v^0=0
$$

$$
m^t=\beta_1 m^{t-1}+(1-\beta_1)g_t
$$

$$
v^t=\beta_2 v^{t-1}+(1-\beta_2)g_t^2
$$

$$
\theta^{t+1}=\theta^t-\frac{\alpha}{\sqrt{v^t+\epsilon}}m%t
$$

![image-20231021161930422](img\37.png)

#### AdaBelief

![image-20231021162335145](img/38.png)
$$
s^t=\beta_2 s^{t-1}+(1-\beta_2)(g_t-m_t)^2
$$

$$
\theta^{t+1}=\theta^t-\frac{\alpha}{\sqrt{s^t+\epsilon}}m%t
$$

Adam or AdaBelief might be the best overall choice

SGD

* usually achieves to find a minimum, but it might take significantly longer
* is much more reliant on a robust initialization and annealing schedule
* may get stuck in saddle points rather than local minima

### Additional strategies for optimizing SGD

* Shuffling and Curriculum Learning 打乱数据集和课程学习
* Early stopping: Early stopping (is) beautiful free lunch (Geoff Hinton) 早停
* Gradient noise: add noise to each gradient update 梯度噪声

### 二阶方法：牛顿法

迭代规则
$$
\theta:=\theta-\frac{f(\theta)}{f'(\theta)}
$$

$$
\theta:=\theta-\frac{f(\theta)}{f'(\theta)},\min_\theta J(\theta) \to J'(\theta)=0,f(\theta)=J'(\theta)
$$

$$
\theta=\theta-\frac{J'(\theta)}{J''(\theta)}
$$

$\theta$是矢量，则$\theta:=\theta-H^{-1} \nabla_\theta J(\theta) ,H_{ij}=\frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}$

由于二阶方法需要计算Hessian矩阵，在参数数目较多的情况下计算复杂，目前主要仍采用一阶方法

### Gradient checking

$$
\frac{\partial J(\theta)}{\partial \theta} \approx \frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}
$$

![image-20231021165426644](img/39.png)

### 初始化

不能zero initialization，否则训练出全部相同的神经元

Random initialization (Gaussian with zero mean and 1e-2 standard deviation)，Works ~okay for small networks, but problems with deeper networks

### 数据预处理

![image-20231021165926172](img/40.png)

### 网格结构选择与设计

前一层的输出与后一层的输入需要一致

### 训练网络

* Initialize weights
* Implement forward propagation
* Implement code to compute cost function
* Implement backpropagation code to compute gradient
* Implement numerical gradient check (disable your gradient checking code before training)
* Double check that the loss is reasonable
* Make sure that you can overfit very small portion of the training data
* Start with small regularization and find learning rate that makes the loss go down

## Support Vector Machines

$$
\hat{\gamma}^{(i)}=y^{(i)}(w^Tx^{(i)}+b)
$$

$$
y=\operatorname{sign}\left(w^T x+b\right)= \begin{cases}+1, & w^T x+b>0 \\ -1, & w^T x+b<0\end{cases}
$$

要求：
$$
y^{(i)}(w^Tx^{(i)}+b)>0
$$

### 最大间隔分类器(Max Margin Classifier)
![image-20231025082744416](img/41.png)
$$
w^Tx^++b=+1,w^Tx^-+b=-1,x^+=x^-+\lambda w
$$

$$
margin=\parallel x^+-x^- \parallel
$$

$$
w^T(x^+-x^-)=2
$$

$$
\lambda=\frac{2}{w^Tw}
$$

$$
margin=\parallel x^+-x^-\parallel =\parallel \lambda w \parallel=\frac{2}{\parallel w \parallel}
$$

$$
margin:\max_w \frac{2}{\parallel w \parallel} \to \min_w \frac{1}{2}\parallel w \parallel^2
$$

### The Primal Hard SVM

假设数据线性可分，即$y^{(i)}(w^Tx^{(i)}+b) \ge 1$

*关于函数间隔为什么可设置为1，参考[机器学习SVM中关于函数间隔为什么可以设置为1？](https://www.zhihu.com/question/64568136)，这里面写的比较详细*
$$
\begin{aligned}
\min _{w, b} & \frac{1}{2}\|w\|^2 \\
\text { s.t. } & y^{(i)}\left(w^T x^{(i)}+b\right) \geq 1, \quad i=1, \ldots, m
\end{aligned}
$$
典型的二次规划问题

#### Equality constraint

$$
\max_x f(x)
$$

$$
\text { s.t. }  g(x)=0
$$

$$
\nabla f(x^*,y^*)+\lambda \nabla g(x^*,y^*)=0,\beta \neq 0
$$

定义Lagrange函数$\mathcal L(x,y,\lambda)=f(x,y)+\lambda g(x,y)$

令$\nabla_{x,y} \mathcal L=0$得到上式

令$\nabla_{\lambda} \mathcal L=0$得到约束条件$g(x,y)=0$

![image-20231025091620975](img/42.png)

#### Inequality constraint

$$
\max_x f(\mathbf x)
$$

$$
\text{s.t.} g(\mathbf x) \ge 0
$$

$$
\mathcal{L}(\mathbf x,\lambda)=f(\mathbf x)+\lambda g(\mathbf x)
$$

无效：

最优解本身满足约束$g(\mathbf x) >0$，约束条件不起作用，即
$$
\lambda=0
$$
有效：

最优解再约束条件边界上，$g(\mathbf x) =0$，此时$\nabla g(\mathbf x)$方向必定与$\nabla f(\mathbf x)$相反，即
$$
\nabla f(\mathbf x)+\lambda \nabla g(\mathbf x)=0,\lambda >0
$$
两种情况都有$\lambda g(\mathbf x)=0$

#### Lagrange Multiplier

##### equality

$$
\min_w f(w)
$$

$$
\text{s.t.} h_i(w)=0,i=1,\dots,l
$$

$$
\mathcal{L} (w,\beta)=f(w)+\sum_{i=1}^l \beta_ih_i(w),\beta_i:\mathbf{Lagrange\ multipliers}
$$

$$
\frac{\partial \mathcal L}{\partial w}=0;\frac{\partial \mathcal L}{\partial \beta_i}=0
$$

##### inequality&equality

$$
\min_w f(w)
$$

$$
\text{s.t.}\ g_i(w) \le 0,i=1,\dots,k
$$

$$
\ \ \ \ \ h_i(w)=0,i=1,\dots,l
$$

$$
\mathcal L(w,\alpha,\beta)=f(w)+\sum_{i=1}^k \alpha_i g_i(w)+\sum_{i=1}^l \beta_i g_i(w)
$$


$$
\text{s.t.}\ \alpha_i g_i(w)=0,i=1,\dots,k
$$

$$
\ \ \ \alpha_i \ge 0,i=1,\dots,k
$$

$$
\theta_{\mathcal{P}}(w)=\max _{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)
$$

给出$w$，如果$w$违反了最初的约束(e.g. $g_i(w)<0\ or \ h_i(w) \neq 0 \ for \ some \ i$)，应当能给出
$$
\theta_{\mathcal{P}}(w)=\max _{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)=\max _{\alpha, \beta: \alpha_i \geq 0}  f(w)+\sum_{i=1}^k \alpha_i g_i(w)+\sum_{i=1}^l \beta_i g_i(w)=\infin
$$

#### Primal vs. Dual

##### Primal

$$
\theta_{\mathcal{P}}(w)=\max _{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta),\mathcal L(w,\alpha,\beta)=f(w)+\sum_{i=1}^k \alpha_i g_i(w)+\sum_{i=1}^l \beta_i g_i(w)
$$

$$
\theta_{\mathcal{P}}(w)= \begin{cases}f(w) & \text { if } w \text { satisfies primal constraints } \\ \infty & \text { otherwise. }\end{cases}
$$

$$
\min_w \theta_{\mathcal{P}}(w)=\min_w \max _{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)
$$

##### Dual

$$
\theta_{\mathcal D}(\alpha,\beta)=\min_w \mathcal{L}(w, \alpha, \beta)
$$

$$
\max _{\alpha, \beta: \alpha_i \geq 0} \min_w \mathcal{L}(w, \alpha, \beta)
$$

##### 弱对偶性

$$
d^*=\max _{\alpha, \beta: \alpha_i \geq 0} \min_w \mathcal{L}(w, \alpha, \beta) \le \min_w \max _{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)=p^*
$$

##### 强对偶性

$$
d^*=\max _{\alpha, \beta: \alpha_i \geq 0} \min_w \mathcal{L}(w, \alpha, \beta) = \min_w \max _{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)=p^*
$$

under certain conditions:

* f is convex(its Hessian is positive semi-definite)
* $g_i$'s are convex set
* $h_i$'s are affine, $h_i(w)=a_i^Tw+b_i$

#### KKT Conditions

$$
\min_w f(w)
$$

$$
\text{s.t.}\ g_i(w) \le 0,i=1,\dots,k
$$

$$
\ \ \ \ \ h_i(w)=0,i=1,\dots,l
$$

$$
\mathcal L(w,\alpha,\beta)=f(w)+\sum_{i=1}^k \alpha_i g_i(w)+\sum_{i=1}^l \beta_i g_i(w)
$$

$$
\begin{aligned}
\frac{\partial}{\partial w_i} \mathcal{L}\left(w^*, \alpha^*, \beta^*\right) & =0, \quad i=1, \ldots, n \\
\frac{\partial}{\partial \beta_i} \mathcal{L}\left(w^*, \alpha^*, \beta^*\right) & =0, \quad i=1, \ldots, l \\
\alpha_i^* g_i\left(w^*\right) & =0, \quad i=1, \ldots, k \\
g_i\left(w^*\right) & \leq 0, \quad i=1, \ldots, k \\
\alpha^* & \geq 0, \quad i=1, \ldots,
\end{aligned}
$$

#### SVM:from Primal to Dual

$$
\begin{aligned}
\min _{w, b} & \frac{1}{2}\|w\|^2 \\
\text { s.t. } & y^{(i)}\left(w^T x^{(i)}+b\right) \geq 1, \quad i=1, \ldots, m
\end{aligned}
$$

**Lagrange function**
$$
\mathcal L(w,b,a)=\frac{1}{2}\|w\|^2-\sum_{i-1}^{m}\alpha_i[y^{(i)}\left(w^T x^{(i)}+b\right)-1]
$$

$$
\nabla_w \mathcal L(w,b,a)=w-\sum_{i-1}^{m}\alpha_i y^{(i)}x^{(i)}=0 \Rightarrow w^*=\sum_{i=1}^{m}\alpha_i y^{(i)}x^{(i)}
$$

$$
\frac{\partial}{\partial b}\mathcal L(w,b,a)=\sum_{i=1}^m \alpha_i^*y^{(i)}=0
$$

将上述条件带入$\mathcal L$中，得到
$$
\mathcal{L}(w, b, \alpha)=\sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i, j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j\left(x^{(i)}\right)^T x^{(j)}
$$
**solving the Dual:The SMO Algorithm**
$$
\max_\alpha W(\alpha)=\sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i, j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j\left(x^{(i)}\right)^T x^{(j)}
$$

$$
\text{s.t.}\ \alpha_i \ge0,i=1,\dots,m
$$

$$
\sum_{i=1}^m \alpha_i y^{(i)}=0
$$

![image-20231025112704921](img/43.png)

### The Dual Hard SVM

$$
\max_\alpha W(\alpha)=\sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i, j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j\left(x^{(i)}\right)^T x^{(j)}
$$

$$
\text{s.t.}\ \alpha_i \ge0,i=1,\dots,m
$$

$$
\sum_{i=1}^m \alpha_i y^{(i)}=0
$$

得到$\alpha^*$后代入$w^*=\sum_{i=1}^{m}\alpha_i y^{(i)}x^{(i)}$得到最优解$w^*$

![image-20231025113129561](img/44.png)

![image-20231025113202692](img/45.png)

![image-20231025113224064](img/46.png)

![image-20231025113255025](img/47.png)

* 一般情况下只有少数训练样本对应的Lagrange Multiplier大于零(支持向量)，分类面则是由这些支持向量决定
* 决策时只需计算新样本与所有支持向量的内积

### From Hard SVM to Soft SVM

$$
(w^*_{hard},b^*_{hard})=\arg\min_{w,b}\frac{1}{2}\|w\|^2
$$

$$
\text { s.t. }  y^{(i)}\left(w^T x^{(i)}+b\right) \geq 1, \quad i=1, \ldots, m
$$

$$
(w^*_{hard},b^*_{hard})=\arg\min_{w,b} \sum_{i=1}^m \ell_{0-\infin}(y^{(i)}\left(w^T x^{(i)}+b\right) \geq 1)+\frac{1}{2}\|w\|^2
$$

$$
J(\theta)=L(\theta)+\lambda R(\theta)
$$

### From Logistic Loss to Hinge Loss

$$
z=\theta^Tx=w^Tx+b
$$

$$
\text { Logistic Loss: } \ell=\left\{\begin{array}{ll}
-\log \left(\frac{1}{1+e^{-z}}\right)=\log \left(1+e^{-z}\right), & y=+1 \\
-\log \left(1-\frac{1}{1+e^{-z}}\right)=\log \left(1+e^z\right), & y=-1
\end{array} \quad \Rightarrow \ell=\log \left(1+e^{-y z}\right)\right.
$$

$$
\text{Hinge\ Loss}:\ell=\max(1-yz,0)
$$

### The Primal Soft SVM problem

Logistic loss
$$
(w^*_{hard},b^*_{hard})=\arg\min_{w,b} \sum_{i=1}^m \ell_{0-\infin}(y^{(i)}\left(w^T x^{(i)}+b\right) \geq 1)+\frac{1}{2}\|w\|^2
$$
hinge loss
$$
(w^*_{soft},b^*_{soft})=\arg\min_{w,b} C\sum_{i=1}^m \max(1-y^{(i)}\left(w^T x^{(i)}+b\right),0 )+\frac{1}{2}\|w\|^2
$$
**松弛因子**
$$
\xi_i=\max(1-y^{(i)}\left(w^T x^{(i)}+b\right),0 )
$$
**惩罚因子**$C$
$$
(w^*_{soft},b^*_{soft})=\arg\min_{w,b,\xi} C\sum_{i=1}^m \frac{1}{2}\|w\|^2+C\sum_{i=1}^m \xi_i
$$

$$
\text { s.t. }  y^{(i)}\left(w^T x^{(i)}+b\right) \geq 1-\xi_i, \quad i=1, \ldots, m
$$

$$
\xi_i \ge 0,i=1,\dots,m
$$

### The Dual Soft SVM

**primal**
$$
(w^*_{soft},b^*_{soft})=\arg\min_{w,b,\xi} C\sum_{i=1}^m \frac{1}{2}\|w\|^2+C\sum_{i=1}^m \xi_i
$$

$$
\text { s.t. }  y^{(i)}\left(w^T x^{(i)}+b\right) \geq 1-\xi_i, \quad i=1, \ldots, m
$$

$$
\xi_i \ge 0,i=1,\dots,m
$$

**dual**
$$
\max_\alpha W(\alpha)=\sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i, j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j\left(x^{(i)}\right)^T x^{(j)}
$$

$$
\text{s.t.}\ 0 \le \alpha_i \le C,i=1,\dots,m
$$

$$
\sum_{i=1}^m \alpha_i y^{(i)}=0
$$

对比hard，求解多一个惩罚因子C

