### 运行说明

- python >=3.5
- pytorch = 1.0.1
- tensorboardX = 1.7

### 文件说明

- \data(存放着原始数据)
- \demo(pytorch上的DQN教程，[link](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html))
- \log(训练的loss、reward等相关内容,用tensorboardX记录)
- \memory(存放训练时的数据，即训练过程中生成的数据，记录了收敛的数据(success.pkl)和不收敛的数据(data.pkl))
- \model(模型存放)
- \run(跑WMLFRTMsg的环境)
- \template(存放不收敛的初始样本)
- *.py

### 数据说明

#### **State**

$S=\{G, L\}$,

其中，$G={g_i, i∈[0,N_g ]}, L={l_i, i∈[0,N_l ]}$,分别代表发电机的集合和负荷的集合

$g_i,l_i=[p,q ]$,每台发电机/负荷考虑两个参数，有功功率和无功功率

*\*注\*：在6958节点的电网中，去除掉了无功功率，只保留了有功功率*

$N_g,N_l$分别代表发电机和负荷的数量



我们可以将$S$抽象成一个$2 *(N_g+N_l)$的向量，2代表有功功率和无功功率两个特征,即$S =\left[
\begin{matrix}
p_{g_0}  & ... & p_{g_{N_g}} & p_{l_0}& ...&p_{l_{N_l}}\\
q_{g_0} & ... & q_{g_{N_g}} & q_{l_0}& ...&q_{l_{N_l}}
\end{matrix}
\right]$ 

•通过修改发电机特征达到修改电网状态的目的

•仅修改发电机是为了简化问题



#### Action

$A={a_i, i∈[0,N_a ]}, N_a=N_g∗2∗2∗4$代表每个状态的动作候选集

2代表正负和有功功率、无功功率两种特征

4代表调整特征的值，$v∈[0.1,0.5,1,2]$

*\*注\*只考虑有功功率的时候*，$N_a=N_g*1*2∗4$



#### Reward

$R=[-1,1,-0.01]$ 代表越界、收敛和其它情况



### 自定义环境

需要一个env和agent交互的环境接口，由*env.py*和*TrendData.py*完成，  

*env.py*，包含了环境改变的接口，主要的接口和*gym*库的env类一致，详见*env.py*

*TrendData.py*,包含数据交互的具体细节，读入LF.L\*文件，将其存储在内存当中，开放接口更改文件内容(写入run文件夹中的LF.L\*文件),同步更新state，具体见*TrendData.py*细节





### 训练

用的是double-DQN方法，详见train.py

相关参数放在const.py当中



### 搜索策略

概率随机探索，随着步长的增大，随机探索的概率指数下降



#### Loss

smooth_l1_loss



#### 优化器

RMSprop

