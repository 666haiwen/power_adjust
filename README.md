### 运行说明

- python >=3.5

- pytorch = 1.0.1

- tensorboardX = 1.7


### 文件说明

- env\ 存放着和电科院WMLFRTMsg.exe交互的环境
- utils\存放着在RL过程中，可能用到的辅助代码，分别为replayMemory和openAI所采用的参数随机化
- 需要将群文件data.rar解压放到env目录下，其解压后的目录应为：

```
\env
	\data
	\run
	\template
	__init__.py
	const.py
	env.py
	test.py
	TrendData.py
	utils.py
```



### ENV说明

#### ENV文件说明

**const.py**:环境的配置参数，包括特征量的多少，目标断面是多少，目标值的范围以及计算值与值之间的接近程度的函数

**env.py:**与agent交互的接口环境

**TrendData.py**：env.py的内置操作，主要是为了使用电科院exe程序而完成的一系列文件——内存读取操作

**utils.py**:将36节点数据分成一个json接口供env.py使用

---

潮流问题主要跟这么几个元件相关：

发电机、负荷、电容电抗器；

针对一个特定的电网，母线以及拓扑结构往往是没有太大差异(或者完全一样的)

区别在于，其电网的设置不同，具体表现在发电机、负荷、电容电抗器的参数不同；

在这些参数当中，我们一般可以修改的内容为：发电机的有功出力(Pg)、无功出力(Qg)；

电容电抗器、发电机的挂载(即有效标志位0/1)

其中、负荷参数、电容电抗器的相关参数是不能修改的，其中电容电抗器的参数在一个电网中往往都是不变的，只有挂载与否不同，因此，暂定在潮流问题中的state和action如下设置

#### 1.潮流收敛调整和潮流状态电压调整

这两个的state和action是一样的，reward收敛的给出了，电压调整的再说，可以自己设定

**State**

$S=\{G, L,AC\}$,

其中，$G={g_i, i∈[0,N_g ]}, L={l_i, i∈[0,N_l ]},AC={ac_i,i\in[0,N_{ac}]}$,分别代表发电机的集合、负荷的集合以及电容电抗的集合

$g_i,l_i=[p,q ]$,每台发电机/负荷考虑两个参数，有功功率和无功功率

$ac_i$代表该电容电控的集合

$N_g,N_l,N_{ac}$分别代表发电机、负荷和电容电抗的数量



我们可以将$S$抽象成一个$2*(N_g+N_l + N_{ac})$的向量，2代表有功功率和无功功率两个特征,电容电抗器的mark标志重复一遍$S =\left[
\begin{matrix}
p_{g_0}  & ... & p_{g_{N_g}} & p_{l_0}& ...&p_{l_{N_l}}&ac_{mark_0}&...&ac_{mark_{N_{ac}}}\\
q_{g_0} & ... & q_{g_{N_g}} & q_{l_0}& ...&q_{l_{N_l}}&ac_{mark_0}&...&ac_{mark_{N_{ac}}}
\end{matrix}
\right]$ 



**Action**

$A={a_i, i∈[0, 2 * N_{ac} ]},$

2代表挂载和不挂载

**Reward**

$R=[1,-0.01]$ 代表收敛和没有收敛



#### 2.潮流状态调整——断面

断面简单来看，就是一个电网的拓扑结果，通过一个断面(一组线的集合)，将拓扑图可以分成两个连通子图.

因为在实际操作过程中，断面往往是给定不变的，所以这里直接给定了断面是由哪几条线组成，以及这个断面应该达到的额定功率.

**State**

$S=\{G, L,AC\}$,

其中，$G={g_i, i∈[0,N_g ]}, L={l_i, i∈[0,N_l ]},AC={ac_i,i\in[0,N_{ac}]}$,分别代表发电机的集合、负荷的集合以及交流线的集合

$g_i,l_i,ac_i=[p,q ]$,每台发电机、负荷、交流线考虑两个参数，有功功率和无功功率；其中交流线分为i侧和j侧，这里我们取了i侧的有功功率和无功功率

$N_g,N_l,N_{ac}$分别代表发电机、负荷和交流线的数量



我们可以将$S$抽象成一个$2*(N_g+N_l + N_{ac})$的向量，2代表有功功率和无功功率两个特征,$S =\left[
\begin{matrix}
p_{g_0}  & ... & p_{g_{N_g}} & p_{l_0}& ...&p_{l_{N_l}}&p_{ac_0}&...&p_{ac_{N_{ac}}}\\
q_{g_0} & ... & q_{g_{N_g}} & q_{l_0}& ...&q_{l_{N_l}}&q_{ac_0}&...&q_{ac_{N_{ac}}}
\end{matrix}
\right]$ 



**Action**

措施是调整发电机的有功出力，

离散化了调整的方向和值，分为别[-1, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1]

$A={a_i, i∈[0, 8 * N_{g} ]},$

**Reward**

state的断面值为x，

action之后的next_state的断面值为y.

目标值为target

reward = proximity(y) - proximity(x)

其中proximity(x)代表了x与目标值target的接近程度



#### ENV使用说明

见test.py

