##CloudRanger 云原生系统的 异常根本诱因识别

###Introduction

云原生系统 

- 性质

1. 容器打包
2. 动态管理
3. 微服务导向

- 特点

1. 水平设计 
2. 持续发布 
3. 抽象/模块化实施/复用
4. 组件独立拓展

- 挑战 

  微服务过多，难以管理/控制云原生系统的部署/性能

1. 难以监控微服务 - 动态
2. 难以评估整体性能以及单个微服务性能以控制性能
   1. Baselines难以人工完成
   2. 自动受限于中间件适应性和商业交易范围
3. 更长的calling path,更多的微服务参与，性能问题容易引发系统瘫痪

- CloudRanger

1. anomaly detection,
2. impact graph construction
3. correlation calculation
4. root cause identification



### Related wroks

计算机网络方面，有许多关于异常的根本诱因的研究

- 重要前提：网络结构的链接是可靠的依赖关系

- 在大规模云原生应用里不适用

  1. 云里运行节点以一种松散的方式连接，经常变化（应用规模变化/负载平衡）
  2. 流行的云设计模式进一步改变了错误增值模式

- MonitorRank 实时异常检测框架

  - 无监管 启发式
  - 随机游走策略
  - 不假定服务间依赖可靠

- 问题

  1. 以call graph为基础，在云平台中难以实现
  2. 即使有call graph，无法假定错误永远按call graph增值

- **本文解决方案**

  - data driven impact graph 
    - 代替callgraph
  - 基于第二次序随机游走的启发式调查算法 
    - 处理各种云设计模式/识别问题服务
  - 异常检测 
    - 原创算法
    - 适用于动态工作流环境
    - 无监督
    - 传统方法未考虑高次相关性


### Background and motivation

人工调查根本诱因：

- 度量分析
  - latency 比throughput更有效
  - 寻找合适时间间隔
  - 不同API，度量适用range不同
- topology调查
  - propagation
    pattern复杂，难以完全了解所有services的相关知识

**Motivation** : 人工工作难以处理复杂系统，耗时



###Problem and solution

####问题

- 多服务
- 黑盒

####CloudRanger 框架

- 根据对给定异常的潜在贡献给服务排序
- 步骤
  - 异常检测 -> API警告
  - 构建impact graph -> impact grace
  - 相关性计算/校准 -> 相关性矩阵
  - 根本诱因分析 -> 异常服务排序

####Anomaly Detection 异常检测

- 优势
  - 动态，在线学习算法跟踪动态运行设施/工作流
  - 通过反馈时间和吞吐量单独检测所有服务的性能衰退，动态缩小根诱因范围
- time window
  - 特点
    - 影响impact graph 构建
    - 计算特定服务平均返回时间/吞吐量
    - 在impact graph算法中现实两个异常有影响关系
  - 两个服务直接互相调用的统计平均延迟
  - timewindow过小（1s,2s），过多来自不同服务的异常，不同步
  - time window过大(10s)，在寻找真实错误繁殖拓扑的时候一些异常会被忽略，即使他们同步性很好
  - 结论 5s

####impact graph构建

- causal analysis

- intra-correlations

- PC算法基础上

- **Directed Acyclic Grap**有向无环

- 顶点(服务)之间有**有向边**或者**无向边**

  1. 两个有条件的依赖的服务顶点之间有边
  2. 有向边表示在所有的DAG中都有该有向边，一个服务的改变影响了（impact）另一个
  3. 无向边表示至少有一个DAG中有vi -> vj 或 vi <- vj

- 步骤

  1. 构建V间完全无向图
  2. 在重要程度阿尔法下根据邻接矩阵测试有条件独立，并把满足的边去掉
  3. 给v型结构定向
  4. 给剩余边定向

- 第二步，适用**d-separation**去定义是否条件独立

  1. 第一步得到完全无向图G，adj(G,vi)表示直接和vi相连的点集
  2. 第二步中，逐步检查一步相邻，两步相邻，是否存在vk使得（vi,vj）条件独立，如果独立去掉对应边，并把vk加入S(vi,vj)和S(vj,vi)
  3. 所有当前图中的邻接比条件集要小的时候停止，否则继续检查三步/四步相邻
  4. 重要程度参数阿尔法越大，筛选越严格，移除的边越少

- 三四步，给第二步得到的skeletion定向得到DAG

  1. 考虑所有vivk，vjvk相邻，但是vivj不相邻的三点vi vj vk，定向为vi -> vk <- vj （v型结构），其中vk不属于S(vi,vj)S(vj,vi)
  2. 应用三规则处理剩余无向边，判断两点间任意方向是否会形成新的v型结构或者环，如果形成，标记为不合法

- 三规则

  1. 对所有vi -> vj，且vi和vk都不相邻时，让**vj -> vk**（否则就会有新的v型结构）
  2. 对所有vi -> vk -> vj，让**vi -> vj**（否则就会有有向环） 
  3. 对所有vi - vk -> vj 和vi - vl -> vj，且vk vl不相邻时，让 **vi - > vj**(否则就会有新v型或者有向环)

- 复杂度

  k为任意顶点最大度，n为所需条件独立测试数目

  复杂度为 
  $$
  2(\begin{matrix} n\\k \end{matrix})\sum\nolimits_{i=0}^k (\begin{matrix} n-1\\i \end{matrix})
  $$
  比贝叶斯网络基础算法复杂度低许多



####Correlation calculation 相关性计算

- 计算异常的相似性
- revised Pearson correlation function

####Root Cause Identification 根诱因识别

- 人工方法：根据之前服务的相关性，以及当前节点相关性，跟随G图随机遍历服务
- second-order random walk即考虑前一个便利店的情况下当前便利店几率计算
- 前进/后退/停留
- 随机移动 记录每个服务访问次数



### Empirical study

#### Testbed and ecaluation metric

#####测试环境: 

- 模拟环境 Pymicro
- 真实产品 Bluemix

#####对比算法:

- Random selection
  - 最基础的缺陷搜寻方式（**无领域知识**）
  - 表现**metric-based knowledge**的重要性
- TBAC(Timing behavior Anomaly Correlation)
  - **非启发式**，致力于异常度量相关性，基于依赖关系的权重排名
  - 表现**启发式算法**的优点和健壮性
- MonitorRank
  - 启发式，**first-order**随机游走，需要目标系统的**真实调用拓扑**
  - 验证**impact graph**对比于真实服务调用拓扑的优势

#####算法性能量化：

性能度量 每个算法给出的最上k个结果成功预测真实异常根诱因的几率 **AC@k**

k越小，AC@k越大，说明算法越好，越能准确预测，最小化诱因范围



#### Simulation experiment and analysis





### Conclusion

CloudRanger在正确率，速度上完胜。

对比现行（state-of-art）方法，优势有：

1. 黑盒，无需背景知识，更广应用场景
2. 易于集成领域知识，维护经历，只需替换相关性函数或者提前定义拓扑关系（用历史观察以及当前度量的模拟替换相关值函数）
3. 高适用型，可以拓展到云原生系统中的更多场景，比如诊断分许其他复杂网络（传感/社会/生物网络）

