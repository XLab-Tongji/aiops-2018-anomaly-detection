#### Problem  Description

​	1、most methods lack flexibility, either has too complex structure or training settings or is domain-	

​              specific that  keeps it out of production

   	    

​              1.1 before-and-afterwards tricks to join domain knowledge, compromise should be made between 

​	            time and  AI performance, let a adjustment exists that when human resource is enough certain 

​		    accuracy can be reduced to trade for a better recall score

   	

​	2、it appears to be difficult to raise an understandable logic that an transform log from different    

​              domain into  an uniform input-domain for a standard solution to eliminate the domain difference

​	     

​              2.1 feed-guided online training is being studied to make up for the  lack of knowledge

​	       

#### Innovation Point

​	1、what can Deep Learning bring us, when log data are normally in large-scale, does the 

​	      complexity of the network necessarily mean better performance in domain-specific feature   

​              auto-dig 

​	2、the champion's network architecture is like the MNIST tutorial of baseline I've learned. does it mean 

​	      that the feature engineering work has already made the **web-service log data** highly identifiable

​	3、based on the above mentioned two questions:

​		1、model iterative learning ——> how to make use of learning loss definition to 		

​	     	      improve certain learning performance elements. dynamic/(+)parameter/level

​		2、data difference and redundancy in re-expression

​		3、train labels' reliability [arXiv:1011.1669v3]

​	4、earthquake theory

​	5、gray regions



##### Todo-Experiment

1、对数据进一步的观察，在时间上观察是否具有含义，如果有可以引入类似平均宕机间隔时间，越接近越提高，根据数据点的采样间隔，调整方法?

2、对数据进一步的观察，因为KPI是脱敏人工标注的数据，算是人可识别，那么，是否可以应用卷积网络，还可以通过并行，比LSTM要好？？冠军方案主要网络结构并不复杂，是否是其有专业的观察者，进而融入了领域知识进行特征抽取。Bi-Directional 的不好在哪里

3、迄今为止，所有的方法都是非黑即白，无论是仅构造正常数据模型进而设置偏差阈值，还是解决了类别不平衡，进行二分类。score-based，class-based，什么应用场景常用score？之前看的KDD文章是score-based，通过人工筛查top-ranked的正误，打标，进一步训练。我们的数据集使class-based，那么在class-based里如何让机器可靠报正常，设置宽泛的阈值，使recall 100%，查看accuracy，在框定的范围内把FN,FP挑出来，在容易混淆的数据里进一步看，影响因子。

