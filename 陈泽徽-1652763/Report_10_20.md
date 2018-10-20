## Report for 10.20





### 1. Online Anomaly Detection on the Webscope S5 Dataset: A Comparative Study

In this paper, a relatively Simple Online Regression Anomaly Detector is proposed which is quite successful compared to other anomaly detection algorithm on Webscope S5 dataset. Since I mainly focus on algorithm in this project, most of my notes are related to algorithm implementation. 

- Methods

  - Feature Generation Using Sliding Windows(What we've already used in our works)

  - Offline Regression Anomaly Detection (Offline-RAD)

    ![img](/Users/kirito/Documents/aiops-2018-anomaly-detection/%E9%99%88%E6%B3%BD%E5%BE%BD-1652763/fig/offline_rad.png )

  - Specification for e-quantile removement

    ![img](/Users/kirito/Documents/aiops-2018-anomaly-detection/%E9%99%88%E6%B3%BD%E5%BE%BD-1652763/fig/e_method.png)

  - Online Regression Anomaly Detection(Online-RAD)

  ![img](/Users/kirito/Documents/aiops-2018-anomaly-detection/%E9%99%88%E6%B3%BD%E5%BE%BD-1652763/fig/online_rad.png)

- Summary

  This paper proposed an algorithm that outperforms benchmarks by a large margin. The interesting thing is, at the begining of the paper, the author thinks most of anomaly detection algorithms lacks of robustness which means they can not achieve statistying performance on various dataset. They attempts to gives us an scalable and adaptive algorithm but end up with only testing it on Webscope S5 dataset. So how can I judge if this is algorithm is scalable and adaptive?

  Another problem is that the algorithm proposed in this paper is totally different from methods we will used in our future work. The former one mainly focus on traditional machine learning algorithm utilizing regression and density estimation, on the contrary, we want to develop an end-to-end, neural network based framework to solve this problem.
