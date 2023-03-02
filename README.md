# Deep-UCB
This repository contains the codes for our paper, which is in submission.
## Abstract
From the perspective of the deep reinforcement learning algorithm, the training effect of the agent will be affected because of the excessive randomness of the ε-greedy method. This paper proposes a novel action decision method to replace the ε-greedy method and avoid excessive randomness. First, a confidence bound span fitting model based on a deep neural network is proposed to fundamentally solve the problem that UCB cannot estimate the confidence bound span of each action in high-dimensional state space. Then, a confidence bound span balance model based on target value in reverse order is proposed. The parameters of the U network are updated after each action decision using the backpropagation of the neural network to balance the confidence bound span. Finally, an exploration-exploitation dynamic balance factor α is introduced to balance exploration and exploitation in the training process. Experiments are conducted using the Nature DQN and Double DQN algorithms, and the results demonstrate that the proposed method achieves higher performance than the ε-greedy method under the basic algorithm and experimental environment of this paper. The method presented in this paper has significance for applying a confidence bound to solve complex reinforcement problems.
## Hardware and Software Environment
-	CPU: 32 bits or more.
-	Hard Disk: Recommended over 1G for Pycharm installation.
-	Memory: 2G or more.
-	OS：Windows 10.
-	Utility Software：Pycharm.
## Operating Instructions
Before running this system, PyCharm, Pytorch, Gym, and other necessary Python libraries need to be installed in advance.  
1.Double-click the PyCharm icon to start the program, as shown in Figure 1. The PyCharm program after startup is shown in Figure 2.  
2. Copy the file Confidence Interval Based Intensive Learning System V1.0 into any .py file, click File->Open in the PyCharm interface, and open the .Py file, as shown in Figure 3.  
3. Right-click, and then click the Run (File Name) button to run the system, as shown in Figure 4.  
4. The system works as shown in Figure 5.  
5. The illustration is Pong, an reinforcement learning environment in the gym library. You can choose another environment for training or use a locally built intensive learning environment.
## Acknowledgments
This work was supported by the National Natural Science Foundation of China (Grant No. 62073245), the Natural Science Foundation of Shanghai (20ZR1440500), and Pudong New Area Science & Technology Development Fund (PKX2021-R07).
