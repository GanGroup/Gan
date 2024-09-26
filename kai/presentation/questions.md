#### 为什么只有生成器中的函数 log(1-D)被替换成 log(D)而判别器中的没有被替换

因为如图所示判别器的损失是 logD + log(1-D) 损失始终为1，所以不存在梯度消失的问题。

![image-20240926162328527](/Users/dengkai/Library/Application Support/typora-user-images/image-20240926162328527.png)



