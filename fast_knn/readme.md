In this task we use KNN to perform the task in hand but we use many different ways of finding the optimal 'k'. In this work we propose a Fast_knn algorithm that utilizes binary search method to get the optimal k.


The findings obtained are as follows

Data prepared

**Best k_using ELBO: 15**

**Training Error for Best k: 0.15378405650857718**

**Test Error for Best k: 0.17352703793381763**

Accuracy score:0.8264729620661824

Number of updates is: 4955

**Time taken by elbo method: 3.537790060043335**

Get the accuracies: 0.8200161420500404 ==> 0.8159806295399515 for k: 8

Get the accuracies: 0.8240516545601292 ==> 0.8232445520581114 for k: 12

Get the accuracies: 0.8256658595641646 ==> 0.8248587570621468 for k: 14

Get the accuracies: 0.8248587570621468 ==> 0.8264729620661824 for k: 15

**Best k using fast_knn is: 15**

**Time taken by fast-knn method: 1.8269157409667969**
