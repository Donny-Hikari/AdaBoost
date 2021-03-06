
## Introduction

AdaBoost is short for Adaptive Boosting. Boosting is one method of Ensemble Learning. There are other Ensemble Learning methods like Bagging, Stacking, etc.. The differences between Bagging, Boosting, Stacking are:

1. Bagging:  
   Equal weight voting. Trains each model with a random drawn subset of training set.

2. Boosting:  
   Trains each new model instance to emphasize the training instances that previous models mis-classified. Has better accuracy comparing to bagging, but also tends to overfit.

3. Stacking:  
   Trains a learning algorithm to combine the predictions of several other learning algorithms.

## The Formulas

Given a N*M matrix X, and a N vector y, where N is the count of samples, and M is the features of samples. AdaBoost trains T weak classifiers with the following steps:

1. Initialize.

![F-1-1 Initialization](doc/img/F-1-1.png)

2. Train i-th weak classifier with training set {X, y} and distribution W_i.

3. Get the predict result \\( h_i \\) on the weak classifier with input X.

![F-1-2 Predict Inputs](doc/img/F-1-2.png)

4. Update.

![F-1-3 Update](doc/img/F-1-3.png)

Z is a normalization factor.

5. Repeat steps 2 ~ 4 until i reaches T.

6. Output the final hypothesis:

![F-1-4 Output](doc/img/F-1-4.png)

## Test

Using a demo from [sklearn AdaBoost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), I got the following result.

Weak classifiers: 200; Iteration steps in each weak classifier: 200:

![Result of my AdaBoost, 200-200](doc/img/result-200-200.png)

Weak classifiers: 60; Iteration steps in each weak classifier: 60:

![Result of my AdaBoost, 60-60](doc/img/result-60-60.png)

Weak classifiers: 400; Iteration steps in each weak classifier: 400:

![Result of my AdaBoost, 400-400](doc/img/result-400-400.png)

We can see the result varies as the number of weak classifiers and the iteration steps change.

Compares with the AdaBoostClassifier from sklearn with 200 estimators (weak classifiers):

![Result of sklearn AdaBoost, 200](doc/img/result-sklearn.png)

More comparison for my AdaBoostClassifier with different parameters:

| estimators | iteration steps |  time | accuracy |
|-----------:|----------------:|------:|---------:|
|30          |30               | 0.0621|0.8540    |
|30          |60               | 0.1095|0.8760    |
|30          |200              | 0.3725|0.8620    |
|30          |400              | 0.7168|0.8720    |
|60          |30               | 0.1291|0.8620    |
|60          |60               | 0.2328|0.8600    |
|60          |200              | 0.7886|0.8780    |
|60          |400              | 1.4679|0.8840    |
|200         |30               | 0.3942|0.8600    |
|200         |60               | 0.7560|0.8700    |
|200         |200              | 2.4925|0.8900    |
|200         |400              | 4.7178|0.9020    |
|400         |30               | 0.8758|0.8640    |
|400         |60               | 1.6578|0.8720    |
|400         |200              | 5.0294|0.9040    |
|400         |400              |10.0294|0.9260    |

## References / Acknowledgement

1. [AdaBoost - Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
2. [AdaBoost -- 从原理到实现](https://blog.csdn.net/Dark_Scope/article/details/14103983)
3. [A Short Introduction to Boosting](https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf)
4. [Multi-class AdaBoost](https://web.stanford.edu/~hastie/Papers/samme.pdf)

## Author
[Donny](https://github.com/Donny-Hikari)

Find out further information, refer to my blog [AdaBoost - Donny](http://konno-yuuki.com/blog/posts/machinelearning/2018/654416/) (In Chinese & English).
