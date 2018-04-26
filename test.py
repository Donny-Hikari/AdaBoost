
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.utils import shuffle

import time

Use_Sklearn_AdaBoost = False
Estimators = 60
Iteration_Steps = 60

if Use_Sklearn_AdaBoost:
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
else:
    from adaboost import AdaBoostClassifier, DecisionStumpClassifier

X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)

X = np.concatenate((X1, X2))
y = np.concatenate((y1, 1 - y2))

X, y = shuffle(X, y, random_state=1)

trainning_time = 0
if Use_Sklearn_AdaBoost:
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                            algorithm="SAMME",
                            n_estimators=200)
    bdt.fit(X, y)
else:
    bdt = AdaBoostClassifier(Estimators, DecisionStumpClassifier(Iteration_Steps))
    tbegin = time.time()
    bdt.train(X, y)
    tend = time.time()
    trainning_time = tend - tbegin

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

fig = plt.figure(figsize=(15, 5))

# Plot the decision boundaries
plt.subplot(131)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
if Use_Sklearn_AdaBoost:
    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
else:
    Z, _ = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the trainning points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

if Use_Sklearn_AdaBoost:
    yPred = bdt.predict(X)
    CI = bdt.predict_proba(X)
else:
    yPred, CI = bdt.predict(X)

accuracy = np.mean(yPred == y)

if Use_Sklearn_AdaBoost:
    fig.canvas.set_window_title('Sklearn AdaBoost Test - accuracy: %0.3f' % accuracy)
else:
    fig.canvas.set_window_title('AdaBoost Test' + ' - '
        + 'estimators: %d, iteration steps: %d, accuracy: %0.3f, trainning time: %0.4f' %
        (Estimators, Iteration_Steps, accuracy, trainning_time))

# Plot the two-class decision scores
if Use_Sklearn_AdaBoost:
    twoclass_output = bdt.decision_function(X)
else:
    twoclass_output = bdt.weightedSum(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(132)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
             bins=10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5,
             edgecolor='k')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

# Plot the two-class confidence
confidence = CI # bdt.decision_function(X)
plot_range = (confidence.min(), confidence.max())
plt.subplot(133)
plt.hist(confidence,
         bins=10,
         range=plot_range,
         facecolor='b',
         alpha=.5,
         edgecolor='k')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.ylabel('Samples')
plt.xlabel('Confidence')
plt.title('Confidence Distrubution')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()
