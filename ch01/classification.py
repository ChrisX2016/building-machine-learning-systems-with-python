from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
for raw in data:
    print(raw)
    print('-----')
features = data['data']
feature_names = data['feature_names']
target = data['target']
labels = data['target_names'][target]
for t,marker,c in zip(range(3),">ox","rgb"):
    # We plot each class on its own to get different colored markers
    plt.scatter(features[target == t,0],
                features[target == t,1],
                marker=marker,
                c=c)
plt.grid()
plt.show()

plength = features[:, 2]
# use numpy operations to get setosa features
is_setosa = (labels == 'setosa')
# This is the important step:
max_setosa =plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))

# print (features[:,2])
# if features[:,2] < 2:
# 	print ("Iris Setosa")
# else:
# 	print("Iris Virginica or Iris Versicolour")

features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')

best_acc = -1.0
for fi in range(features.shape[1]):
    thresh = features[:,fi].copy()
    thresh.sort()
    for t in thresh:
        pred = (features[:,fi]>t)
        acc = (pred==virginica).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t

print("best_acc",best_acc)
print("best_fi",fi)
print("best_t",t)

