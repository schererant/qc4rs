#%%
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

#%%
def plot(x_index, y_index, iris_data):
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
    plt.scatter(iris_data.data[:, x_index], iris_data.data[:, y_index], c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris_data.feature_names[x_index])
    plt.ylabel(iris_data.feature_names[y_index])

iris = load_iris()
plt.figure(figsize=(14, 4))
plt.subplot(121)
plot(0, 1, iris)
plt.subplot(122)
plot(2, 3, iris)