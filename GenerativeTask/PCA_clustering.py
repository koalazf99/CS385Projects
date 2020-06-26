"""
t-SNE dimension compression for MNIST dataset
author: zf
2020/6/11
"""
from subprocess import call
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
import os
from sklearn.datasets import load_digits
from sklearn import manifold
from sklearn.cluster import KMeans
from utils import PCA
from matplotlib import offsetbox
import matplotlib


save_dir = "./PCA_compare/"
os.makedirs(save_dir, exist_ok=True)
layer_name = "relu" # "repara" "fc12" "fc13"
save_path_2D = "./t-SNE_result/2Dresults"
save_path_3D = "./t-SNE_result/" + layer_name + "_PCA"
if not os.path.exists(save_path_2D):
    os.mkdir(save_path_2D)
if not os.path.exists(save_path_3D):
    os.mkdir(save_path_3D)
X = np.load("./t-SNE_feature/mnist_encoder_{}_feature.npy".format(layer_name))
Y = np.load("./t-SNE_feature/mnist_label.npy")
X = X[:2000, :]
Y = Y[:2000]
dim = 2
X = (X - np.min(X)) / (np.max(X) - np.min(X))
print(X.shape, Y.shape)
pca = PCA(data=X, n_components=dim)
x_pca = pca.reduce_feature()
eigenvalue = pca.get_feature()
res = []
for item in eigenvalue:
    # print(item)
    res.append(item['w'])
x = [i for i in range(len(res))]
a = plt.plot(x, res, label=layer_name)
plt.legend()
plt.savefig(os.path.join(save_dir, layer_name + ".png"))


