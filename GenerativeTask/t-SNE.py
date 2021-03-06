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
from matplotlib import offsetbox
import matplotlib


layer_name = "fc13" # "fc12" "relu" "repara"
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

from utils import PCA
pca = PCA(X, 20)
X = pca.reduce_feature()

print(X.shape, Y.shape)
tsne = manifold.TSNE(n_components=dim, init='random', random_state=0, perplexity=100)
start_time = time.time()
X_tsne = tsne.fit_transform(X)
#降维
print(X_tsne.shape)

if dim == 2:
    #绘图
    cmap = plt.get_cmap('rainbow', 10)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, edgecolor='black', cmap=cmap)
    plt.colorbar(drawedges=True)
    plt.savefig(os.path.join(save_path_2D, "t-SNE_{}_2D_PCA_20.jpg").format(layer_name))
    plt.show()

if dim == 3:
    cmap = plt.get_cmap('rainbow', 10)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                     c=Y, cmap=cmap, edgecolor='black')
    fig.colorbar(p, drawedges=True)
    plt.show()

    # Build animation from many static figures
    build_anim = True
    if build_anim:
        angles = np.linspace(180, 360, 20)
        i = 0
        for angle in angles:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(10, angle)
            p = ax.scatter3D(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                             c=Y, cmap=cmap, edgecolor='black')
            fig.colorbar(p, drawedges=True)
            outfile = os.path.join(save_path_3D, '3dplot_step_' + chr(i + 97) + '.png')
            plt.savefig(outfile, dpi=96)
            i += 1
        call(['convert', '-delay', '50', save_path_3D + '/3dplot*', save_path_3D + '/3dplot_anim_' + '.gif'])

