# %%
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = np.genfromtxt('C:/Users/ouyangyan/Desktop/CE1/data/data.csv', delimiter=',')

# %%
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(data[:, :-1])
pca = PCA().fit_transform(data[:, :-1])

# %%
print(sum((tsne[:, 0]>35)*(data[:,-1]==0)))
print(sum((tsne[:, 0]>35)*(data[:,-1]==1)))
print(sum((tsne[:, 0]<35)*(data[:,-1]==0)))
print(sum((tsne[:, 0]<35)*(data[:,-1]==1)))

# %%
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(tsne[:, 0], tsne[:, 1], c=data[:, -1])
plt.subplot(122)
plt.scatter(pca[:, 0], pca[:, 1], c=data[:, -1])
plt.colorbar()
plt.show()
# %%
