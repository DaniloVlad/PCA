import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

raw_data = np.fromfile("faces.dat", sep=" ")

#Reshape the data and display it
data = raw_data.reshape((400, 4096))
print("Displaying 100th image")
plt.imshow(data[100].reshape(64, 64), cmap="gray")
plt.show()

#subtract the data mean from the data to center the values
n_samples = data.shape[0]
data_mean = data.mean()
data_centered = data - data_mean
print(data_centered.shape)
print("Mean of the data: ", data_mean)

#Display the picture after being centered
print("Displaying 100th image after centering")
plt.imshow(data_centered[100].reshape((64,64)), cmap="gray")
plt.show()

#compute the pca of all 400 components
pca = PCA()
pca.fit(data_centered)
print("Last Eigen Value: ", pca.explained_variance_[399])
principle_componenets = pca.components_
eigen_values = pca.explained_variance_
eig_range = np.arange(400)
#Sort the data, although it will already be sorted!
eig_range = eig_range[np.argsort(eigen_values)[::-1]]
eigen_values = np.sort(eigen_values)[::-1]
#plot the eigen values
plt.plot(eig_range, eigen_values)
plt.title("Eigen Values vs Component")
plt.show()

# pic1 = pca.inverse_transform(data_centered)
comps = principle_componenets
plt.subplot(1, 5, 1)
plt.imshow(comps[0].reshape(64, 64), cmap="gray")

plt.subplot(1, 5, 2)
plt.imshow(comps[1].reshape(64, 64), cmap="gray")

plt.subplot(1, 5, 3)
plt.imshow(comps[2].reshape(64, 64), cmap="gray")


plt.subplot(1, 5, 4)
plt.imshow(comps[3].reshape(64, 64), cmap="gray")


plt.subplot(1, 5, 5)
plt.imshow(comps[4].reshape(64, 64), cmap="gray")

plt.show()
eig_sum = np.sum(eigen_values)

#plot the cummulative sum of the ratios
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Variance vs D-Components")
plt.show()

#Compute pca for 10, 100, 200, 399 components
comp1 = PCA(n_components = 10)
comp_tran = comp1.fit_transform(data_centered)
rev = comp1.inverse_transform(comp_tran)
plt.imshow(rev[100].reshape(64, 64), cmap="gray")
plt.show()


comp1 = PCA(n_components = 100)
comp_tran = comp1.fit_transform(data_centered)
rev = comp1.inverse_transform(comp_tran)
plt.imshow(rev[100].reshape(64, 64), cmap="gray")
plt.show()

comp1 = PCA(n_components = 200)
comp_tran = comp1.fit_transform(data_centered)
rev = comp1.inverse_transform(comp_tran)
plt.imshow(rev[100].reshape(64, 64), cmap="gray")
plt.show()

comp1 = PCA(n_components = 399)
comp_tran = comp1.fit_transform(data_centered)
rev = comp1.inverse_transform(comp_tran)
plt.imshow(rev[100].reshape(64, 64), cmap="gray")
plt.show()