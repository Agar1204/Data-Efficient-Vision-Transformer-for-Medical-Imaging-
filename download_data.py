from medmnist import PathMNIST
import numpy as np

# Download PathMNIST dataset
# split=train is misleading, as it downloads the entire dataset (train, val, test) 
# and saves it as a single numpy file
PathMNIST(split='train', download=True, root="data", size=128)
data = dict(np.load("data/pathmnist_128.npz"))

# Store means and standard deviations of each RGB channel in the dataset 
# to later center the data around zero for normalization during training 
train_images = data['train_images'].astype(np.float32) / 255.0
data["channel_means"] = np.mean(train_images, axis=(0, 1, 2))
data["channel_stds"] = np.std(train_images, axis=(0, 1, 2))
np.savez("data/pathmnist_128.npz", **data)
