import numpy as np
import os
from medmnist import PathMNIST


def load_data():
    """
    Download PathMNIST if needed, compute channel statistics, and return data.
    Returns: train_images, train_labels, val_images, val_labels, 
             test_images, test_labels, channel_means, channel_stds
    """
    npz_path = "data/pathmnist_128.npz"
    
    # Download if file doesn't exist (one call gets everything)
    if not os.path.isfile(npz_path):
        print("Downloading PathMNIST dataset...")
        PathMNIST(split='train', download=True, root="data", size=128)
    
    # Load the .npz file
    with np.load(npz_path) as data:
        # Check if stats exist, if not compute them
        if "channel_means" not in data or "channel_stds" not in data:
            print("Computing channel statistics...")
            train_images = data['train_images']
            random_indices = np.random.choice(len(train_images), size=5000, replace=False)
            sample = train_images[random_indices]
            
            means = np.mean(sample, axis=(0, 1, 2)) / 255.0
            stds = np.std(sample, axis=(0, 1, 2)) / 255.0
            
            # Save stats back
            data_dict = {k: v for k, v in data.items()}
            data_dict['channel_means'] = means
            data_dict['channel_stds'] = stds
            
            np.savez(npz_path, **data_dict)

        print("Data loaded!")
        return (data['train_images'], data['train_labels'],
                data['val_images'], data['val_labels'],
                data['test_images'], data['test_labels'],
                data['channel_means'], data['channel_stds'])
