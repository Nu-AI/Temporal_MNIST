import torch
import torchvision
import numpy as np

# Temporal MNIST Parameters
seq_length = 10
num_classes = 10
samples_per_class = 1000
random_seed = 1

# Repeatable Dataset Production (remove if randomized dataset is required)
np.random.seed(random_seed)

# Generate Temporal MNIST Class Format
tMNIST_format = np.random.randint(10, size=(num_classes, seq_length))
print(tMNIST_format)

# Generate Temporal MNIST Labels
tMNIST_labels = np.random.randint(10, size=(samples_per_class*num_classes))
print(tMNIST_labels.shape)

# Generate Temporal MNIST Samples


# MNIST Parameters
batch_size_train = 64
batch_size_test = 1000

# Download MNIST Dataset
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=False)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=False)

#print(train_loader)


