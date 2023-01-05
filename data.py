import torch
import os
# import ros.getcwd()e
import numpy as np

# train_files = [x for x in os.listdir('corruptmnist') if re.match('train.*',x)]
# test_file = [x for x in os.listdir('corruptmnist') if re.match('test.*',x)]

# class MyDataset(Dataset):
#   def __init__(self, *filepaths):
#     content = [np.load('corruptmnist/'+f) for f in filepaths]
#     self.imgs, scelf.labels = _concat_content(content)
  
#   def __len__(self):
#     return self.imgs.shape[0]

#   def __getitem__(self, idx):
#     return (self.imgs[idx], self.labels[idx])


def mnist():
    # exchange with the corrupted mnist dataset
    train_ims = torch.tensor(np.array([np.load(os.getcwd()+f"/corruptmnist/train_{i}.npz")["images"] for i in range(5)]).reshape(-1,28,28))
    train_labels = torch.tensor(np.array([np.load(os.getcwd()+f"/corruptmnist/train_{i}.npz")["labels"] for i in range(5)]).reshape(-1))
    test_ims = torch.tensor(np.load(os.getcwd()+f"/corruptmnist/test.npz")["images"])
    test_labels = torch.tensor(np.load(os.getcwd()+f"/corruptmnist/test.npz")["labels"])
    return list(zip(train_ims, train_labels)), list(zip(test_ims, test_labels))


