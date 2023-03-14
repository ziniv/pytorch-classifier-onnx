import os
import pickle
import glob
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

class Cifar10Dataset(Dataset):
    def __init__(self, data_path, transforms, is_train, input_size=(32, 32)) -> None:
        super(Cifar10Dataset, self).__init__()
        
        meta_datapath = os.path.join(data_path, 'batches.meta')
        batch_datapath = glob.glob(os.path.join(data_path, 'data_batch*'))
        test_datapath = glob.glob(os.path.join(data_path, 'test_batch'))
        
        self.transforms = transforms
        self.is_train = is_train
        self.input_size = input_size
        self.classes = []
        with open(meta_datapath, 'rb') as meta_fo:
            data = pickle.load(meta_fo, encoding='latin1')
            self.classes = data['label_names']
        self.data_list, self.label_list = self.unpacking_data(batch_datapath if self.is_train else test_datapath)
        self.indexes = None
    
    
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    
    def unpacking_data(self, file_list):
        data_list = []
        label_list = []
        for i, data in enumerate(file_list):
            train_file = self.unpickle(data)
            train_data = train_file[b'data']
            train_data_reshape = np.vstack(train_data).reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
            train_labels = train_file[b'labels']
            data_list = train_data_reshape if i == 0 else np.concatenate((data_list, train_data_reshape), axis=0)
            label_list = train_labels if i == 0 else np.concatenate((label_list, train_labels), axis=0)
        return data_list, label_list

    
    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, index):
        img = self.data_list[index]
        label = self.label_list[index]
        transformed = self.transforms(image=img)['image']
        return transformed, label


class Cifar10(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, workers, transforms, input_size=(32, 32)) -> None:
        super(Cifar10, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.batch_size = batch_size
        self.workers = workers
        self.input_size = input_size
    
    def prepare_data(self):
        pass
    
    def train_dataloader(self):
        return DataLoader(Cifar10Dataset(
                                        data_path=self.data_path,
                                        transforms=self.transforms,
                                        is_train=True,
                                        input_size=self.input_size),
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0,
                          )
    
    def val_dataloader(self):
        val_transform = albumentations.Compose([
            albumentations.Normalize(0, 1), 
            ToTensorV2()
        ])
        return DataLoader(Cifar10Dataset(
                                        data_path=self.data_path,
                                        transforms=val_transform,
                                        is_train=False,
                                        input_size=self.input_size),
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0,
                          )


if __name__ == "__main__":
    
    from utils.visualize import visualize
    
    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.Normalize(0, 1),
        ToTensorV2()])
    
    loader = DataLoader(Cifar10Dataset(data_path='../../datasets/cifar-10-batches-py/', transforms=train_transforms, is_train = True))
    
    for batch, sample in enumerate(loader):
        visualize(sample[0], sample[1])
        break
    
    




