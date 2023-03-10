# Lib Load
import os
import cv2
import glob
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

class TinyImageNetDataset(Dataset):
    def __init__(self, path, transforms, is_train, input_size=(224, 224)):
        super(TinyImageNetDataset, self).__init__()
        self.transforms = transforms
        self.is_train = is_train
        self.input_size = input_size
        with open(path + '/wnids.txt', 'r') as f:
            self.label_list = f.read().splitlines()

        if is_train:
            self.data = glob.glob(path + '/train/*/images/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)

        else:
            self.data = glob.glob(path + '/val/images/*.JPEG')
            self.val_list = dict()
            with open(path + '/val/val_annotations.txt', 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.resize(cv2.imread(img_file), self.input_size)
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]
        transformed = self.transforms(image=img)['image']
        return transformed, label

class TinyImageNet(pl.LightningDataModule):
    def __init__(self, path, workers, transforms, input_size=(224, 224), batch_size=None):
        super(TinyImageNet, self).__init__()
        self.path = path
        self.train_transforms = transforms
        self.batch_size = batch_size
        self.workers = workers
        self.input_size = input_size
    
    def train_dataloader(self):
        return DataLoader(TinyImageNetDataset(self.path,
                                              transforms=self.train_transforms,
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
        return DataLoader(TinyImageNetDataset(self.path,
                                              transforms=val_transform,
                                              is_train=False,
                                              input_size=self.input_size),
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0,
                          )


if __name__ == '__main__':
    '''
    Dataset Loader Test
    run$ python -m dataset.classification/tiny_imagenet
    '''

    import albumentations
    import albumentations.pytorch
    from utils.visualize import visualize

    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2()])

    loader = DataLoader(TinyImageNetDataset(path='../../datasets/tiny-imagenet-200/', transforms=train_transforms, is_train=True))
    for batch, sample in enumerate(loader):
        visualize(sample[0], sample[1])
        break