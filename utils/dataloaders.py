import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import pandas as pd
from PIL import Image

class HandwritingDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform = None, target_transform = None):
        self.df = pd.read_csv(csv_path, header = None)
        self.transform = transform
        self.target_transform = target_transform
        self.x = np.asarray(self.df.iloc[:len(self.df),1:]).reshape([len(self.df),28,28]) # taking all columns expect column 0
        self.x = self.x.astype('uint8')
        self.y = np.asarray(self.df.iloc[:len(self.df),0]).reshape([len(self.df)]) # taking column 0

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        target = self.y[index]
        image = self.x[index]
        PIL_image = Image.fromarray(image)
        if self.transform is not None:
            PIL_image = self.transform(PIL_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return PIL_image, target


def get_handwriting_operators_dataloaders(batch_size=128, 
                                          path_to_train_csv='/Users/aashishkumar/Documents/notebooks/handwriting_operators_train_temp.csv',
                                          path_to_test_csv='/Users/aashishkumar/Documents/notebooks/handwriting_operators_test_temp.csv'):
    """ Handwriting Operators dataloader with (32, 32) images 
    handwriting_operators_train_temp.csv this file does not have bg class, 12 classes in total """

    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = HandwritingDataset(path_to_train_csv, transform=all_transforms)
    test_data = HandwritingDataset(path_to_test_csv, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def get_handwriting_letters_dataloaders(batch_size=128, 
                                          path_to_train_csv='/Users/aashishkumar/Documents/Projects/forked_repos/no_cuda/IB-INN/handwriting_letters_train.csv',
                                          path_to_test_csv='/Users/aashishkumar/Documents/Projects/forked_repos/no_cuda/IB-INN/handwriting_letters_test.csv'):
    """ Handwriting Operators dataloader with (32, 32) images 
    handwriting_letters_train.csv file has 26 classes only. No bg class, no mirror class """

    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = HandwritingDataset(path_to_train_csv, transform=all_transforms)
    test_data = HandwritingDataset(path_to_test_csv, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def get_emnist_uppercase_dataloaders(batch_size=128, 
                                          path_to_train_csv='/Users/aashishkumar/Documents/notebooks/emnist_uppercase_train_3rd_May_2021.csv',
                                          path_to_test_csv='/Users/aashishkumar/Documents/notebooks/emnist_uppercase_test_3rd_May_2021.csv'):
    """ Handwriting Operators dataloader with (32, 32) images 
    emnist_uppercase_train_3rd_May_2021.csv has 26 uppercase classes """

    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = HandwritingDataset(path_to_train_csv, transform=all_transforms)
    test_data = HandwritingDataset(path_to_test_csv, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def get_emnist_lowercase_dataloaders(batch_size=128, 
                                          path_to_train_csv='/Users/aashishkumar/Documents/notebooks/emnist_lowercase_train_13th_May.csv',
                                          path_to_test_csv='/Users/aashishkumar/Documents/notebooks/emnist_lowercase_test_13th_May.csv'):
    """ Handwriting Operators dataloader with (32, 32) images 
    emnist_uppercase_train_3rd_May_2021.csv has 26 uppercase classes """

    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = HandwritingDataset(path_to_train_csv, transform=all_transforms)
    test_data = HandwritingDataset(path_to_test_csv, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def get_emnist_uppercase_reduced_dataloaders(batch_size=128, 
                                          path_to_train_csv='/Users/aashishkumar/Documents/notebooks/emnist_uppercase_train_11th_May_2021_reduced.csv',
                                          path_to_test_csv='/Users/aashishkumar/Documents/notebooks/emnist_uppercase_test_11th_May_2021_reduced.csv'):
    """ Handwriting Operators dataloader with (32, 32) images 
    emnist_uppercase_train_11th_May_2021_reduced.csv has 10 uppercase classes """

    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = HandwritingDataset(path_to_train_csv, transform=all_transforms)
    test_data = HandwritingDataset(path_to_test_csv, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_mnist_dataloaders(batch_size=128, path_to_data='/Users/aashishkumar/Documents/pytorch_datasets'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128,
                                  path_to_data='/Users/aashishkumar/Documents/pytorch_datasets'):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,
                            path_to_data='../dsprites-data/dsprites_data.npz'):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                 shuffle=True)
    return dsprites_loader


def get_chairs_dataloader(batch_size=128,
                          path_to_data='../rendered_chairs_64'):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=True)
    return chairs_loader


def get_chairs_test_dataloader(batch_size=62,
                               path_to_data='../rendered_chairs_64_test'):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=False)
    return chairs_loader


def get_celeba_dataloader(batch_size=128, path_to_data='../celeba_64'):
    """CelebA dataloader with (64, 64) images."""
    celeba_data = CelebADataset(path_to_data,
                                transform=transforms.ToTensor())
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=True)
    return celeba_loader


class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data)['imgs'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0