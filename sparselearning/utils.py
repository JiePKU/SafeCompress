import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import random
from typing import Any, Callable, List, Optional, Union, Tuple

class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""
    def __init__(self,parent_dataset,split_start=-1,split_end= -1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and     split_start < split_end , "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start


    def __getitem__(self,index):
        assert index < len(self),"index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


def get_cifar100_dataloaders(args, validation_split=0.0, max_threads=10):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomResizedCrop(32,scale=(0.2,1)),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2)],p=0.5),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    trainset = torchvision.datasets.CIFAR100(root='/home/pc/zhujie/data/cifar100', train=True, download=True,
                                             transform=transform_train)
    trainset_length = len(trainset)
    random.seed(42)
    trainset_index = random.sample(range(trainset_length),int(trainset_length/2))
    # unseen part to test inference model
    train_inf_test_index = list(set(range(trainset_length)).difference(set(trainset_index)))
    knowntrainset = torch.utils.data.Subset(trainset,trainset_index)
    infset = torch.utils.data.Subset(trainset,train_inf_test_index)

    infset_loader = torch.utils.data.DataLoader(infset,batch_size=args.batch_size, shuffle=True, num_workers=2)
    knownset_loader = torch.utils.data.DataLoader(knowntrainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    random.seed(18)
    testset = torchvision.datasets.CIFAR100(root='/home/pc/zhujie/data/cifar100', train=False, download=True, transform=transform_test)
    testset_length = len(testset)
    testset_index = random.sample(range(testset_length), int(testset_length / 2))
    referenceset = torch.utils.data.Subset(testset,testset_index)
    test_inf_test_index = list(set(range(testset_length)).difference(set(testset_index)))
    test_infset = torch.utils.data.Subset(testset, test_inf_test_index)

    reference_loader = torch.utils.data.DataLoader(referenceset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_infset_loader = torch.utils.data.DataLoader(test_infset, batch_size = args.batch_size,shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, test_loader, knownset_loader, infset_loader, reference_loader, test_infset_loader


