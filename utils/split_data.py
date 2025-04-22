
from torchvision import models, utils, datasets, transforms
from models.resnet import resnet18
from utils.sampling import noniid, sampling_train, sampling_test
from utils.longsampling import langcon
from Dataset.long_tailed_cifar10 import train_long_tail
import os
import json
import copy
from PIL import Image
import os.path
import torch
import warnings
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from utils.autoaugment import CIFAR10Policy, Cutout
from Dataset.dataset import classify_label


trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),  # fill parameter needs torchvision installed from source
         transforms.RandomHorizontalFlip(),
         CIFAR10Policy(),
         transforms.ToTensor(),
         Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
         ])

def dir_get_data(args):
    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        augmented_train_dataset = datasets.CIFAR10('data/cifar10',train=True, download=True,transform=transform_train)
        
    stage = args.num_users//args.clients
    train_sta_client, train_all_user, rand_set,all_clients_idx, stage_idx , many, mid, few,cl_ratio = sampling_train(dataset_train, args, stage)
    
    test_sta_client, dict_users_test = sampling_test(dataset_test,args,stage,stage_idx)

    return dataset_train, dataset_test, train_sta_client,test_sta_client, rand_set,stage, \
           dict_users_test, train_all_user, stage_idx, all_clients_idx, many,mid,few, augmented_train_dataset,cl_ratio

def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data1/data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data1/data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        augmented_train_dataset = datasets.CIFAR10('data1/data/cifar10',train=True, download=True,transform=transform_train)
    
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        augmented_train_dataset = datasets.CIFAR100('data/cifar100',train=True, download=True,transform=transform_train)

    elif args.dataset == 'femnist':
        apply_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

        dataset_train = FEMNIST(args,'./data/femnist/' , train=True, download=True,
                                        transform=apply_transform)
        dataset_test = FEMNIST(args, './data/femnist/', train=False, download=True,
                                       transform=apply_transform)
    list_label2indices = classify_label(dataset_train, args.num_classes)
    _, list_label2indices_train_new, train_num , many,mid,few = train_long_tail(copy.deepcopy(list_label2indices),
                                                                     args.num_classes, args.imb_factor, args.imb_type,args.seed)

    tarin_rand_set, train_sta_client, stage, train_all_user,per_client_weight,_ = langcon(train_num, list_label2indices_train_new,
                                                                          dataset_train, args.num_users,
                                                                          args.shard_per_user, args.num_classes,
                                                                          args.clients, args.seed)                                                                 
    test_rand_set, test_sta_client, teststage, dict_users_test, test_all_user, test_stage_ind, test_stage_idx = noniid(
            dataset_test, args.num_users,
            args.shard_per_user,
            args.num_classes,
            args.clients, args.seed,
            rand_set_all=tarin_rand_set)
    return dataset_train, dataset_test, train_sta_client, test_sta_client, tarin_rand_set, test_rand_set, stage, \
           dict_users_test, train_all_user, test_all_user, test_stage_ind, test_stage_idx,many,mid,few,augmented_train_dataset,per_client_weight

def get_model(args):
    if args.model == 'resnet18' and 'cifar' in args.dataset:
        args.stride = [2, 2]
        net_glob = resnet18(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    return net_glob
