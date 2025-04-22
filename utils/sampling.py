#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch
import copy

from Dataset.dataset import classify_label, show_clients_data_distribution, split_list_n_list
from Dataset.dataset import clients_indices, idx_clients, idx_clients_dir
from Dataset.long_tailed_cifar10 import train_long_tail

def noniid(dataset, num_users, shard_per_user, num_classes,clients,seed,rand_set_all=[],testb=True):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.seed(seed)
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    sta_client = int(num_users / clients)  
    stage_users = [[] for r in range(clients)]  
    stage_ind = [[] for r in range(clients)]
    test_all_user=dict_users.copy()
    for i in range(clients):
        for j in range(sta_client):
            c = i*sta_client+j
            stage_users[i].append(test_all_user[c])
            stage_ind[i].append(rand_set_all[c])


    dict_users_test=[]
    for i in range(clients):
        c=[]
        for r in range(sta_client):
            c.extend(stage_users[i][r])
        dict_users_test.append(c)

    stage_idx = []
    for i in range(clients):
        a = i*sta_client
        b = (i+1)*sta_client
        c=[]
        for j in range(a,b):
            c.extend(rand_set_all[j])
        stage_idx.append(c)

    return rand_set_all, stage_users, sta_client,dict_users_test,dict_users,stage_ind,stage_idx
def sampling_train(dataset_train, args,staclient):
    list_label2indices = classify_label(dataset_train, args.num_classes) 
    cl_ratio, list_label2indices_train_new, train_num, many,mid,few = train_long_tail(copy.deepcopy(list_label2indices),args.num_classes, args.imb_factor, args.imb_type, args.seed)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,args.clients, args.non_iid_alpha, args.seed)
    dict_per_client, dict_per_client_class,per_client_weight = show_clients_data_distribution(dataset_train, list_client2indices, args.num_classes,args.clients)

    dict_split_client, stage_idx = split_list_n_list(copy.deepcopy(list_client2indices),dict_per_client_class,staclient)
    
    all_clients_idx = []
    for i in range(args.clients):
        a = list(set(dict_per_client_class[i]))
        all_clients_idx.append(a)

    return dict_split_client, list_client2indices, dict_per_client, all_clients_idx, stage_idx, many,mid,few,per_client_weight



def sampling_test(dataset_test, args, staclient, train_all_clients_idx):
    list_label2indices = classify_label(dataset_test, args.num_classes)  
    _, list_label2indices_train_new, train_num,_,_,_ = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes, 1, args.imb_type, args.seed)
    test_stage, test= idx_clients_dir(list_label2indices_train_new, dataset_test, staclient, args.clients, train_all_clients_idx, args.num_classes, args.seed)

    return test_stage, test