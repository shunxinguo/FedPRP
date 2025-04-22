#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch

def langcon(train_num,list_label2indices_train_new,dataset, num_users, shard_per_user, num_classes,clients,seed,rand_set_all=[], testb=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(list_label2indices_train_new)):
        #label = torch.tensor(dataset.targets[i]).item()
        idxs_dict[i]=list_label2indices_train_new[i]

    '''
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1
    '''
    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( train_num/num_users )
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
        a = len(rand_set_all) % num_users
        if (a != 0):
            rand_set_all = rand_set_all[:len(rand_set_all)-a]   
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    dict_per_client = []
    dict_per_client_class=[]
    per_client_weight = []
    per_client_data = []
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
        nums_data = [0 for _ in range(num_classes)]
        class_data = []
        for idx in dict_users[i]:
            label = dataset[idx][1]
            nums_data[label] += 1
            class_data.append(label)
            dict_per_client.append(nums_data)
        dict_per_client_class.append(class_data)
        cl_ratio = []
        max_num= max(nums_data)
        for j in range (num_classes):
            cl_ratio.append(nums_data[j]/max_num)
        print(f'{i}: {nums_data}')
        cl_ratio = torch.tensor(cl_ratio)
        cl_ratio = cl_ratio.cuda()
        per_client_weight.append(cl_ratio)
        per_client_data.append(nums_data)
        print(f'{i}: {cl_ratio}')

    
    #构建stage数据，共10个客户端，根据总体的users
    sta_client = int(num_users/clients)#一个客户端有多少个stage
    stage_users = [[] for r in range(clients)]#给每个客户端输入stage的数据
    #按客户端和几个stage进行顺序10个一组，在每次训练数据时选择每个客户端的每个stage，结果得到的是客户端内包含的stage
    train_all_user=dict_users.copy()
    for i in range(clients):
        for j in range(sta_client):
            c = i*sta_client+j
            stage_users[i].append(train_all_user[c])

    return rand_set_all,stage_users,sta_client,dict_users,per_client_weight,per_client_data
   





