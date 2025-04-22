from ast import Return
from genericpath import exists
from http import client
from os import replace
from pickle import FALSE
from re import L
import numpy as np
from torch.utils.data.dataset import Dataset
import copy
import math
import functools
import numpy as np
import itertools
import random
import torch

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def show_clients_data_distribution(dataset, clients_indices: list, num_classes, num_users):
    dict_per_client = []
    dict_per_client_class=[]
    per_client_weight = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        class_data = []
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1
            class_data.append(label)
        dict_per_client.append(nums_data)
        dict_per_client_class.append(class_data)
        cl_ratio = []
        max_num= max(nums_data)
        for i in range (num_classes):
            cl_ratio.append(nums_data[i]/max_num)
        print(f'{client}: {nums_data}')
        cl_ratio = torch.tensor(cl_ratio)
        cl_ratio = cl_ratio.cuda()
        per_client_weight.append(cl_ratio)
        print(f'{client}: {cl_ratio}')
    return dict_per_client, dict_per_client_class,per_client_weight


def split_list_n_list(original_dict_per_client: list,dict_per_client_class,n):
    dict_per_clinet_split = []*n
    inx_per_clinet_split = []*n
    for i,origin_list in enumerate(original_dict_per_client):
        per_clinet_split=[]
        if len(origin_list) % n == 0:
            cnt = len(origin_list) // n
        else:
            cnt = len(origin_list) // n + 1
        for j in range(0, n):
            per_clinet_split.append(origin_list[j*cnt:(j+1)*cnt])
        dict_per_clinet_split.append(per_clinet_split)
    

    for i,idx_list in enumerate(dict_per_client_class):
        inx_clinet_split=[]
        uni_client_inx =[]
        if len(idx_list) % n == 0:
            nt = len(idx_list) // n
        else:
            nt = len(idx_list) // n + 1
        for j in range(0, n):
            inx_clinet_split.append(idx_list[j*nt:(j+1)*nt])
            a = list(set(inx_clinet_split[j]))
            uni_client_inx.append(a)
        inx_per_clinet_split.append(uni_client_inx)
    return dict_per_clinet_split,inx_per_clinet_split

def select_clients(dict_split_client,idx_per_client_split,clients,stage,seed):
    for i in range(clients):
        random.seed(seed)
        random.shuffle(idx_per_client_split[i])
    stage_users=[[]for r in range(clients)]
    for j in range(stage):
        dict_users = {r: np.array([], dtype='int64') for r in range(clients)}
        for k in range(clients):
            rand_set = np.array(idx_per_client_split[k][j])
            dict_users[k] = rand_set        
        stage_users[j]=dict_users #其中的每个列表表示一个stage，里面的是每个客户端选取的数据，总共有10个类别，最外层的循环表示是一个stage，里面包括10个客户端
    return stage_users



def partition_train_teach(list_label2indices: list, num_data_train: int, seed=None):
    random_state = np.random.RandomState(seed)
    list_label2indices_train = []
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_train.append(indices[:num_data_train // 10])
        list_label2indices_teach.append(indices[num_data_train // 10:])
    return list_label2indices_train, list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int, seed):
    random_state = np.random.RandomState(seed)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_class_num(class_list):
    index = []
    compose = []
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose



def clients_indices(list_label2indices: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed):
    indices2targets = []
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients)
    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
    list_client2indices = partition_balance(indices_dirichlet, num_clients)

    return list_client2indices


def partition_balance(idxs, num_split: int):

    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def build_non_iid_by_dirichlet(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)

    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        #n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch


def idx_clients(dataset, stage, clients, clients_idx, n, seed):
    random.seed(seed)
    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < n and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < n:
            idxs_dict[label].append(i)
            count += 1
    for label in range(n):
        random.shuffle(idxs_dict[label])
        
    list_label = [u for u in range(n)]
    label_count = [0]*n
    for t in range(n):
        for i in range(clients):
            for j in range(stage):
                for x in (clients_idx[i][j]):
                    if x == t:
                        label_count[t]+=1
    length_class = [0]*n
    for i in range(n):
        length_class[i]= len(idxs_dict[i])//label_count[i]
    
    matrix_stage = [[[]for n in range(stage)] for n in range(clients)]
    matrix = [[] for u in range(clients)]
    for i in range(clients):
        for j in range(stage):
            a = clients_idx[i][j]
            for idx, x in enumerate(a):
                label_count[x] = label_count[x]-1
                if label_count[x] == 0:
                    b = len(idxs_dict[x])
                else:
                    b = length_class[x]
                sele = idxs_dict[x]
                matrix_stage[i][j].extend(sele[:b])
                matrix[i].extend(sele[:b])
                idxs_dict[x] = idxs_dict[x][b:]
    
    for i in range(clients):
        random.seed(seed)
        random.shuffle(matrix[i])
        for j in range(stage):
            random.seed(seed)
            random.shuffle(matrix_stage[i][j])

    return matrix_stage, matrix


def idx_clients_dir(list_label2indices_train_new, dataset, stage, clients, clients_idx, n, seed):
    idxs_dict = {}
    for i in range(len(list_label2indices_train_new)):
        random.seed(seed)
        random.shuffle(list_label2indices_train_new)
        #label = torch.tensor(dataset.targets[i]).item()
        
    idxs_dict = list_label2indices_train_new
    label_count = [0]*n
    for t in range(n):
        for i in range(clients):
            for j in range(stage):
                for x in (clients_idx[i][j]):
                    if x == t:
                        label_count[t]+=1
    length_class = [0]*n
    for i in range(n):
        length_class[i]= len(idxs_dict[i])//label_count[i]
    
    matrix_stage = [[[]for n in range(stage)] for n in range(clients)]
    matrix = [[] for u in range(clients)]
    for i in range(clients):
        for j in range(stage):
            a = clients_idx[i][j]
            for idx, x in enumerate(a):
                label_count[x] = label_count[x]-1
                if label_count[x] == 0:
                    b = len(idxs_dict[x])
                else:
                    b = length_class[x]
                sele = idxs_dict[x]
                matrix_stage[i][j].extend(sele[:b])
                matrix[i].extend(sele[:b])
                idxs_dict[x] = idxs_dict[x][b:]

    return matrix_stage, matrix