from torchvision import utils, transforms
import os
import json
import copy
from PIL import Image
import os.path
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def update_alllocal_protos(ave, agg_protos, alllocal_protos):
    for label in agg_protos:
        if label in alllocal_protos:
            alllocal_protos[label] = agg_protos[label].data * ave + alllocal_protos[label].data* (1-ave)
        else:
            alllocal_protos[label] = agg_protos[label]

    return alllocal_protos

def allproto_aggregation(local_protos_list,clients):
    agg_protos_label = dict()
    for ind, idx in enumerate(clients):
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


def update_global_protos(global_protos_old,global_protos_new):
    global_protos = dict()
    a = global_protos_old.keys() - global_protos_new.keys()
    b = global_protos_new.keys() - global_protos_old.keys()
    c = global_protos_new.keys() & global_protos_old.keys()
    for label in a:
        global_protos[label] = global_protos_old[label]
    for label in b:
        global_protos[label] = global_protos_new[label]
    for label in c:
        #global_protos[label] = [sum(global_protos_new[label]+global_protos_old[label])/2.0]
        global_protos[label] = [i * 0.5 for i in global_protos_new[label]] + [i * 0.5 for i in global_protos_old[label]]

    return global_protos

def iter_proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0][0].data
            for i in proto_list:
                proto += i[0].data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0][0].data]

    return agg_protos_label