from random import random
import numpy as np
from Dataset.dataset import label_indices2indices
import copy
import random


def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes
    img_num_per_cls = []
    many,mid,few =[],[],[]
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
            if _classes_idx in range(0, int(num_classes * 0.2)):
                many.append(_classes_idx)
            elif _classes_idx in range(int(num_classes*0.2), int(num_classes*0.5)):  
                mid.append(_classes_idx)
            else:
                few.append(_classes_idx)  
            # if num >= img_max * 0.5:
            #     many.append(_classes_idx)
            # elif img_max * 0.5 > num > img_max * imb_factor:
            #     mid.append(_classes_idx)
            # else:
            #    few.append(_classes_idx) 
    return img_num_per_cls,many,mid,few


def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type,seed):
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list,many,mid,few = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')
    print(img_num_list)
    # print('many class')
    # print(many)
    # print('mid class')
    # print(mid)
    # print('few class')
    # print(few)

    list_clients_indices = []
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.seed(seed)
        np.random.shuffle(indices)#把类别的所有数据打乱顺序进行选择
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))
    cl_ratio = []
    max_num= max(img_num_list)
    for i in range (num_classes):
        cl_ratio.append(img_num_list[i]/max_num)
    return cl_ratio, list_clients_indices,len(num_list_clients_indices), many, mid, few





