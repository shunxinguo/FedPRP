import copy
from itertools import count
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from torch import stack, max, eq, no_grad

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label

def test_img_local(net_g, dataset, args,idx=None,indd=None, user_idx=-1, idxs=None):
    net_g.eval()
    test_loss = 0
    correct = 0

    leaf=False
    
    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=args.local_bs, shuffle=False)
    else:
        data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs,shuffle=False, num_workers=4,pin_memory=True,prefetch_factor=2, persistent_workers=True)

    count = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
        log_probs,_ = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        count += args.local_bs
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return  accuracy, test_loss

def test_img_local_all(num_idxxs,net, args, dataset_test, dict_users_test, w_locals=None, w_glob_keys=None, indd=None,
                       dataset_train=None, return_all=False):
    tot = 0
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for ind in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[ind].keys():
                if w_glob_keys is not None and k not in w_glob_keys:
                    w_local[k] = w_locals[ind][k] 
                elif w_glob_keys is None:
                    w_local[k] = w_locals[ind][k]  
            net_local.load_state_dict(w_local)
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, user_idx = ind, idxs=dict_users_test[ind])
        tot += len(dict_users_test[ind])
        acc_test_local[ind] = a * len(dict_users_test[ind])
        loss_test_local[ind] = b * len(dict_users_test[ind])
        del net_local

    if return_all:
        return acc_test_local, loss_test_local
    return sum(acc_test_local) / tot, sum(loss_test_local) / tot


def test_img_local_all_sta(idxs_users,net, args, dataset_test, dict_users_test, w_locals, w_glob_keys=None, indd=None,
                       dataset_train=None, return_all=False):
    num_idxxs = len(idxs_users)
    tot = 0
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for ind, id in enumerate(idxs_users):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[id].keys():
                if w_glob_keys is not None and k not in w_glob_keys:
                    w_local[k] = w_locals[id][k] 
                elif w_glob_keys is None:
                    w_local[k] = w_locals[id][k] 
            net_local.load_state_dict(w_local)
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, user_idx = id, idxs=dict_users_test[ind])
        tot += len(dict_users_test[ind])
        acc_test_local[ind] = a * len(dict_users_test[ind])
        loss_test_local[ind] = b * len(dict_users_test[ind])
        del net_local

    if return_all:
        return acc_test_local, loss_test_local
    return sum(acc_test_local) / tot, sum(loss_test_local) / tot


def test_proto(idxs_users, args, net, test_dataset, classes_list,w_glob_keys, user_groups_gt, global_protos=[],local_protos=[], local_model_list=None):
    """ Returns the test accuracy and loss.
    """
    loss_mse = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    device = args.device
    criterion = nn.CrossEntropyLoss().to(device)

    acc_list_g = []
    acc_list_lg=[]
    acc_list_l = []
    loss_l = []
    loss_list_g = []
    loss_list_l=[]
    for ind, idx in enumerate(idxs_users):
        net = copy.deepcopy(net)
        if local_model_list is not None:
            w_local = net.state_dict()
            for k in local_model_list[idx].keys():
                if w_glob_keys is not None and k not in w_glob_keys:
                    w_local[k] = local_model_list[idx][k]  
                elif w_glob_keys is None:
                    w_local[k] = local_model_list[idx][k]  
            net.load_state_dict(w_local)
        net.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[ind]), batch_size=args.local_bs, shuffle=False,num_workers=4,pin_memory=True,prefetch_factor=2, persistent_workers=True)

        # test (local model)
        net.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            net.zero_grad()
            outputs, _ = net(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = 100.00 * float(correct) / total
        lo = 100.00 * float(loss) / total
        #print('| User: {} | Global Test Acc no protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)
        loss_l.append(lo)

        if local_protos[idx] != []:
            total, correct, cos_correct = 0.0, 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                net.zero_grad()
                outputs, protos = net(images)
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(
                    device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in local_protos[idx].keys() and j in classes_list[ind]:
                            d = loss_mse(protos[i, :], local_protos[idx][j])
                            dist[i, j] = d
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in local_protos[idx].keys():
                        proto_new[i, :] = local_protos[idx][label.item()].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.cpu().detach().numpy()

            acc = 100.00 * float(correct) / total
            lop = 100.00 * float(loss) / total
            #print('| User: {} | Local Test Acc with local protos: {:.5f}'.format(idx, acc))
            acc_list_lg.append(acc)
            loss_list_l.append(lop)

        # test (use global proto)
        if global_protos != []:
            total, correct = 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                net.zero_grad()
                outputs, protos = net(images)
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(
                    device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[ind]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss1 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss1 = loss1.cpu().detach().numpy()
                else:
                    loss1 = loss1.cpu().detach().numpy()

            acc = 100.00 * float(correct) / total
            #print('| User: {} | Global Test Acc with global protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            loss_list_g.append(loss1)

    return acc_list_l, acc_list_lg, acc_list_g, loss_l, loss_list_l,loss_list_g

def global_eval(args,net, data_test, batch_size_test):
    test_loss = 0
    count = 0
    net.eval()
    net.to(args.device)
    with no_grad():
        test_loader = DataLoader(data_test, batch_size_test)
        num_corrects = 0
        for data_batch in test_loader:
            images, labels = data_batch
            images, labels = images.to(args.device), labels.to(args.device)
            outputs,_ = net(images)
                # sum up batch loss
            test_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicts = max(outputs, -1)
            num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            count += batch_size_test
        accuracy = 100.00 * float(num_corrects) / len(data_test)
        test_loss /= count
    return accuracy, test_loss

def tail_test_proto(idxs_users, many, mid, few, args, net, test_dataset, classes_list, w_glob_keys, user_groups_gt, global_protos=[],local_protos=[], local_model_list=None):
    """ Returns the test accuracy and loss.
    """
    loss_mse = nn.MSELoss()
    device = args.device
    criterion = nn.CrossEntropyLoss().to(device)

    many_lis, mid_lis, few_lis = [], [], []
    many_lis_lg, mid_lis_lg, few_lis_lg = [], [], []
    many_lis_g, mid_lis_g, few_lis_g = [], [], []
    loss_l = []
    loss_list_g = []
    loss_list_l=[]
    for ind, idx in enumerate(idxs_users):
        net = copy.deepcopy(net)
        if local_model_list is not None:
            w_local = net.state_dict()
            for k in local_model_list[idx].keys():
                if w_glob_keys is not None and k not in w_glob_keys:
                    w_local[k] = local_model_list[idx][k] 
                elif w_glob_keys is None:
                    w_local[k] = local_model_list[idx][k] 
            net.load_state_dict(w_local)
        net.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[ind]), batch_size=args.local_bs, shuffle=False)
        many_num, mid_num, few_num = 0,0,0
        many_cor, mid_cor, few_cor = 0,0,0
        # test (local model)
        net.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            net.zero_grad()
            outputs, protos = net(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            predicts = pred_labels.tolist()
            tail_labels = labels.tolist()
            for u, idxs in enumerate(tail_labels):
                    if idxs in many:
                        many_num += 1
                        if predicts[u] == idxs:
                            many_cor += 1

                    elif idxs in mid:
                        mid_num += 1
                        if predicts[u] == idxs:
                            mid_cor += 1
                    
                    elif idxs in few:
                        few_num += 1
                        if predicts[u] == idxs:
                            few_cor += 1

        # acc = 100.00 * float(tail_correct) / tail_all
        lo = 100.00 * float(loss) / total
        #print('| User: {} | Global Test Acc no protos: {:.3f}'.format(idx, acc))
        if many_num > 0:
            many_lis.append(100.00 * float(many_cor) / many_num)
        else:
            many_lis.append(0.0) 

        if mid_num > 0:
            mid_lis.append(100.00 * float(mid_cor) / mid_num)
        else:
            mid_lis.append(0.0) 
        if few_num > 0:
            few_lis.append(100.00 * float(few_cor) / few_num)
        else:
            few_lis.append(0.0)     
        loss_l.append(lo)


        if local_protos[idx] != []:
            many_num, mid_num, few_num = 0.0001,0.0001,0.0001
            many_cor, mid_cor, few_cor = 0,0,0
            total, correct = 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                net.zero_grad()
                outputs, protos = net(images)
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(
                    device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in local_protos[idx].keys() and j in classes_list[ind]:
                            d = loss_mse(protos[i, :], local_protos[idx][j])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                
                tail_predicts = pred_labels.tolist()
                tail_labels = labels.tolist()
                for v, idxt in enumerate(tail_labels):
                        if idxt in many:
                            many_num += 1
                            if tail_predicts[v] == idxt:
                                many_cor+=1
                        elif idxt in mid:
                            mid_num += 1
                            if tail_predicts[v] == idxt:
                                mid_cor+=1
                        elif idxt in few:
                            few_num += 1
                            if tail_predicts[v] == idxt:
                                few_cor+=1
                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in local_protos[idx].keys():
                        proto_new[i, :] = local_protos[idx][label.item()].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.cpu().detach().numpy()

            if many_num > 0:
                many_lis_lg.append(100.00 * float(many_cor) / many_num)
            else:
                many_lis_lg.append(0.0) 
            if mid_num > 0:
                mid_lis_lg.append(100.00 * float(mid_cor) / mid_num)
            else:
                mid_lis_lg.append(0.0) 
            if few_num > 0:
                few_lis_lg.append(100.00 * float(few_cor) / few_num)
            else:
                few_lis_lg.append(0.0) 
            loss_list_l.append(loss2)

        # test (use global proto)
        if global_protos != []:
            total, correct = 0.0, 0.0
            many_num, mid_num, few_num = 0,0,0
            many_cor, mid_cor, few_cor = 0,0,0

            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                net.zero_grad()
                outputs, protos = net(images)
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(
                    device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[ind]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                
                predi = pred_labels.tolist()
                tail_labels = labels.tolist()
                for r, idxr in enumerate(tail_labels):
                        if idxr in many:
                            many_num += 1
                            if predi[r] == idxr:
                                many_cor += 1
                        elif idxr in mid:
                            mid_num += 1
                            if predi[r] == idxr:
                                mid_cor += 1  
                        elif idxr in few:
                            few_num += 1
                            if predi[r] == idxr:
                                few_cor += 1   
                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss1 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss1 = loss1.cpu().detach().numpy()
                else:
                    loss1 = loss1.cpu().detach().numpy()

            if many_num > 0:
                many_lis_g.append(100.00 * float(many_cor) / many_num)
            else:
                many_lis_g.append(0.0) 
            if mid_num > 0:
                mid_lis_g.append(100.00 * float(mid_cor) / mid_num)
            else:
                mid_lis_g.append(0.0) 
            if few_num > 0:
                few_lis_g.append(100.00 * float(few_cor) / few_num)
            else:
                few_lis_g.append(0.0)     
            loss_list_g.append(loss1)
    return many_lis, mid_lis, few_lis, many_lis_lg, mid_lis_lg, few_lis_lg, many_lis_g, mid_lis_g, few_lis_g