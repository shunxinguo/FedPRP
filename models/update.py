from grpc import protos
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import numpy as np
import copy
from models.losses import DROLoss
from sklearn.datasets import load_digits
        
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args,epsilons=None, class_weights=None,dataset=None, argu = None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=False, num_workers=8)
        self.feat_train_loader = DataLoader(DatasetSplit(argu, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=False, num_workers=8)
        if indd is not None:
            self.indd = indd
        else:
            self.indd=None    
        self.dataset=dataset
        self.device = args.device
        self.idxs=idxs
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.temperature = args.temperature
        self.base_temperature = args.base_temperature
        self.class_weights = class_weights
    def pairwise_cosine_sim(self, x, y):
            x = x / x.norm(dim=1, keepdim=True)
            y = y / y.norm(dim=1, keepdim=True)
            return torch.matmul(x, y.T)
    def pairwise_euaclidean_distance(self, x, y):
            return torch.cdist(x, y)       

    def validate(self, model):
        feat_dim = 512
        features = torch.empty((0, feat_dim)).cuda()
        targets = torch.empty(0, dtype=torch.long).cuda()
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.feat_train_loader):
                if self.args.gpu is not None:
                    input = input.cuda(self.args.gpu, non_blocking=True)
                target = target.cuda(self.args.gpu, non_blocking=True)
                output, feats = model(input)
                features = torch.cat((features, feats))
                targets = torch.cat((targets, target))
        return features, targets
    
    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=False)
        return trainloader

    
    def train(self, stage_iter, net, args, idx, w_glob_keys, lr, last, net_glob, global_protos, global_round,stage, localpros, base_features, base_targets, num_workers=8):

        feat_params = []
        feat_params_names = []
        cls_params = []
        cls_params_names = []
        learnable_epsilons = torch.nn.Parameter(torch.ones(args.num_classes))
        for name, params in net.named_parameters():
            if "fc" in name:
                cls_params_names += [name]
                cls_params += [params]
            else:
                feat_params_names += [name]
                feat_params += [params]

        if self.args.optimizer == 'sgd':  
            optimizer = torch.optim.SGD([     
            {'params': feat_params+[learnable_epsilons], 'weight_decay':0.0001},
            {'params': cls_params, 'weight_decay':0}
            ], lr=lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam([     
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
            ], lr=lr, weight_decay=1e-4)
        local_eps = self.args.local_ep
        if last:
            local_eps =  max(10,local_eps-self.args.local_rep_ep)
        
        head_eps = local_eps-self.args.local_rep_ep
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}
        num_updates = 0
        for iter in range(local_eps):
            done = False
    
            if (iter < head_eps) or last:
                for name, param in net.named_parameters():
                    if "fc" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                        learnable_epsilons.requires_grad = False
            elif iter == head_eps and not last:
                for name, param in net.named_parameters():
                    if "fc" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), label_g.to(self.args.device)
                net.zero_grad() 

                log_probs, protos = net(images)
                protos = protos.to(self.args.device)
                loss1 = self.loss_func(log_probs, labels)
                loss3 = 0 * loss1
                loss2 = 0 * loss1
                if args.lam_1!=0:
                    loss_mse = nn.MSELoss()
                    if len(global_protos) == 0:
                        loss2 = 0*loss1
                    else:
                        proto_new = copy.deepcopy(protos.data)
                        i = 0
                        for label in labels:
                            if label.item() in global_protos.keys():
                                proto_new[i, :] = global_protos[label.item()][0].data
                            i += 1
                        loss2 = loss_mse(proto_new, protos)
                if args.lam_2!=0:
                    loss3 = DROLoss.count(self, protos, labels, base_features, base_targets, learnable_epsilons)   
                loss = loss1 + args.lam_1 * loss2 + args.lam_2 * loss3
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['3'].append(loss3.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['3'].append(sum(batch_loss['3']) / len(batch_loss['3']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['3'] = sum(epoch_loss['3']) / len(epoch_loss['3'])
        return net.state_dict(), epoch_loss, self.indd, agg_protos_label

