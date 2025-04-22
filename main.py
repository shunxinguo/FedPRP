from ast import arg
import copy
import itertools
import random
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.split_data import get_data,dir_get_data, get_model
from utils.update_proto import agg_func, proto_aggregation, update_global_protos, iter_proto_aggregation, update_alllocal_protos, allproto_aggregation
from utils.log import Logger
from models.update import LocalUpdate
from models.test import test_img_local_all, test_img_local_all_sta, test_proto, global_eval, tail_test_proto
from utils.data_type import onli

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    lens = np.ones(args.num_users)
    logger = Logger('./save/' +'_'+args.noniid+'_'+ args.alg + '_'+ args.loss + '_'   + args.model + '_' + args.dataset+ '_'  + \
                    str(args.num_users) + '_' + str(args.clients)  + '_' + str(
                    args.local_rep_ep)+ '_' + str(args.local_ep)+ '_' + str(args.imb_factor)+ '_' 
                    + str(args.lr)+ '_' + str(args.local_bs)+'_'+str(args.optimizer)+ '_' + str(args.lam_1)
                    +'_' + str(args.lam_2)+ '_'+ str(args.num_online_clients)+ '_'+str(args.epochs)+'_'+ 
                    str(args.shard_per_user)+'_'+ str(args.ave)+'_'+args.loss+'_'+ str(args.non_iid_alpha)+'_'+str(args.temperature)+'.txt')
    if args.noniid == 'dir':    
        dataset_train, dataset_test, sta_client, test_sta_client, rand_set, \
            stage, dict_users_test, train_all_user, test_stage_ind, test_stage_idx,\
            many, mid, few,train_dataset,per_client_weight = dir_get_data(args)
    elif args.noniid == 'shard':
        dataset_train, dataset_test, sta_client, test_sta_client, tarin_rand_set, test_rand_set,\
            stage, dict_users_test, train_all_user, test_all_user,test_stage_ind,test_stage_idx, many,\
            mid, few,train_dataset,per_client_weight = get_data(args)

    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    if 'cifar' in args.dataset and 'cnn' == args.model:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
            w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'cifar' in args.dataset and 'resnet18' == args.model:
            w_glob_keys = net_keys.copy()
            w_glob_keys.remove('fc.weight')
            w_glob_keys.remove('fc.bias')
    print(total_num_layers)
    print('**************')
    print(w_glob_keys)
    print('------------')
    print(net_keys)

    num_param_glob = 0
    num_param_local = 0
    for key in net_glob.state_dict().keys():
        num_param_local += net_glob.state_dict()[key].numel()
        
        if key in w_glob_keys:
            num_param_glob += net_glob.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    
    print("model, non-iid type, loss : {}, {}, {} ".format(args.model, args.noniid, args.loss))
    print("learning rate, batch size,non_iid_alpha: {}, {}, {}".format(args.lr, args.local_bs, args.non_iid_alpha))
    print("local_rep_ep, local_ep, IF: {}, {}, {}".format(args.local_rep_ep,args.local_ep,args.imb_factor))
    print("epochs, stage, clients, num_users: {}, {}, {}, {} ".format(args.epochs, stage, args.clients, args.num_users))
    print("lam_1, lam_2, ave : {}, {}, {} ".format(args.lam_1, args.lam_2, args.ave))
    
    random.seed(args.seed)
    listone=[random.randint(0,100) for i in range(args.epochs+1)]
    print(listone)
    global_protos = []
    glaccs, glaccs_lopro,glaccs_nopro, glaccs_pro= [],[],[],[]
    glaccs_lopro_t,glaccs_nopro_t, glaccs_pro_t = [],[],[]
    glaccs_loprostd,glaccs_noprostd, glaccs_prostd = [],[],[]
    g_accs, g_accs_pro, g_accs_lopro, g_accs_nopro = [],[],[],[]

    g_accs_pro_f, g_accs_lopro_f, g_accs_nopro_f = [],[],[]
    g_accs_pro_ma, g_accs_lopro_ma, g_accs_nopro_ma = [],[],[]
    g_accs_pro_mid, g_accs_lopro_mid, g_accs_nopro_mid = [],[],[]
    g_accsstd, g_accs_prostd, g_accs_loprostd, g_accs_noprostd = [],[],[],[]
    alllocal_protos=[{}for i in range(args.clients)]
    re_trained_acc = []
    re_trained_loss =[]

    all_locals = {}
    for user in range(args.clients):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            if key in w_glob_keys:
                w_local_dict[key] = net_glob.state_dict()[key]
            else:
                w_local_dict[key] = net_glob.state_dict()[key]
        all_locals[user] = w_local_dict
 
    gl_accs10, gl_accs10_pro, gl_accs10_lopro, gl_accs10_nopro = 0,0,0,0
    gl_accs10_t, gl_accs10_pro_t, gl_accs10_lopro_t, gl_accs10_nopro_t = 0,0,0,0
    gl_accs10std, gl_accs10_prostd, gl_accs10_loprostd, gl_accs10_noprostd = 0,0,0,0
    g_accs10, g_accs10_pro, g_accs10_lopro, g_accs10_nopro = 0,0,0,0

    g_accs10_pro_ma, g_accs10_lopro_ma, g_accs10_nopro_ma = 0,0,0
    g_accs10_pro_mi, g_accs10_lopro_mi, g_accs10_nopro_mi = 0,0,0
    g_accs10_pro_t, g_accs10_lopro_t, g_accs10_nopro_t = 0,0,0
    g_accs10std, g_accs10_prostd, g_accs10_loprostd, g_accs10_noprostd = 0,0,0,0
    stage_acc_t, stage_noacc_t, stage_loacc_t, stage_goacc_t = [[] for r in range(stage)], [[] for r in range(stage)], [[] for r in range(stage)], [[] for r in range(stage)]
    stage_acc, stage_noacc,stage_loacc,stage_goacc  = [[] for r in range(stage)], [[] for r in range(stage)], [[] for r in range(stage)], [[] for r in range(stage)]
    stage_accstd, stage_noaccstd,stage_loaccstd,stage_goaccstd  = [[] for r in range(stage)], [[] for r in range(stage)], [[] for r in range(stage)], [[] for r in range(stage)]
    for sta in range(args.epochs):
        all_glob = {}
        w_locals = {}
        for user in range(args.clients):
            w_local_dict = {}
            for key in net_glob.state_dict().keys():
                if key in w_glob_keys:
                    w_local_dict[key] = net_glob.state_dict()[key]
                else:
                    w_local_dict[key] = net_glob.state_dict()[key]
            w_locals[user] = w_local_dict
        
        # training
        indd = None 
        accs_nopro_t, accs_pro_t,accs_lopro_t = [],[],[]
        loss_train, accs_train_sta, accs_test_sta, accs_nopro, accs_pro,accs_lopro = [],[],[],[],[],[]
        accs_noprostd, accs_prostd,accs_loprostd = [],[],[]
        accs_test_sta_l,accs_nopro_l, accs_pro_l,accs_lopro_l = [],[],[],[]
        
        dataset_test_user = []
        dataset_test_ind = []
        dataset_train_niid = []
        dataset_train_arg = []
        glob_local_protos= {}
        total_len = 0
        m = args.num_online_clients
        np.random.seed(listone[sta])
        idxs_users = np.random.choice(range(args.clients), m, replace=False)
        print("idxs_users: {}".format(idxs_users))
        for iter in range(stage):
            w_glob = {}
            dataset_train_niid, dataset_test_user, dataset_test_ind = onli(dataset_test_user,dataset_test_ind,dataset_train_niid,sta_client,test_sta_client,test_stage_ind,idxs_users,iter,args.seed)
            loss_total, loss_ce, loss_pro, loss_kl = [], [], [], []

            w_keys_epoch = w_glob_keys
            localpros = []
            total_len = 0
            p_idx = 0

            for ind, idx in enumerate(idxs_users):
                if len(dataset_train_niid[ind]) != 0:
                    local = LocalUpdate(args=args,class_weights =per_client_weight[ind],  dataset=dataset_train, argu = train_dataset, idxs=dataset_train_niid[ind])
                net_local = copy.deepcopy(net_glob)
                w_local = net_local.state_dict()
            
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
                net_local.load_state_dict(w_local)
                last = iter == args.epochs
                if len(dataset_train_niid[ind]) != 0:
                        features, targets = local.validate(model=net_local.to(args.device))
                        w_local, loss, indd, protos = local.train(iter, net=net_local.to(
                            args.device), args=args, idx=idx, w_glob_keys=w_glob_keys, lr=args.lr, last=False,
                            net_glob=net_glob, global_protos=global_protos, global_round=round, stage=stage,
                             localpros = alllocal_protos[p_idx],base_features = features, base_targets =targets)
                loss_total.append(copy.deepcopy(loss['total']))
                loss_ce.append(copy.deepcopy(loss['1']))
                loss_pro.append(copy.deepcopy(loss['2']))
                loss_kl.append(copy.deepcopy(loss['3']))
                
                agg_protos = agg_func(protos)
                alllocal_protos[idx] = update_alllocal_protos(args.ave, agg_protos, alllocal_protos[idx])
                p_idx = idx
                total_len += lens[idx]

                if len(w_glob) == 0:
                    w_glob = copy.deepcopy(w_local)
                    for k, key in enumerate(net_glob.state_dict().keys()):
                        w_glob[key] = w_glob[key] * lens[idx]
                        w_locals[idx][key] = w_local[key]
                else:
                    for k, key in enumerate(net_glob.state_dict().keys()):
                        if key in w_glob_keys:
                            w_glob[key] += w_local[key] * lens[idx]  
                        else:
                            w_glob[key] += w_local[key] * lens[idx]
                        w_locals[idx][key] = w_local[key]
            loss_total_avg = sum(loss_total) / len(loss_total)
            loss_ce_avg = sum(loss_ce) / len(loss_ce)
            loss_pro_avg = sum(loss_pro) / len(loss_pro)
            loss_kl_avg = sum(loss_kl) / len(loss_kl)
            loss_train.append(loss_total_avg)

            for user in range(args.clients):
                for key in w_locals[user].keys():
                    if len(all_glob) == 0:
                        all_locals[user][key] = w_locals[user][key]
                    else:
                        all_locals[user][key] += w_locals[user][key]
           
            for k in net_glob.state_dict().keys():
                w_glob[k] = torch.div(w_glob[k], total_len)
            w_local = net_glob.state_dict()
            for k in w_glob.keys():
                w_local[k] = w_glob[k]
            if args.epochs != iter:
                net_glob.load_state_dict(w_glob)
            
            if len(all_glob) == 0:
                all_glob = copy.deepcopy(w_glob)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    all_glob[key] = all_glob[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        all_glob[key] += w_glob[key]
                    else:
                        all_glob[key] += w_glob[key]
            localpros = allproto_aggregation(alllocal_protos,idxs_users)
            global_protos = allproto_aggregation(alllocal_protos,idxs_users)

            ### Local testing of the client

            sta_acc_list_l,sta_acc_list_lg, sta_acc_list_g,sta_loss_n, sta_loss_list_l, sta_loss_list_g = test_proto(idxs_users,args, net_glob, dataset_test,
                                                                                dataset_test_ind, w_glob_keys, dataset_test_user, global_protos,alllocal_protos, w_locals)
            many_lis, mid_lis, sta_acc_tail_l, many_lis_lg, mid_lis_lg, sta_acc_tail_lg, many_lis_g, mid_lis_g, sta_acc_tail_g = tail_test_proto(idxs_users, many, mid, few, args, net_glob, dataset_test,
                                                                                dataset_test_ind, w_glob_keys, dataset_test_user, global_protos,alllocal_protos, w_locals)
            
            print('Sta {:3d},online-For all users (global protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of test loss is {:.3f}, mean of test tail acc is {:.2f}'.format(iter,
                    np.mean(sta_acc_list_g), np.std(sta_acc_list_g), np.mean(sta_loss_list_g), np.mean(sta_acc_tail_g)))
            print('Sta {:3d},online-For all users (local protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of cos_acc is {:.3f}, mean of test tail acc is {:.2f}'.format(iter,
                    np.mean(sta_acc_list_lg), np.std(sta_acc_list_lg), np.mean(sta_loss_list_l), np.mean(sta_acc_tail_lg)))
            print('Sta {:3d},online-For all users (no protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of test loss is {:.3f}, mean of test tail acc is {:.2f}'.format(iter,
                    np.mean(sta_acc_list_l), np.std(sta_acc_list_l), np.mean(sta_loss_n), np.mean(sta_acc_tail_l)))
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            
            accs_pro.append(np.mean(sta_acc_list_g))
            accs_lopro.append(np.mean(sta_acc_list_lg))
            accs_nopro.append(np.mean(sta_acc_list_l))

            accs_pro_t.append(np.mean(sta_acc_tail_g))
            accs_lopro_t.append(np.mean(sta_acc_tail_lg))
            accs_nopro_t.append(np.mean(sta_acc_tail_l))

            accs_prostd.append(np.std(sta_acc_list_g))
            accs_loprostd.append(np.std(sta_acc_list_lg))
            accs_noprostd.append(np.std(sta_acc_list_l))
            print(accs_pro,accs_lopro,accs_nopro)
            print(accs_pro_t,accs_lopro_t,accs_nopro_t)
                
            stage_goacc[iter].append(accs_pro[iter])
            stage_loacc[iter].append(accs_lopro[iter])
            stage_noacc[iter].append(accs_nopro[iter])

            stage_goacc_t[iter].append(accs_pro_t[iter])
            stage_loacc_t[iter].append(accs_lopro_t[iter])
            stage_noacc_t[iter].append(accs_nopro_t[iter])

            stage_goaccstd[iter].append(accs_prostd[iter])
            stage_loaccstd[iter].append(accs_loprostd[iter])
            stage_noaccstd[iter].append(accs_noprostd[iter])

        
        a_net = copy.deepcopy(net_glob)
        for k in net_glob.state_dict().keys():
            all_glob[k] = torch.div(all_glob[k], stage)
        a_net.load_state_dict(all_glob)

        for user in range(args.clients):
            for k, key in enumerate(net_glob.state_dict().keys()):
                all_locals[user][key] = torch.div(all_locals[user][key], stage)
        
        ### Test all clients, using the client's own local model parameters
        num = [i for i in range(args.clients)]
        acc_list_l, acc_list_lg, acc_list_g,loss_list_n, loss_list_l,loss_list_g = test_proto(num, args, a_net, dataset_test,
                                                                        test_stage_idx, w_glob_keys, dict_users_test, global_protos,alllocal_protos, all_locals)
        many_lis, mid_lis, acc_tail_l, many_lis_lg, mid_lis_lg, acc_tail_lg, many_lis_g, mid_lis_g, acc_tail_g = tail_test_proto(num, many, mid, few, args, a_net, dataset_test,
                                                                        test_stage_idx, w_glob_keys, dict_users_test, global_protos,alllocal_protos, all_locals)
        print('Round {:3d}, all users (with global protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of test loss is {:.3f}, mean of test tail is {:.3f}'.format(sta,
            np.mean(acc_list_g), np.std(acc_list_g), np.mean(loss_list_g), np.mean(acc_tail_g)))

        print('Round {:3d}, all users (with local protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of test loss is {:.3f}, mean of test tail is {:.3f}'.format(
            sta,np.mean(acc_list_lg), np.std(acc_list_lg), np.mean(loss_list_l),np.mean(acc_tail_lg)))

        print('Round {:3d}, all users (no protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of test loss is {:.3f}, mean of test tail is {:.3f}'.format(sta,
            np.mean(acc_list_l), np.std(acc_list_l), np.mean(loss_list_n), np.mean(acc_tail_l)))
        print('*****************************************************************')
        glaccs_pro.append(np.mean(acc_list_g))
        glaccs_lopro.append(np.mean(acc_list_lg))
        glaccs_nopro.append(np.mean(acc_list_l))
        
        glaccs_pro_t.append(np.mean(acc_tail_g))
        glaccs_lopro_t.append(np.mean(acc_tail_lg))
        glaccs_nopro_t.append(np.mean(acc_tail_l))

        glaccs_prostd.append(np.std(acc_list_g))
        glaccs_loprostd.append(np.std(acc_list_lg))
        glaccs_noprostd.append(np.std(acc_list_l))

        if sta >= args.epochs-10:
            gl_accs10_pro += (np.mean(acc_list_g))/10
            gl_accs10_lopro += (np.mean(acc_list_lg))/10
            gl_accs10_nopro += (np.mean(acc_list_l))/10

            gl_accs10_pro_t += (np.mean(acc_tail_g))/10
            gl_accs10_lopro_t += (np.mean(acc_tail_lg))/10
            gl_accs10_nopro_t += (np.mean(acc_tail_l))/10

        ### Test all clients, using the server-side global model parameters
        num = [i for i in range(args.clients)]
        g_acc_l, g_acc_lg, g_acc_g, g_loss_n, g_loss_l, g_loss_g = test_proto(num, args, a_net, dataset_test,
                                                                        test_stage_idx, w_glob_keys, dict_users_test, global_protos,alllocal_protos, None)
        many_lis, mid_lis, few_lis, many_lis_lg, mid_lis_lg, few_lis_lg, many_lis_g, mid_lis_g, few_lis_g = tail_test_proto(num, many, mid, few, args, a_net, dataset_test,
                                                                        test_stage_idx, w_glob_keys, dict_users_test, global_protos,alllocal_protos, None)                                                                
        print('Round {:3d}, all users (with global protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of test loss is {:.3f}'.format(sta,
            np.mean(g_acc_g), np.std(g_acc_g), np.mean(g_loss_g)))
        print('Round {:3d}, global protos, mean of many acc is {:.2f}, mean of mid acc is {:.2f},mean of few acc is {:.2f}'.format(sta,
            np.mean(many_lis_g), np.mean(mid_lis_g), np.mean(few_lis_g)))

        print('Round {:3d}, all users (with local protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of test loss is {:.3f}'.format(
            sta,np.mean(g_acc_lg), np.std(g_acc_lg), np.mean(g_loss_l)))
        print('Round {:3d}, local protos, mean of many acc is {:.2f}, mean of mid acc is {:.2f},mean of few acc is {:.2f}'.format(sta,
            np.mean(many_lis_lg), np.mean(mid_lis_lg), np.mean(few_lis_lg)))

        print('Round {:3d}, all users (no protos), mean of test acc is {:.2f}, std of test acc is {:.2f}, mean of test loss is {:.3f}'.format(sta,
            np.mean(g_acc_l), np.std(g_acc_l), np.mean(g_loss_n)))
        print('Round {:3d}, no protos, mean of many acc is {:.2f}, mean of mid acc is {:.2f},mean of few acc is {:.2f}'.format(sta,
            np.mean(many_lis), np.mean(mid_lis), np.mean(few_lis)))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        g_accs_pro.append(np.mean(g_acc_g))
        g_accs_lopro.append(np.mean(g_acc_lg))
        g_accs_nopro.append(np.mean(g_acc_l))


        g_accs_pro_ma.append(np.mean(many_lis_g))
        g_accs_lopro_ma.append(np.mean(many_lis_lg))
        g_accs_nopro_ma.append(np.mean(many_lis))

        g_accs_pro_mid.append(np.mean(mid_lis_g))
        g_accs_lopro_mid.append(np.mean(mid_lis_lg))
        g_accs_nopro_mid.append(np.mean(mid_lis))
        
        g_accs_pro_f.append(np.mean(few_lis_g))
        g_accs_lopro_f.append(np.mean(few_lis_lg))
        g_accs_nopro_f.append(np.mean(few_lis))

        g_accs_prostd.append(np.std(g_acc_g))
        g_accs_loprostd.append(np.std(g_acc_lg))
        g_accs_noprostd.append(np.std(g_acc_l))

        if sta >= args.epochs-10:
            g_accs10_pro += (np.mean(g_acc_g))/10
            g_accs10_lopro += (np.mean(g_acc_lg))/10
            g_accs10_nopro += (np.mean(g_acc_l))/10

            g_accs10_pro_ma += (np.mean(many_lis_g))/10
            g_accs10_lopro_ma += (np.mean(many_lis_lg))/10
            g_accs10_nopro_ma += (np.mean(many_lis))/10

            g_accs10_pro_mi += (np.mean(mid_lis_g))/10
            g_accs10_lopro_mi += (np.mean(mid_lis_lg))/10
            g_accs10_nopro_mi += (np.mean(mid_lis))/10

            g_accs10_pro_t += (np.mean(few_lis_g ))/10
            g_accs10_lopro_t += (np.mean(few_lis_lg))/10
            g_accs10_nopro_t += (np.mean(few_lis))/10

            gl_accs10_prostd += (np.std(acc_list_g))/10
            gl_accs10_loprostd += (np.std(acc_list_lg))/10
            gl_accs10_noprostd += (np.std(acc_list_l))/10

            g_accs10_prostd += (np.std(g_acc_g))/10
            g_accs10_loprostd += (np.std(g_acc_lg))/10
            g_accs10_noprostd += (np.std(g_acc_l))/10

    print("The accs of select local clients: ")
    se_gacc, se_lacc, se_nacc = [],[],[]
    se_gacc_t, se_lacc_t, se_nacc_t = [],[],[]
    se_gaccstd, se_laccstd, se_naccstd = [],[],[]
    for i in range(stage):
        print('sta{:3d}, g_acc, l_acc, n_acc:{},{},{}'.format(i, stage_goacc[i], stage_loacc[i], stage_noacc[i]))
        print('sta{:3d}, g_acc_tail, l_acc_tail, n_acc_tail :{},{},{}'.format(i, stage_goacc_t[i], stage_loacc_t[i], stage_noacc_t[i]))
        se_gacc.append(sum(stage_goacc[i][args.epochs-10:args.epochs])/10)
        se_lacc.append(sum(stage_loacc[i][args.epochs-10:args.epochs])/10)
        se_nacc.append(sum(stage_noacc[i][args.epochs-10:args.epochs])/10)   

        se_gacc_t.append(sum(stage_goacc_t[i][args.epochs-10:args.epochs])/10)
        se_lacc_t.append(sum(stage_loacc_t[i][args.epochs-10:args.epochs])/10)
        se_nacc_t.append(sum(stage_noacc_t[i][args.epochs-10:args.epochs])/10)  

        se_gaccstd.append(sum(stage_goaccstd[i][args.epochs-10:args.epochs])/10)
        se_laccstd.append(sum(stage_loaccstd[i][args.epochs-10:args.epochs])/10)
        se_naccstd.append(sum(stage_noaccstd[i][args.epochs-10:args.epochs])/10) 
    
    print('select_stage_acc:{}, {}, {}'.format(se_gacc,se_lacc,se_nacc))
    print('select_stage_acc_tail:{}, {}, {}'.format(se_gacc_t,se_lacc_t,se_nacc_t))

    print(glaccs)
    print(glaccs_pro)
    print(glaccs_lopro)
    print(glaccs_nopro)

    print(glaccs_pro_t)
    print(glaccs_lopro_t)
    print(glaccs_nopro_t)

    print(g_accs)
    print(g_accs_pro)
    print(g_accs_lopro)
    print(g_accs_nopro)

    print(g_accs_pro_ma)
    print(g_accs_lopro_ma)
    print(g_accs_nopro_ma)

    print(g_accs_pro_mid)
    print(g_accs_lopro_mid)
    print(g_accs_nopro_mid)

    print(g_accs_pro_f)
    print(g_accs_lopro_f)
    print(g_accs_nopro_f)
    print('allpro_wklt: Average all clients global accuracy (10): {}, {}, {}'.format(g_accs10_pro, g_accs10_lopro, g_accs10_nopro))
    print('allpro_wklt: Average all clients global many accuracy (10): {}, {}, {}'.format(g_accs10_pro_ma, g_accs10_lopro_ma, g_accs10_nopro_ma))
    print('allpro_wklt: Average all clients global mid accuracy (10): {}, {}, {}'.format(g_accs10_pro_mi, g_accs10_lopro_mi, g_accs10_nopro_mi))
    print('allpro_wklt: Average all clients global tail accuracy (10): {}, {}, {}'.format(g_accs10_pro_t, g_accs10_lopro_t, g_accs10_nopro_t))
    logger.reset()

