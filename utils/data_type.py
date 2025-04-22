import numpy as np
import random 
def onli(dataset_test_user,dataset_test_ind,dataset_train_niid,sta_client,test_sta_client,test_stage_ind,idxs_users,iter,seed):
    random.seed(seed)
    for idx, i in enumerate(idxs_users):
        random.shuffle(sta_client[i][iter])
        random.shuffle(test_sta_client[i][iter])
        random.shuffle(test_stage_ind[i][iter])
        
        if iter == 0:
            dataset_test_user.append(test_sta_client[i][iter])
            dataset_test_ind.append(test_stage_ind[i][iter])
            dataset_train_niid.append(sta_client[i][iter])
        else:
            a = np.array(dataset_test_user[idx])
            a = np.append(a, test_sta_client[i][iter])
            dataset_test_user[idx] = a

            b = np.array(dataset_test_ind[idx])
            b = np.append(b, test_stage_ind[i][iter])
            dataset_test_ind[idx] = b

            c = np.array(dataset_train_niid[idx])
            c = np.append(c, sta_client[i][iter])
            dataset_train_niid[idx] = c
    
    return dataset_train_niid, dataset_test_user, dataset_test_ind