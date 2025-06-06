import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--ave', default=0.5, type=float, help="average prototype")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--clients', type=int, default=20, help="rounds of continuelearning")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: n,epochs*clients")
    parser.add_argument('--shard_per_user', type=int, default=4, help="classes per user")
    parser.add_argument('--imb_factor', default=0.5, type=float, help='IF, imbalance factor')
    parser.add_argument('--temperature', default=0.07, type=float, help='')
    parser.add_argument('--base_temperature', default=0.07, type=float, help='')

    parser.add_argument('--num_online_clients', type=int, default=8) 
    
    parser.add_argument('--local_ep', type=int, default=30, help="the number of local epochs: E")
    parser.add_argument('--local_rep_ep', type=int, default=25, help="the number of local epochs for the representation for FedRep")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--noniid', type=str, default='shard', help="the different noniid type")
    parser.add_argument('--loss', type=str, default='kl', help="the different loss, cos | kl")
    
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--non_iid_alpha', type=float, default=0.4, help='Dirichlet distribution')
    parser.add_argument('--simi', type=int, default=1, help='help="1,0')

    parser.add_argument('--momentum', type=float, default=0.1, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=500, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=500, help="maximum number of samples/user to use for fine-tuning")
    parser.add_argument('--ld', type=float, default=1, help="weight of proto loss")
    parser.add_argument('--lam_1', type=float, default='0.5', help='mse parameter lambda')
    parser.add_argument('--lam_2', type=float, default='0.5', help='kl parameter lambda')

    parser.add_argument('--tau', type=float, default='1', help='fedntd')
    parser.add_argument('--beta', type=float, default='1', help='fedntd')

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')
    parser.add_argument('--alg', type=str, default='fedrep', help='FL algorithm to use')
    
    # algorithm-specific hyperparameters
    
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    parser.add_argument('--mu', type=float, default='0.1', help='FedProx parameter mu')
    parser.add_argument('--gmf', type=float, default='0', help='FedProx parameter gmf')
    parser.add_argument('--lr_in', type=float, default='0.0001', help='PerFedAvg inner loop step size')
    parser.add_argument('--bs_frac_in', type=float, default='0.8', help='PerFedAvg fraction of batch used for inner update')
    parser.add_argument('--lam_ditto', type=float, default='1', help='Ditto parameter lambda')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=7, help='random seed (default: 7)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='n', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='runA', help='define fed results save folder')
    parser.add_argument('--save_every', type=int, default=50, help='how often to save models')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    args = parser.parse_args()
    return args
