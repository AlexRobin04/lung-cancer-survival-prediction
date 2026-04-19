import os
from datasets.dataset_generic import *
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=4,
                    help='number of splits (default: 10)')
parser.add_argument('--task',default='task_tcga_stad', type=str)
parser.add_argument('--val_frac', type=float, default= 0.15,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.25,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--dataset', type=str, default='TCGA_STAD')

args = parser.parse_args()

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])

elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_Lung_subtyping.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'LUAD':0, 'LUSC':1},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])

elif args.task == 'task_tcga_esca':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/ubuntu/project/ViLa-MIL/datasets_csv/ESCA.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'low':0,'Moderate':1, 'Elevated':2, 'high':3},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
elif args.task == 'task_tcga_hnsc':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/ubuntu/project/ViLa-MIL/datasets_csv/HNSC.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'low':0,'Moderate':1, 'Elevated':2, 'high':3},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
elif args.task == 'task_tcga_kirc':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/ubuntu/project/ViLa-MIL/datasets_csv/KIRC.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'low':0,'Moderate':1, 'Elevated':2, 'high':3},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
elif args.task == 'task_tcga_lihc':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/ubuntu/project/ViLa-MIL/datasets_csv/LIHC.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'low':0,'Moderate':1, 'Elevated':2, 'high':3},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
elif args.task == 'task_tcga_lusc':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/ubuntu/project/ViLa-MIL/datasets_csv/LUSC.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'low':0,'Moderate':1, 'Elevated':2, 'high':3},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
    
elif args.task == 'task_tcga_stad':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/ubuntu/project/ViLa-MIL/datasets_csv/STAD.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'low':0,'Moderate':1, 'Elevated':2, 'high':3},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])



else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ args.dataset +'/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



