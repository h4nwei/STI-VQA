import argparse
from pathlib import Path




parser = argparse.ArgumentParser(description='UGC VQA challenge (ICMEw2021)')

# Hardware specifications
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--n_threads', type=int, default=32, help='number of threads for data loading')


# Data specifications
# parser.add_argument('--dataset_name', type=str, default='KoNViD-1k-multiple-IQA-motion', help='dataset root directory')
# parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/KoNViD_1k/feature/VSFA_resnetxt101_iqapretrain_ms/', help='dataset root directory')
# # parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/KoNViD_1k/feature/VSFA_resnetxt101_ImageNetpretrain_ms/', help='dataset root directory')
# # parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/KoNViD_1k/feature/fused_feats/', help='dataset root directory')
# parser.add_argument('--data_info_path', type=Path, default='/home/zhw/vqa/code/VQA-framework/data/KoNViD/', help='dataset directory')
# parser.add_argument('--dataset_name', type=str, default='PUGC', help='dataset root directory')
# parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/PUGC/feature/VSFA_resnetxt101_iqapretrain_ms/', help='dataset root directory')
# parser.add_argument('--data_info_path', type=Path, default='/home/zhw/vqa/code/VQA-framework/data/PUGC/', help='dataset directory')
# parser.add_argument('--dataset_name', type=str, default='LSVQ', help='dataset root directory')
# parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/LSVQ/feature/VSFA_resnetxt101_iqapretrain_ms/', help='dataset root directory')
# parser.add_argument('--data_info_path', type=Path, default='/home/zhw/vqa/code/VQA-framework/data/LSVQ/', help='dataset directory')
parser.add_argument('--dataset_name', type=str, default='YouTubeUGC-test', help='dataset root directory')
parser.add_argument('--data_root', type=Path, default='/data/zhw/vqa_dataset/YouTube_UGC/feature/VSFA_resnetxt101_iqapretrain_ms/', help='dataset root directory')
# parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/YouTube_UGC/feature/VSFA_resnetxt101_ImageNetpretrain_ms/', help='dataset root directory')
parser.add_argument('--data_info_path', type=Path, default='/home/zhw/vqa/code/VQA-framework/data/YouTubeUGC/', help='dataset directory')
# parser.add_argument('--dataset_name', type=str, default='LIVE-VQC-multiple-ImageNet-motion', help='dataset root directory')
# # parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/VQC/feature/VSFA_resnetxt101_iqapretrain_ms/', help='dataset root directory')
# parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/VQC/feature/VSFA_resnetxt101_ImageNetpretrain_ms/', help='dataset root directory')
# # parser.add_argument('--data_root', type=Path, default='/home/zhw/vqa/dataset/VQC/feature/VSFA_resnet50/', help='dataset root directory')
# parser.add_argument('--data_info_path', type=Path, default='/home/zhw/vqa/code/VQA-framework/data/VQC/', help='dataset directory')
parser.add_argument('--max_len', type=int, default=1000, help='dataset directory')#650 for LSVQ
# parser.add_argument('--feat_dim', type=int, default=1024, help='dataset directory')
# parser.add_argument('--data_root', type=Path, default='../../icme_data/resnet50feat', help='dataset directory')
# parser.add_argument('--train_file', type=Path, default='./ugcset_mos.json', help='train file path, video_name-mos')
# Model specifications
modelparsers = parser.add_subparsers(dest='model', help='model arch name')


# Option for VSFA method
vsfa_cmd = modelparsers.add_parser('vsfa', formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='VSFA method')
vsfa_cmd.add_argument('--model_name', type=str, default='VSFA', help='name of the model')
vsfa_cmd.add_argument('--input_size', type=int, default=4096, help='iput size of each frame')#4608
vsfa_cmd.add_argument('--d_feat', type=int, default=4096, help='iput size of each frame')
vsfa_cmd.add_argument('--reduced_size', type=int, default=128, help='reduced dimension in model')
vsfa_cmd.add_argument('--hidden_size', type=int, default=32, help='hidden dimension in model')

# Option for ViT method
vit_cmd = modelparsers.add_parser('vit', formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='VIT method')
vit_cmd.add_argument('--model_name', type=str, default='vit', help='name of the model')
vit_cmd.add_argument('--d_feat', type=int, default=4096, help='input image size for ViT') ##default 8192
vit_cmd.add_argument('--depth', type=int, default=5, help='number of transformer blocks') # default  5
vit_cmd.add_argument('--att_head', type=int, default=6, help='number of heads in multi-head attention layer') #default 6
vit_cmd.add_argument('--mlp_dim', type=int, default=128, help='dimension of the MLP (FeedForward) layer') # default 128
vit_cmd.add_argument('--dim_head', type=int, default=64, help='dimension of Q K V') # default 64
vit_cmd.add_argument('--output_channel', type=int, default=1, help='Output channel number')
vit_cmd.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
vit_cmd.add_argument('--pool', type=str, default='reg', help='output result')
vit_cmd.add_argument('--emb_dropout', type=float, default=0.1, help='embedding dropout rate')

# Training specifications
parser.add_argument('--test_every', type=int, default=1, help='do test per every N epochs')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')#default 128
parser.add_argument('--test_batch', type=int, default=1, help='input batch size for test')

# Testing specifications
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
#parser.add_argument('--test_only', default=False, help='set this option to test the model')
parser.add_argument('--pre_train', type=str, default='/home/zhw/vqa/code/VQA-framework/ckpts/vit/LSVQ/1/best_val.pth', help='where saved trained checkpoints')
parser.add_argument('--predict_res', type=str, default=None, help='where to save predicted results')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')#1e-3 defatule
# parser.add_argument('--decay', type=str, default='3-6-9-12-15-18-21-24-27', help='learning rate decay type')
parser.add_argument('--decay', type=str, default='2-4-6-8-10-12-14-16-18', help='learning rate decay type')
# parser.add_argument('--decay', type=str, default='2-4-6-8-10-12-14-16-18-20-22-24-26-28-30-32-34-36-38', help='learning rate decay type')
# parser.add_argument('--decay', type=str, default='50-100-150-200-250-300', help='learning rate decay type')
# parser.add_argument('--decay', type=str, default='200-400', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.8, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMW'),
                    help='optimizer to use (SGD | ADAM | RMSprop | ADAMW)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Loss specifications
# parser.add_argument('--loss', type=str, default='1*L1', help='loss function weights and types')
parser.add_argument('--loss', type=str, default='1*norm-in-norm', help='loss function weights and types')
# parser.add_argument('--loss', type=str, default='1*MSE', help='loss function weights and types')

# Log specifications
parser.add_argument('--log_root', type=Path, default='./logs/', help='directory for saving model weights and log file')
parser.add_argument('--ckpt_root', type=Path, default='./ckpts/', help='dataset root directory')
# parser.add_argument('--log_dir', type=Path, default='./ugcvqa_res/vsfa', help='directory for saving model weights and log file')
parser.add_argument('--save_weights', type=int, default=1000, help='how many epochs to wait before saving model weights')
parser.add_argument('--save_scatter', type=int, default=1000, help='how many epochs to wait before saving scatter plot')

# args = parser.parse_args(['vsfa'])
# args = parser.parse_args(['VQATransformer'])
# args = parser.parse_args(['cross-vit'])
args = parser.parse_args(['vit'])
# args = parser.parse_args
