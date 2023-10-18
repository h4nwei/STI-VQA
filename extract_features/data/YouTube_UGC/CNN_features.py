"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
#
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8

from re import X, sub
from numpy.core.arrayprint import printoptions
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import extract
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import argparse
from pathlib import Path
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES']='3'

class VideoFrameDataset(Dataset):
    """
    Test or Validation
    """
    def __init__(self, video_names, video_folder):
        self.video_folder = video_folder
        self.video_names = video_names
        self.data_list = self._make_dataset()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _make_dataset(self):
        data_list = []
        for idx in range(len(self.video_names)):
            video_name = self.video_names[idx]
            frame_names = list(os.walk(os.path.join(self.video_folder, str(video_name))))[0]
            frame_paths = list(map(lambda x: os.path.join(frame_names[0], x),
                                              sorted(frame_names[2], key=lambda x: (x.split('.')[0]))))
            data_list.append((str(video_name), frame_paths))
        return data_list
        
    def __getitem__(self, index):
        video_name, frame_paths = self.data_list[index]
        
        len_video = len(frame_paths)
        frames = []
        for i in range(len_video):
            # print(f'\t{video_name} image idx {i}')
            img = Image.open(frame_paths[i])
            img = self.transform(img)
            frames.append(img)
        transformed_data = torch.zeros([len_video, *frames[0].shape])
        for i in range(len_video):
            transformed_data[i] = frames[i]
        return video_name, transformed_data

    def __len__(self):
        return len(self.video_names)

class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        # self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        # video_dir = self.videos_dir[idx]
        
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name+'.yuv'), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
        else:
            # if os.path.isfile(os.path.join(self.videos_dir, video_name)):
            # print(os.path.join(self.videos_dir, video_name))
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
            # else:
            # #     video_name_ = video_name.split('/')[1]
            #     video_data = skvideo.io.vread(os.path.join(self.videos_dir, 'Valid', video_name_))

        # video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        # print('video_width: {} video_height: {}'.format(video_width, video_height))
        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        # transformed_video = []
        for frame_idx in range(video_length):
            # print(frame_idx)
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            # frame.show()
            frame = transform(frame)
            # transformed_video.append(frame)
            transformed_video[frame_idx] = frame
        # transformed_data = torch.zeros([video_length, *frames[0].shape])
        # for i in range(len_video):
        #     transformed_data[i] = frames[i]
        # sample = {'video': transformed_video,
        #           'score': video_score}
        video_name_ = video_name.split('_')[0]+'_'+video_name.split('_')[1]
        return video_name_, transformed_video

def local_mean_std_pool2d(x):
    mean_x = nn.functional.avg_pool2d(x, kernel_size=3, padding=1, stride=1)
    mean_x_square = nn.functional.avg_pool2d(x * x, kernel_size=3, padding=1, stride=1)
    var_x = mean_x_square - mean_x * mean_x 
    # std_x = torch.sqrt(var_x + 1e-8)
    return var_x / (mean_x + 1e-8) #spatial motion variation,

def globale_mean_std_pool2d(x):
    mean = nn.functional.adaptive_avg_pool2d(x, 1)
    std = global_std_pool2d(x)
    result = torch.cat([mean, std], 1)
    return result

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def SPSP(x, P=1, method='avg'):
    batch_size = x.size(0)
    map_size = x.size()[-2:]
    pool_features = []
    for p in range(1, P+1):
        pool_size = [np.int(d / p) for d in map_size]
        if method == 'maxmin':
            M = F.max_pool2d(x, pool_size)
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(torch.cat((M, m), 1).view(batch_size, -1))  # max & min pooling
        elif method == 'max':
            M = F.max_pool2d(x, pool_size)
            pool_features.append(M.view(batch_size, -1))  # max pooling
        elif method == 'min':
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(m.view(batch_size, -1))  # min pooling
        elif method == 'avg':
            a = F.avg_pool2d(x, pool_size)
            pool_features.append(a.view(batch_size, -1))  # average pooling
        else:
            m1  = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(torch.cat((m1, rm2), 1).view(batch_size, -1))  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)


class IQAModel(nn.Module):
    def __init__(self, arch='resnext101_32x8d', pool='avg', use_bn_end=False, P6=1, P7=1):
        super(IQAModel, self).__init__()
        self.pool = pool
        self.use_bn_end = use_bn_end
        if pool in ['max', 'min', 'avg', 'std']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #
        features = list(models.__dict__[arch](pretrained=True).children())[:-2]
        if arch == 'alexnet':
            in_features = [256, 256]
            self.id1 = 9
            self.id2 = 12
            features = features[0]
        elif arch == 'vgg16':
            in_features = [512, 512]
            self.id1 = 23
            self.id2 = 30
            features = features[0]
        elif 'res' in arch:
            self.id1 = 6
            self.id2 = 7
            if arch == 'resnet18' or arch == 'resnet34':
                in_features = [256, 512]
            else:
                in_features = [1024, 2048]
        else: 
            print('The arch is not implemented!')
        self.features = nn.Sequential(*features)
        self.dr6 = nn.Sequential(nn.Linear(in_features[0] * c * sum([p * p for p in range(1, self.P6+1)]), 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), nn.ReLU())
        self.dr7 = nn.Sequential(nn.Linear(in_features[1] * c * sum([p * p for p in range(1, self.P7+1)]), 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), nn.ReLU())

        if self.use_bn_end:
            self.regr6 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regr7 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regression = nn.Sequential(nn.Linear(64 * 2, 1), nn.BatchNorm1d(1))
        else:
            self.regr6 = nn.Linear(64, 1)
            self.regr7 = nn.Linear(64, 1)
            self.regression = nn.Linear(64 * 2, 1)

    def extract_features(self, x):
 
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 4:
                mu_std_1 = globale_mean_std_pool2d(x)
            elif ii == 5:
                mu_std_2 = globale_mean_std_pool2d(x)
            elif ii == 6:
                mu_std_3 = globale_mean_std_pool2d(x)
            elif ii == 7:
                mu_std_4 = globale_mean_std_pool2d(x)

        result_1 = {'mu_std_1': mu_std_1}
        result_2 = {'mu_std_2': mu_std_2}
        result_3 = {'mu_std_3': mu_std_3}
        result_4 = {'mu_std_4': mu_std_4}

        return result_1, result_2, result_3, result_4

        # return feat
    def forward(self, x):
        f = self.extract_features(x)
        return f





def get_features(video_data, frame_batch_size=16, device='cuda'):
    """feature extraction"""
    # extractor = torch.nn.DataParallel(ResNet50()).to(device)
    # torch.nn.DataParallel(model).cuda()
    # extractor = CNNModel(model='SpatialExtractor')
    extractor = IQAModel()
    torch.nn.DataParallel(extractor).cuda()
    # checkpoint = torch.load('/home/zhw/vqa/code/VQA-framework/feature/LinearityIQA/weight/p1q2plus0.1variant.pth')
    # extractor.load_state_dict(checkpoint['model'])
    video_length = video_data.shape[0]
    # print('video length', video_length)
    frame_start = 0
    frame_end = frame_start + frame_batch_size

    r1_mu_std = torch.Tensor().to(device)
    r2_mu_std = torch.Tensor().to(device)
    r3_mu_std = torch.Tensor().to(device)
    r4_mu_std = torch.Tensor().to(device)


    extractor.eval()
    result = []
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            r1,r2,r3,r4= extractor(batch)
            
            r1_mu_std = torch.cat((r1_mu_std, r1['mu_std_1']), 0)
            r2_mu_std = torch.cat((r2_mu_std, r2['mu_std_2']), 0)
            r3_mu_std = torch.cat((r3_mu_std, r3['mu_std_3']), 0)
            r4_mu_std = torch.cat((r4_mu_std, r4['mu_std_4']), 0)
           
            frame_end += frame_batch_size
            frame_start += frame_batch_size
        last_batch = video_data[frame_start:video_length].to(device)
        r1,r2,r3,r4 = extractor(last_batch)
       
        r1_mu_std = torch.cat((r1_mu_std, r1['mu_std_1']), 0)
        r2_mu_std = torch.cat((r2_mu_std, r2['mu_std_2']), 0)
        r3_mu_std = torch.cat((r3_mu_std, r3['mu_std_3']), 0)
        r4_mu_std = torch.cat((r4_mu_std, r4['mu_std_4']), 0)
       
        
        mu_std = torch.cat((r1_mu_std.squeeze(), r2_mu_std.squeeze(), r3_mu_std.squeeze(), r4_mu_std.squeeze()), 1)

    
    return mu_std


def main(dataset, features_folder):
    '''
    '''
    
    for i in range(len(dataset)):
        video_name, current_data = dataset[i]
        print('Video {}: video name {}: length {}'.format(i, video_name, current_data.shape[0]))
   
        saved_name = os.path.join(features_folder, video_name)
        # save_folder = os.path.join(features_folder, video_name.split('/')[1])
        # if not os.path.exists(save_folder):
            # os.makedirs(save_folder)
        feature_path = saved_name + '.npy'
        if os.path.isfile(feature_path):
            continue

        feat_mu_std = get_features(current_data, frame_batch_size=4)
        # feat_save = torch.cat((feat_diff_mu_std, feat_mu_std), 1)
        # print(video_name, ': ', feat_diff_mu_std.shape, feat_mu_std.shape, feat_save.shape)
        np.save(feature_path, feat_mu_std.to('cpu').numpy())
        # np.savez(feature_path, mu_std=feat_mu_std.to('cpu').numpy())

if __name__ == "__main__":
    # import fire
    # fire.Fire()

    features_folder = r'/home/zhw/vqa/dataset/YouTube_UGC/feature/VSFA_resnetxt101_ImageNetpretrain_ms'
    video_folder = r'/data/zhw/YouTube_UGC/original_videos_h264'
    csv_file = Path('/home/zhw/vqa/code/VQA-framework/feature/data/YouTube_UGC/YouTube_UGC.csv')
    video_info = pd.read_csv(csv_file, header=0)
    video_names = os.listdir(video_folder)
    
    # video_names = video_info.iloc[:, 1].tolist()    
    # sub_video_names = sub_video_names[:len(sub_video_names)//2]
    # category_video_names = video_info.iloc[:, 3].tolist()
    # category_video_names = category_video_names[:len(category_video_names)//2]
    # video_moses = video_info.iloc[:, 2].tolist()
    # video_moses = video_moses[len(video_moses)//2:]
    # video_names = os.listdir(video_root)
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    for index, exist_name in enumerate(os.listdir(features_folder)):
        exist_name = exist_name[:-4] + '_crf_10_ss_00_t_20.0.mp4'
        # ind = video_names.index(exist_name)
        # category_video_names.pop(ind)
        # video_moses.pop(ind)
        video_names.remove(exist_name)
    video_names = video_names[:len(video_names)//2]
    print('exist video number: {}'.format(len(video_names)))
    # video_names = []
    # for cat_name, video_name in zip(category_video_names, sub_video_names):
    #     video_names.append(os.path.join(cat_name.replace(" ", ""), video_name))
    
    # video_names = [os.path.join(video_name.split('/')[0], video_name) for video_name in sub_video_names]
    # df_info = pd.read_csv(database_info_path)
    # video_names = df_info['video_name'].values

    

    mp.set_start_method('spawn')
    num_processes = 1
    processes = []
    nvideo_per_node = int(len(video_names) / num_processes)
    # nvideo_per_node = 1
    for rank in range(num_processes):
        video_names_ = video_names[rank*nvideo_per_node:(rank+1)*nvideo_per_node]
        # video_mos_ = video_moses[rank*nvideo_per_node:(rank+1)*nvideo_per_node]
        dataset = VideoDataset(video_folder, video_names_)
        p = mp.Process(target=main, args=(dataset, features_folder))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
