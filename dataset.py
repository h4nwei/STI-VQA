from PIL import Image
import json
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepFeatDatabase(Dataset):
    def __init__(self, data_root, mos_file, phase='train'):
        with open(mos_file, 'r') as f:
            mos_data = json.load(f)
        mos_data = mos_data[phase]

        self.video_name = mos_data['dis']
        self.video_mos = mos_data['mos']
        self.feat = [np.load(data_root / f'{vn[:-4]}.npy') for vn in self.video_name]
        self.video_mos = [(float(s)-1.0)/4.0 for s in self.video_mos]
        # self.video_mos = [float(s) for s in self.video_mos]


    def __len__(self) -> int:
        return len(self.video_name)

    def __getitem__(self, item):
        data = self.feat[item]
        mos = self.video_mos[item]
        return data, mos


class DeepFeatDataset(Dataset):
    def __init__(self, arg, phase: str = None, shuffle: str = False):
        super(DeepFeatDataset, self).__init__()
        assert phase == 'train' or phase == 'val' or phase == 'test'

        # 数据存放根目录
        self.data_root = arg.data_root
        # 每个视频的最大帧数
        self.max_len = arg.max_len
        # 每帧的特征维度
        self.feat_dim = arg.d_feat
        
        '''
        根据 phase 得到训练/测试/验证数据
        '''
        with open(arg.data_info_dir) as f:
            info: dict = json.load(f)
        # 获取所有数据名
        self.data_name_list = info['video_name_list']
        # 获得所有数据标签
        self.label_list = info['video_label_list']
        # 获取MOS最大值，用于归一化训练数据
        self.scale = max(self.label_list)
        # 根据 state 获得所需数据的索引
        if phase == 'train':
            idx_list = info['train_idx']
        elif phase == 'val':
            idx_list = info['val_idx']
        elif phase == 'test':
            idx_list = info['test_idx']
        self.idx_list = idx_list
        # # 通过索引获得训练/验证/测试数据绝对路径
        # self.data_path = [os.path.join(dataset_info['data_root'], data_name_list[idx]) for idx in idx_list]
        # # 获得相应数据的MOS
        # self.labels = [info['video_label_list'][idx] for idx in idx_list]
        # 特征是否shuffle
        self.shuffle = shuffle
        self.feat_len = [512, 1024, 2048, 4096]
        self.video = []
        self.len = []
        self.label = []
        # self.scale = []


    def __getitem__(self, idx):
        # 获得所需数据的索引
        data_idx = self.idx_list[idx]
        # 获得所需数据名
        data_name = self.data_name_list[data_idx]
        # 加载数据
        npy_name = os.path.join(self.data_root, data_name) + '.npy'
        if os.path.isfile(npy_name) is False:
            print(npy_name)
            return self.video, self.len, self.label, self.scale
        # self.scale
        feature_data = np.load(npy_name)
        # feature_data = feature_data[:, 4096:8192]
        feature_data = np.split(feature_data, [512, 1536, 3584, 7680], 1)
        # feature_data = d[3]
        # mu = feature_data[:, 0:2048]
        # std = feature_data[:, 2048:2096]

        # feature_data = np.concatenate((feature_data[:, 0:2048], feature_data[:, 4096:6144]), 1)
        # 打乱数据
        if self.shuffle:
            np.random.RandomState(123).shuffle(feature_data)
        # 每个视频的帧数
        length = feature_data[0].shape[0]
        video = []

        for index, feat_len in enumerate(self.feat_len):
            data = np.zeros([self.max_len, feat_len], dtype=np.float32)
            data[:length] = feature_data[index]
            video.append(data)
        # 获得标签
        label = self.label_list[data_idx]

        # self.video = video
        # self.len = length
        # self.label = label / self.scale
        # self.scale = []

        # data = np.zeros([self.max_len, self.feat_dim], dtype=np.float32)
        # data[:length] = np.squeeze(feature_data)
        # # 获得标签
        # label = self.label_list[data_idx]

        # return data, length, label / self.scale, self.scale

        return video, length, label / self.scale, self.scale
        
    
        # return torch.cat([last_feature.unsqueeze(1), diff_feature], dim=1)
    def __len__(self):
        # print(len(self.idx_list))
        return len(self.idx_list)-1




class DeepMSFeatDataset(Dataset):
    def __init__(self, arg, phase: str = None, shuffle: str = True):
        super(DeepMSFeatDataset, self).__init__()
        assert phase == 'train' or phase == 'val' or phase == 'test'

        # 数据存放根目录
        self.data_root = arg.data_root
        # 每个视频的最大帧数
        self.max_len = arg.max_len
        # 每帧的特征维度
        self.feat_dim = arg.input_size
        
        '''
        根据 phase 得到训练/测试/验证数据
        '''
        with open(arg.data_info_dir) as f:
            info: dict = json.load(f)
        # 获取所有数据名
        self.data_name_list = info['video_name_list']
        # 获得所有数据标签
        self.label_list = info['video_label_list']
        # 获取MOS最大值，用于归一化训练数据
        self.scale = max(self.label_list)
        # 根据 state 获得所需数据的索引
        if phase == 'train':
            idx_list = info['train_idx']
        elif phase == 'val':
            idx_list = info['val_idx']
        elif phase == 'test':
            idx_list = info['test_idx']
        self.idx_list = idx_list
        # # 通过索引获得训练/验证/测试数据绝对路径
        # self.data_path = [os.path.join(dataset_info['data_root'], data_name_list[idx]) for idx in idx_list]
        # # 获得相应数据的MOS
        # self.labels = [info['video_label_list'][idx] for idx in idx_list]
        # 特征是否shuffle
        self.shuffle = shuffle

    
    def __getitem__(self, idx):
        # 获得所需数据的索引
        data_idx = self.idx_list[idx]
        # 获得所需数据名
        data_name = self.data_name_list[data_idx]
        # 加载数据
        # feature_data = np.load(os.path.join(self.data_root, data_name) + '.npy')
        feature_datas = np.load(os.path.join(self.data_root, data_name) + '.npz')
        ti = feature_datas['ti']
        diff_mu_std = feature_datas['diff_mu_std']
        d = np.split(diff_mu_std, [512, 1536, 3584, 7680], 1)
        mu_std = feature_datas['mu_std']
        s = np.split(mu_std, [512, 1536, 3584, 7680], 1)
        feature_data = np.concatenate((s[3], d[3]), 1) ##4096*2
        # feature_data = s[3] ##4096*2
        # 打乱数据
        if self.shuffle:
            np.random.RandomState(123).shuffle(feature_data)
        # 每个视频的帧数
        length = feature_data.shape[0]
        data = np.zeros([self.max_len, self.feat_dim], dtype=np.float32)
        data[:length] = np.squeeze(feature_data)
        # 获得标签
        label = self.label_list[data_idx]

        return data, length, label / self.scale, self.scale
        
        
    def __len__(self):
        # print(len(self.idx_list))
        return len(self.idx_list)







class FrameDatabase(Dataset):
    def __init__(self, data_root, mos_file, phase='train', video_len=50, size=(448, 448), rgb='RGB', multi_crop=False):
        """
        video_len: all available frame number for each video
        size(h, w): crop video to f frames with h x w
        """
        with open(mos_file, 'r') as f:
            mos_data = json.load(f)
        mos_data = mos_data[phase]        
        
        self.phase = phase
        self.video_len = video_len
        self.th, self.tw = size
        self.rgb = rgb
        self.multi_crop = multi_crop
        self.video_name = mos_data['dis']
        self.video_mos = mos_data['mos']
        self.video_mos = [(float(s)-1.0)/4.0 for s in self.video_mos]
        self.frame_path = [[data_root / vn[:-4] / f'{fi:03d}.png' for fi in range(1, video_len + 1)] for vn in self.video_name]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.spatial_crop = RandomImageCrop((self.th, self.tw))
        self.spatial_multi_crop = FiveVideoCrop((self.th, self.tw))

    def __len__(self) -> int:
        return len(self.video_name)

    def __getitem__(self, item):
        frame_path = self.frame_path[item]
        mos = self.video_mos[item]

        if not self.multi_crop:
            fi = torch.randint(0, self.video_len, size=(1, )).item()
            frame = Image.open(frame_path[fi]).convert(self.rgb)
            frame = self.transform(frame)
            crop = self.spatial_crop(frame)
        else:
            video = torch.Tensor()
            for fi in range(self.video_len):
                frame = Image.open(frame_path[fi]).convert(self.rgb)
                frame = self.transform(frame)
                video = torch.cat((video, frame.unsqueeze(0)), 0)
            crop = self.spatial_multi_crop(video)
        return crop, mos


class FrFrameDatabase(Dataset):
    def __init__(self, data_root, mos_file, phase='train', video_len=50, size=(448, 448), rgb='RGB'):
        """
        video_len: all available frame number for each video
        size(h, w): crop video to f frames with h x w
        """
        with open(mos_file, 'r') as f:
            mos_data = json.load(f)
        mos_data = mos_data[phase]        
        
        self.phase = phase
        self.video_len = video_len
        self.th, self.tw = size
        self.rgb = rgb
        self.video_name = mos_data['dis']
        self.video_mos = mos_data['mos']
        self.video_mos = [float(s)/5.0 for s in self.video_mos]
        self.dis_frames = [[data_root / vn[:-4] / f'{fi:03d}.png' for fi in range(1, video_len + 1)] for vn in self.video_name]
        self.ref_frames = [[data_root / f'{vn[:-6]}00' / f'{fi:03d}.png' for fi in range(1, video_len + 1)] for vn in self.video_name]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.spatial_crop = RandomImageCrop((self.th, self.tw))
        self.multi_crop = FiveVideoCrop((self.th, self.tw))

    def __len__(self) -> int:
        return len(self.video_name)

    def __getitem__(self, item):
        ref_frames = self.ref_frames[item]
        dis_frames = self.dis_frames[item]
        mos = self.video_mos[item]

        fi = torch.randint(0, self.video_len, size=(1, )).item()
        ref = Image.open(ref_frames[fi]).convert(self.rgb)
        dis = Image.open(dis_frames[fi]).convert(self.rgb)
        ref, dis = self.transform(ref), self.transform(dis)
        ref, dis = self.spatial_crop(ref, dis)
        return (ref, dis), mos


class RandomImageCrop(torch.nn.Module):
    """Crop the given frame at a random location.
    """
    def __init__(self, size):
        super().__init__()
        self.th, self.tw = size

    def forward(self, dis, ref=None):
        c, h, w = dis.shape
        i = torch.randint(0, h - self.th + 1, size=(1, )).item()
        j = torch.randint(0, w - self.tw + 1, size=(1, )).item()
        if ref is not None:
            return dis[:, i:i+self.th, j:j+self.tw], ref[:, i:i+self.th, j:j+self.tw]
        return dis[:, i:i+self.th, j:j+self.tw]


class CenterImageCrop(torch.nn.Module):
    """Crop the given frame at a center location.
    """
    def __init__(self, size):
        super().__init__()
        self.th, self.tw = size

    def forward(self, dis, ref=None):
        c, h, w = dis.shape

        i = (h - self.th) // 2
        j = (w - self.tw) // 2
        if ref is not None:
            return dis[:, i:i+self.th, j:j+self.tw], ref[:, i:i+self.th, j:j+self.tw]
        return dis[:, i:i+self.th, j:j+self.tw]


class FiveVideoCrop(torch.nn.Module):
    """Crop the given video to multi patches, four corners and center.
    """
    def __init__(self, size):
        super().__init__()
        self.th, self.tw = size

    def forward(self, video):
        n, c, h, w = video.shape

        hc = (h - self.th) // 2
        wc = (w - self.tw) // 2
        left_top = video[:, :, 0:self.th, 0:self.tw]
        left_down = video[:, :, h-self.th:h, 0:self.tw]
        right_top = video[:, :, 0:self.th, w-self.tw:w]
        right_down = video[:, :, h-self.th:h, w-self.tw:w]
        center = video[:, :, hc:hc+self.th, wc:wc+self.tw]
        video_slice = torch.cat([left_top, left_down, right_top, right_down, center], dim=0)
        return video_slice
