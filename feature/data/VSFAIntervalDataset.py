from torch.utils.data import Dataset
import numpy as np
import os
import random


class VSFAIntervalDataset(Dataset):
    def __init__(self, idx_info: dict, database_info: dict,
                 state: str = None, feature_shuffle: bool = None,
                 max_len: int = 300, feat_dim: int = 4096,
                 n_frames: int = 16):
        super(VSFAIntervalDataset, self).__init__()
        assert state == 'train' or state == 'val' or state == 'test'
        
        idx_list = idx_info[state]
        self.idx_list = idx_list
        feature_file_names = database_info['feature_file_names']
        self.feature_file_names = feature_file_names[idx_list]
        labels = database_info['labels'] / database_info['scale']
        labels = labels.astype(np.float32)
        self.labels = labels[idx_list]
        assert self.labels.dtype == np.float32
        
        self.feature_folder = database_info['feature_folder']
        self.max_len = max_len
        self.feat_dim = feat_dim
        assert feature_shuffle == True or feature_shuffle == False
        self.feature_shuffle = feature_shuffle
        # n_frames = -1 -> fetch all frames
        self.n_frames = n_frames
    
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        feature_data = np.load(os.path.join(self.feature_folder, self.feature_file_names[idx]))

        length = feature_data.shape[0]
        if self.n_frames > 0:
            # start_idx = list(range(0, length, length // self.n_frames))
            # len_start_idx = len(start_idx)
            # choosed_idx = []
            # for i in range(len_start_idx):
            #     start = start_idx[i]
            #     end = start_idx[i + 1] if i != len_start_idx - 1 else length
            #     choosed_idx.append(random.choice(range(start, end)))
            start_idx = list(range(0, length, self.n_frames))
            end_idx = start_idx[1:] + [length]
            choosed_idx = []
            for i in range(len(start_idx)):
                choosed_idx.append(random.choice(range(start_idx[i], end_idx[i])))
            feature_data = feature_data[choosed_idx]
            length = feature_data.shape[0]
        
        if self.feature_shuffle:
            np.random.shuffle(feature_data)
        data = np.zeros([self.max_len, self.feat_dim], dtype=np.float32)
        data[:feature_data.shape[0]] = feature_data
        label = self.labels[idx]
        
        return data, length, label

def tmp():
    feature_data = np.load(r"C:\freetime\code\VQA\dataset\vsfa_resnet50_feature\KoNViD\9445782126.npy")
    n_frames = 16
    length = feature_data.shape[0]
    # length = 17
    start_idx = list(range(0, length, n_frames))
    end_idx = start_idx[1:]+[length]
    choosed_idx = []
    for i in range(len(start_idx)):
        choosed_idx.append(random.choice(range(start_idx[i], end_idx[i])))
    print(start_idx)
    print(end_idx)
    print(choosed_idx)
    print(len(choosed_idx))

if __name__ == '__main__':
    tmp()