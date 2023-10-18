from torch.utils.data import Dataset
import numpy as np
import os

    
class VSFADataset(Dataset):
    def __init__(self, idx_info: dict, database_info: dict,
                 state: str = None, feature_shuffle: bool = None,
                 max_len: int = 300, feat_dim: int = 4096):
        super(VSFADataset, self).__init__()
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
        
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        feature_data = np.load(os.path.join(self.feature_folder, self.feature_file_names[idx]))
        length = feature_data.shape[0]
        if self.feature_shuffle:
            np.random.shuffle(feature_data)
        data = np.zeros([self.max_len, self.feat_dim], dtype=np.float32)
        data[:feature_data.shape[0]] = feature_data
        label = self.labels[idx]
        
        return data, length, label
        
        
if __name__ == '__main__':
    pass