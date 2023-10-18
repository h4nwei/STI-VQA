import json
import os

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def make_video_dataset(video_frames_folder, dataset_info_path, n_frames):
    # with open(dataset_info_path, 'r') as f:
    #     # [{}, {}]
    #     list_info = json.load(f)
    df_database_info = pd.read_csv(dataset_info_path)
    video_names_list = df_database_info['video_name'].values
    video_labels_list = df_database_info['MOS'].values
    num_videos = len(video_names_list)
    dataset = []
    labels = []

    for i in range(num_videos):
        # video_length = list_info[i]['video_length']
        video_name = video_names_list[i]
        video_path_info = list(os.walk(os.path.join(video_frames_folder, str(video_name))))[0]
        video_frames_absolute_path = list(map(lambda x: os.path.join(video_path_info[0], x),
                                              sorted(video_path_info[2], key=lambda x: int(x.split('.')[0]))))
        video_length = len(video_frames_absolute_path)
        
        start_indexes = list(range(0, video_length, n_frames))  # e.g n_frames==4 -> [0, 4, 8, ....]
        if video_length % n_frames != 0:
            # 最后不足n_frames帧, 用视频最后n_frames帧, 这里帧可以与前n_frames帧重叠
            start_indexes[-1] = video_length - n_frames
        frames_set = []
        for j in start_indexes:
            frames_set.append(video_frames_absolute_path[j: j + n_frames])
        dataset.append(frames_set)
        labels.append(video_labels_list[i])
    
    return dataset, labels
    
    
class VideoDataset(Dataset):
    """
    Test or Validation
    """
    def __init__(self, video_frames_folder, dataset_info_path, n_frames):
        self.n_frames = n_frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # data: [
        #          [ [] , []... ],
        #          [ [] , []... ],
        #       ]
        self.data, self.label = make_video_dataset(video_frames_folder, dataset_info_path, n_frames)
    
    def __getitem__(self, index):
        batch_frames_path = self.data[index]
        len_batch_frames_path = len(batch_frames_path)
        label = self.label[index]
        # std = self.stds[index]
        
        img_path = batch_frames_path[0][0]
        img = Image.open(img_path)
        width, height = img.size
        
        frames = torch.zeros([len_batch_frames_path, self.n_frames, 3, height, width])
        for i in range(len_batch_frames_path):
            for j in range(self.n_frames):
                img = Image.open(batch_frames_path[i][j])
                img = self.transform(img)
                frames[i, j] = img
        frames = frames.permute(0, 2, 1, 3, 4)
        return frames, label
        
    def __len__(self):
        return len(self.data)

def t_make_dataset(video_frames_folder, database_info_path, n_frames):
    data_path, labels = make_video_dataset(video_frames_folder, database_info_path, n_frames)
    len_data = len(data_path)
    for i in range(len_data):
        # print(data_path[i])
        # i = 5
        path = data_path[i]
        for p in path:
            print(p)
        print(labels[i])
        print(type(labels[i]))
        break


def t_VideoDataset(video_frames_folder, database_info_path, n_frames):
    dataset = VideoDataset(video_frames_folder, database_info_path, n_frames)
    for data, label in dataset:
        print(data.shape)
        print(label)
        break

def main():
    database_info_path = r'KoNViD_1k/KoNViD_1k.csv'
    video_frames_folder = r'd:\Video_Dataset\VQA_frames\KoNViD_1k'
    n_frames = 16
    
    # t_make_dataset(video_frames_folder, database_info_path, n_frames)
    t_VideoDataset(video_frames_folder, database_info_path, n_frames)
    
    


if __name__ == '__main__':
    main()