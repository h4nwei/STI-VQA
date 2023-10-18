import pandas as pd
import os
from PIL import Image
import json
from pathlib import Path

def main():
    # database_name = '/home/zhw/vqa/code/3D-DISITS/feature/data/CSIQ/CSIQ.json'

    split_file_path = r'/home/zhw/vqa/code/VQA-framework/data/LSVQ/'
    split_idx_file_path_list = ["/home/zhw/vqa/code/VQA-framework/feature/data/LSVQ/train_val_test_split_1080p.xlsx",
                                "/home/zhw/vqa/code/VQA-framework/feature/data/LSVQ/train_val_test_split_full.xlsx"
    ]
    database_name = 'LSVQ'

    for csv_file in split_idx_file_path_list:
        # train_val_test_split_0.xlsx
        filename = split_file_path + database_name + '_' + csv_file.split('/')[-1].split('.')[0] + '.json'
        
        print(filename)
        # video_info = pd.read_csv(csv_file)

        # video_info = pd.read_csv(csv_file, header=0)
        video_info = pd.read_excel(csv_file)
        idx_all = video_info.iloc[:, 0].values

        video_names = video_info.iloc[:, 1].tolist()
        # video_category = video_info.iloc[:, 2].tolist()
        video_mos = video_info.iloc[:, 2].tolist()

        split_status = video_info['is_test'].values
        train_idx = idx_all[split_status == 'train']
        # validation_idx = idx_all[split_status == 'validation']
        test_idx = idx_all[split_status == 'test']
        data = {'video_name_list': video_names,
        'video_label_list': video_mos,
        'train_idx': train_idx.tolist(),
        # 'val_idx': validation_idx.tolist(),
        'test_idx': test_idx.tolist()
        }
        
       
        with open(filename, 'w') as f:
            json.dump(data, f)
  
    
        
        

if __name__ == '__main__':
    main()