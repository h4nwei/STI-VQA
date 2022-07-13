import pandas as pd
import os
from PIL import Image
import json


def main():
    database_name = 'VQC.json'
    database_info = pd.read_csv(r'C:\freetime\code\VQA\revisit-vqa\data\VQC\VQC.csv')
    video_frames_folder = r'D:\Video_Dataset\VQA_frames\VQC'
    format = 'mp4'
    distorted_type = 'wild'
    
    
    video_name_list = database_info['video_name'].tolist()
    video_label_list = database_info['MOS'].tolist()
    
    
    database_info = {}
    
    for i in range(len(video_name_list)):
        video_name = str(video_name_list[i])
        label = video_label_list[i]
        video_path = os.path.join(video_frames_folder, str(video_name))
        img_name = os.listdir(video_path)
        video_length = len(img_name)
        img_path = os.path.join(video_path, img_name[0])
        img = Image.open(img_path)
        width, height = img.size
        # print(video_name)
        # print(height, width)
        video_info = {'video_name': video_name, 'format': format,
                      'distorted_type': distorted_type,
                      'height': height,
                      'width': width,
                      'label': label}
        database_info[i] = video_info
    print(database_info)
    with open(database_name, 'w') as f:
        json.dump(database_info, f)
        
        

if __name__ == '__main__':
    main()