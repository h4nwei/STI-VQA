import pandas as pd
import numpy as np

def generate_split(idx):
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    split_file = '/home/zhw/vqa/code/VQA-framework/feature/data/PUGC/train_val_test_split_{}.xlsx'.format(idx)
    print(split_file)
    
    database_info_file_path = '/data/zhw/PUGC/PUGC_all.csv'
    video_info = pd.read_csv(database_info_file_path, header=0)
    video_names = video_info.iloc[:, 1].tolist()    
    # sub_video_names = sub_video_names[:len(sub_video_names)//2]
    category_video_names = video_info.iloc[:, 2].tolist()
    # category_video_names = category_video_names[:len(category_video_names)//2]
    video_moses = video_info.iloc[:, 3].tolist()
    # df_info = pd.read_csv(database_info_file_path)
    # file_names = np.array([str(name) + '.mp4' for name in df_info['flickr_id'].values])
    num_videos = len(video_names)
    num_train_videos = int(train_ratio * num_videos)
    num_val_videos = int(val_ratio * num_videos)
    num_test_videos = num_videos - num_train_videos - num_val_videos
    status = np.array(['train'] * num_train_videos + ['validation'] * num_val_videos + ['test'] * num_test_videos)
    np.random.shuffle(status)
    
    split_info = np.array([video_names, category_video_names, video_moses, status]).T
    df_split_info = pd.DataFrame(split_info, columns=['video_name', 'category_video_names', 'mos', 'status'])
    # print(df_split_info.head())
    df_split_info.to_excel(split_file)

def main():
    for idx in range(10):
        generate_split(idx)

if __name__ == '__main__':
    main()