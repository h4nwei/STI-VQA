import scipy.io as scio
import numpy as np
import pandas as pd


def generate_database_info():
    info_path = 'data.mat'
    # data = h5py.File(info_path, 'r')
    data = scio.loadmat(info_path)
    video_names = data['video_list']
    scores = data['mos']
    
    video_name_list = []
    score_list = []
    for idx in range(len(video_names)):
        # video_name: A001.mp4 --> A001
        video_name = video_names[idx][0][0]
        video_name = video_name.split('.')[0]
        
        score = scores[idx][0]
        video_name_list.append(video_name)
        score_list.append(score)
    database_info = np.array([video_name_list, score_list]).T
    df_database_info = pd.DataFrame(database_info, columns=['video_name', 'MOS'])
    df_database_info.to_csv('VQC.csv')
        

def main():
    generate_database_info()


if __name__ == '__main__':
    main()