import os
from re import sub
# import skvideo.io
from PIL import Image
import gc
import pandas as pd
from multiprocessing import Process
from pathlib import Path
import subprocess
# from yuv_frame_io import YUV_Read
# def iframes_ffmpeg(filename, save_path, i_frames):
#     for frame_no in i_frames:
#         outname = os.path.join(save_path, str(frame_no + 1).zfill(4) + '.png')
#         os.system('ffmpeg -i {} -video_size {} -pix_fmt yuv420p10le {}  '.format(filename, '3840x2160', outname))

def get_frames(video_names, frame_folder, video_folder):
    # pbar= tqdm(video_names)
    num_videos = len(video_names)
    
    for idx, video_name in enumerate(video_names):
        # pbar.set_description("Processing [%s] " % video_name)
        print(f"Processing [{idx+1}/{num_videos} {video_name}]")
        # video_name: Artifacts/0723_ManUnderTree_GS5_03_20150723_130016.yuv
        # save_folder: D:\datasets\VQAFrames\LIVEQualcomm\0723_ManUnderTree_GS5_03_20150723_130016\
        saved_name = os.path.join(video_name.split('/')[1], video_name.split('/')[2])
        save_folder = os.path.join(frame_folder, saved_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # video_data (ndarray): shape[lenght, H, W, C]
        video_name_mp4 = os.path.join(video_folder, video_name+'.mp4')
        # yuv_reader = YUV_Read(video_name_yuv, h=2160, w=3840, toRGB=True)
        # video_data, _=yuv_reader.read()
        os.system('ffmpeg -i {} -qscale:v 2 {}/output-%04d.png'.format(video_name_mp4, save_folder))
    
        # video_length = video_data.shape[0]
        # for frame_idx in frames:
        #     outname = os.path.join(save_folder, "{0:04d}".format(frame_idx)) + '.png'
        #     os.system('ffmpeg -i {} -video_size {} -pix_fmt yuv420p10le {}'.format(video_name, '3840x2160', outname))
        # for frame_idx in range(video_length):
        #     save_name = os.path.join(save_folder, "{0:04d}".format(frame_idx)) + '.png'
        #     if os.path.isfile(save_name):
        #         continue
        #     frame = video_data[frame_idx]
        #     img = Image.fromarray(frame)
        #     # print(save_name)
        #     img.save(save_name)
        # del video_data
        # gc.collect()

def main():
    get_frames()

if __name__ == '__main__':
    # csv_file = Path('/home/zhw/vqa/code/vqatransformer/feature/data/E-LIVE/ETRI-LIVE_MOS.csv')
    video_root = Path('/data/zhw/LSVQ')
    # csv_file = '/data/zhw/LSVQ/label_train_test.csv'
    csv_file = Path('/data/zhw/LSVQ/LSVQ_all.csv')
    video_info = pd.read_csv(csv_file, header=0)
    sub_video_names = video_info.iloc[:, 1].tolist()
    video_moses = video_info.iloc[:, 2].tolist()
    # video_names = os.listdir(video_root)
    #
    video_names = [os.path.join(video_name.split('/')[0], video_name) for video_name in sub_video_names]
    # exist_frame_folder = '/data/zhw/vqa_data/ETRI-LIVE/frame/'
    frame_folder = '/data/zhw/LSVQ/frames'
    # exist_videos = os.listdir(frame_folder)


    # video_info = pd.read_csv(csv_file, header=0)
    # video_names = video_info.iloc[:, 0].tolist()
    # video_moses = video_info.iloc[:, 1].tolist()
    # height = 480
    # width = 832

    # for video in exist_videos:
    #     video_names.remove(video)
    print('===========> video length', len(video_names))

    # save_root = Path('/mnt/add_Seagate/vqa/data/ETRI-LIVE/distortion')
    # frame_folder.mkdir(exist_ok=True)
    if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)


    num_processes = 1
    processes = []
    nvideo_per_node = int(len(video_names) / num_processes)
    for rank in range(num_processes):
        # videos_ = video_names[rank*nvideo_per_node:(rank+1)*nvideo_per_node]
        video_names_ = video_names[rank*nvideo_per_node:(rank+1)*nvideo_per_node]
        p = Process(target=get_frames, args=(video_names_, frame_folder, video_root))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # main()