# Pre-trained parameters 
We take the backbone used in [LinearityIQA](https://github.com/lidq92/LinearityIQA) as our spatial feature extractor. Thus, you need to download the paramerter at the first. The p1q2plus0.1variant.pth can be downloaded at [Baidu](https://pan.baidu.com/share/init?surl=MRamimHWX8F-SOQ_QsIrvg) (Code:4z7z) and [Google_drive](4z7z).

# Feature extractor

1. Set the path to the source videos in each dataset. 
2. Set the path to save the extracted features.
3. Set the path to csv file of each dataset, which contains all the video names and MOSs.
4. You can also set other parameters according to your computing resource, such as the frame_batch_size and num_processes.

```
python CNN_feature_extractor_video.py 
```