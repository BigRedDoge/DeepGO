from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn as nn
import torch
import torchvision
import cv2
import numpy as np
import os


def get_frames_from_mp4(path):
    """
    :param path: path to mp4 file
    :return: a list of single frames shaped CxWxH
    """
    cap = cv2.VideoCapture(path)
    success, image = cap.read()
    frames = []

    while success:
        frames.append(image)
        success, image = cap.read()

    return frames


def produce_older_steps(frames, time_steps):
    """
    :param frames: a list of single frames shaped CxWxH
    :return: a list of time series frames shaped TxCxWxH
    """
    frames_with_old = []
    for i, frame in enumerate(frames):
        frame_with_old = frames[:i+1][-time_steps:]
        # if no enough frames before, we padding it with zero frames
        if len(frame_with_old) < time_steps:
            frame_with_old = [torch.zeros(frame.size()) for e in range(time_steps-len(frame_with_old))] + frame_with_old
        frame_with_old = torch.stack(frame_with_old)
        frames_with_old.append(frame_with_old)
    return frames_with_old


def get_label_by_name(path):
    """
    :param path: path to label file
    :return: [tick , tensor([ ])]
    """
    li = []
    path = path.strip("\n")
    with open(path, "r") as f:
        header = f.readline()
        for line in f.readlines():
            acts = line.split("\t")[1:]
            acts = [int(each) for each in acts]
            tick = acts[0]
            none_aim_labels = acts[1:5] + acts[7:]
            li.append([tick, torch.tensor(none_aim_labels + [acts[5], acts[6]])])
        return li

def exec_load_job(li):
    mp4_path, label_path = li
    frames = get_frames_from_mp4(mp4_path)
    filter_frames = []
    # there are 32fps videos, we need 16 frames per sec only, filter them
    for e in range(len(frames)):
        if e % 2 == 0 and frames[e] is not None:
            #frame = tfms(frames[e])
            # frame = torch.from_numpy(frame)
            frame = frame.permute(0, 2, 1)
            filter_frames.append(frame)
    if not filter_frames:
        print("null ", mp4_path)
        return
    print("shape of each frame ", filter_frames[-1].shape, len(filter_frames))
    frames = filter_frames
    # label_path = names_list[idx].split(' ')[1]
    labels = get_label_by_name(label_path)
    sample = {'frames': frames, 'labels': labels, "frame_path": mp4_path, "label_path": label_path}
    ## process to cuda
    # 1sec = 32frames = 16 filtered frames =  16acitons; thus, filter abnormal data
    frames, labels = sample['frames'], sample['labels']
    detect_bad = len(sample['frames']) / len(sample['labels'])
    if detect_bad > 1.1 or detect_bad < 0.97:
        print(detect_bad, sample["frame_path"], "is bad! skipping it")
        return
    # cut if there are different numbers of filtered_frames&labels
    min_num = min(len(frames), len(labels))
    frames = frames[:min_num]
    # to shape: batch steps channel width height
    frames = produce_older_steps(frames, 16)
    frames = torch.stack(frames)
    inputs = frames  # shape : batch, channel, width, height
    labels = labels[:min_num]
    labels_tensor = torch.stack([each[1] for each in labels])
    print("frames shape", frames.shape)
    return inputs, labels_tensor


# Create pytorch Dataset class 
class CSGODataset(Dataset):
    def __init__(self, root_dir, meta_path, transform=None):
        self.root_dir = root_dir  # 数据集的根目录
        self.meta_path = meta_path  # meta.csv 的路径
        self.transform = transform  # 预处理函数
        self.size = 0           # 视频个数
        self.names_list = []  # 用来存放meta.csv中的文件名的
        self.data_ram = []
        self.out_ram = []
        # reading from meta
        if not os.path.isfile(self.meta_path):
            print("meta.csv NOT FOUND")
            print(self.meta_path + 'does not exist!')
        print("meta.csv FOUND")
        file = open(self.meta_path)
        for line in file:
            #print("reading lines")
            self.names_list.append(line)
            self.size+=1
            if self.size > 5:
                break
        # preload data in ram
        mp4_path = [self.names_list[idx].split(' ')[0] for idx in range(self.size)]
        label_path = [self.names_list[idx].split(' ')[1] for idx in range(self.size)]
        multiple_results = map(exec_load_job, [i for i in zip(mp4_path, label_path)])   #[pool.apply_async(func=exec_load_job, args = (i,)) for i in zip(mp4_path, label_path)]  # 异步开启4个进程（同时而不是依次等待）
        # outs = [res for res in multiple_results]  # 接收4个进程的输出结果 (inputs, labels_tensor)
        # data_ram content = (video_frames, labels), each video_frames has 406 frames, each frame have 16 subframes(previous)
        self.data_ram = [(each[0].half(), each[1]) for each in multiple_results if each is not None]
        # change data_ram content to = (previous_frames, label)
        for each_video in self.data_ram:
            frames, labels = each_video  # frames = [subframes_1, sub_frames_2, ..., sub_frames_406]
            for frame, label in zip(frames, labels):
                self.out_ram.append((frame, label))
        self.data_ram = []
        self.size = len(self.out_ram)
        print("load complete")

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

# Create training function
