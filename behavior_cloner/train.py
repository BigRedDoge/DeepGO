from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn as nn
import torch
import torchvision
import cv2
import numpy as np


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


# Create pytorch Dataset class 
class CSGODataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

# Create training function
