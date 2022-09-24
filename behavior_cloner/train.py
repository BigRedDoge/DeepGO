from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import cv2
import numpy as np
import os

from model.network import CSGONet


def get_frames_from_mp4(path):
    """
    :param path: path to mp4 file
    :return: a list of single frames shaped CxWxH
    """
    cap = cv2.VideoCapture(path)
    success, image = cap.read()
    frames = []

    while success:
        frame = cv2.resize(image, (256, 144))
        frames.append(frame)
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
            frame = torch.from_numpy(frames[e])
            #print("frame shape ", frame.shape)
            frame = frame.permute(2, 0, 1)
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
    #if detect_bad > 1.1 or detect_bad < 0.97:
    #    print(detect_bad, sample["frame_path"], "is bad! skipping it")
    #    return
    # cut if there are different numbers of filtered_frames&labels
    min_num = min(len(frames), len(labels))
    frames = frames[:min_num]
    # to shape: batch steps channel width height
    frames = produce_older_steps(frames, 16)
    frames = torch.stack(frames)
    inputs = frames  # shape : batch, channel, width, height
    labels = labels[:min_num]
    labels = torch.stack([each[1] for each in labels])
    print("frames shape", frames.shape)
    return inputs, labels


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
            #if self.size > 2:
            #    break
        # preload data in ram
        mp4_path = ['./dataset' + self.names_list[idx].split(' ')[0] for idx in range(self.size)]
        label_path = ['./dataset' + self.names_list[idx].split(' ')[1] for idx in range(self.size)]
        multiple_results = map(exec_load_job, [i for i in zip(mp4_path, label_path)])   #[pool.apply_async(func=exec_load_job, args = (i,)) for i in zip(mp4_path, label_path)]  # 异步开启4个进程（同时而不是依次等待）
        # outs = [res for res in multiple_results]  # 接收4个进程的输出结果 (inputs, labels)
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
        return self.size

    def __getitem__(self, idx):
        return self.out_ram[idx]

# Create training function
def train(load=False):
    model = CSGONet().half()
    if load:
        model.load_state_dict(torch.load(load))
    
    model = torch.nn.DataParallel(model.cuda() if torch.cuda.is_available() else model)

    batch_size = 16
    epochs = 100

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(),  eps=0.0001) #  lr=lr,   # for nan, see : https://discuss.pytorch.org/t/adam-half-precision-nans/1765/9
    dataset = CSGODataset(root_dir="./dataset/", meta_path="./dataset/meta.csv", transform=None)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=SequentialSampler(dataset),
                                num_workers=0, pin_memory=False)

    for epoch in range(epochs):
        print("epoch ", epoch)
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)

            w,a,s,d,fire,scope,jump,crouch,walking,reload,e,switch,aim_x,aim_y = outputs
            l_w, l_a, l_s, l_d, l_fire, l_scope, l_jump = labels[:,[0]].squeeze(),labels[:,[1]].squeeze(),labels[:,[2]].squeeze(),labels[:,[3]].squeeze(),labels[:,[4]].squeeze(),labels[:,[5]].squeeze(),labels[:,[6]].squeeze()
            l_crouch, l_walking, l_reload, l_e, l_switch, l_aim_x, l_aim_y = labels[:,[7]].squeeze(),labels[:,[8]].squeeze(),labels[:,[9]].squeeze(),labels[:,[10]].squeeze(),labels[:,[11]].squeeze(),labels[:,[12]].squeeze(),labels[:,[13]].squeeze()
            # define loss
            loss_w_v, loss_a_v, loss_s_v, loss_d_v = criterion(w,l_w),criterion(a,l_a),criterion(s,l_s),criterion(d,l_d)
            #wprint("aim_x ",aim_x[0], "aimx_label", l_aim_x[0] )
            #print("w", w[0], "l_w", l_w[0])
            loss_fire_v, loss_scope_v, loss_jump_v, loss_crouch_v, loss_walking_v, loss_reload_v, loss_e_v, loss_switch_v = criterion(fire,l_fire),criterion(scope,l_scope),criterion(jump,l_jump),criterion(crouch,l_crouch),criterion(walking,l_walking),criterion(reload,l_reload),criterion(e,l_e),criterion(switch,l_switch)
            loss_aim_x_v, loss_aim_y_v = criterion(aim_x, l_aim_x), criterion(aim_y,l_aim_y)
            #  print("aim_y.shape, l_aim_y.shape ", aim_y.shape, l_aim_y.shape)
            #  torch.Size([143, 33]) torchSize([143])
            loss = loss_w_v + loss_a_v + loss_s_v + loss_d_v + 50*loss_fire_v + loss_scope_v + loss_jump_v + loss_crouch_v + loss_walking_v + loss_reload_v + loss_e_v + loss_switch_v + 10*loss_aim_x_v + 10*loss_aim_y_v
            loss_list = [loss_w_v.item(),loss_a_v.item(), loss_s_v.item(), loss_d_v.item(), loss_fire_v.item(), loss_scope_v.item(), loss_jump_v.item(), loss_crouch_v.item(), loss_walking_v.item(), loss_reload_v.item(), loss_e_v.item(), loss_switch_v.item(), loss_aim_x_v.item(), loss_aim_y_v.item()]
            loss_value = sum(loss_list)
            print("loss_value ", loss_value)
            
            running_loss += loss_value
            print("running_loss ", running_loss)

            loss.backward()
            optimizer.step()
            print("loss ", loss.item())
            if i % 10 == 0:
                torch.save(model.state_dict(), "model.pth")
                print("model saved")
            

if __name__ == "__main__":
    train(load=False)