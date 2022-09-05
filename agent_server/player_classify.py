#from Sequoia.detect import detect
#import sys
#sys.path.append(r"C:\Users\Sean\Documents\CS-AI\Sequoia")

import imp
import cv2
import torch
import numpy as np
from os import path
from time import time
import torch.backends.cudnn as cudnn
from numpy import asarray, random, reshape, swapaxes
#from strektref import set_pos
import copy
from PIL import Image
import threading


from Sequoia.light_inference import light_run
from Sequoia.light_inference import load_light_weights
import Sequoia.models.experimental as experimental
from Sequoia.utils.general import (apply_classifier, check_img_size,
                           non_max_suppression, plot_one_box, scale_coords,
                           set_logging, strip_optimizer, xyxy2xywh)

class PlayerClassify:

    def __init__(self, window_x=2560, window_y=1440, conf_threshold=0.2, view_img=True, benchmark=False):
        self.window_x = window_x
        self.window_y = window_y
        self.conf_threshold = conf_threshold
        self.view_img = view_img
        self.benchmark = benchmark

        self.yolo_weights_path = "agent_server\sequoiaV1.pt"
        self.light_weights_path = "agent_server\light_classifier_v1.th"

        self.use_light = True

    def classify(self, frame):
        bboxes = self.detect(frame)
        return bboxes

    def detect(self, frame):
        # Initialize
        device = torch.device("cuda:0")
        #print("detecting on: %s"%(torch.cuda.get_device_name(device)))

        # Load model
        model = experimental.attempt_load(self.yolo_weights_path, map_location=device)
        load_light_weights(self.light_weights_path)

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        #print(frame)
        #frame = copy.deepcopy(frame)
        img = frame.resize((512, 512))
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        im0 = img #save raw image for later
        ## preparing img for torch inference
        img = swapaxes(img, 0, 2)
        img = swapaxes(img, 1, 2)
        img = img.reshape(1, 3, 512, 512)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Inference
        tic_yolo = time()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred) 
        toc_yolo = (time() - tic_yolo)*1000
        if self.benchmark:
            print(f"yolo: {toc_yolo} ms")

        # Process detections
        bboxes = []
        for i, det in enumerate(pred):  # detections per image
            p, s = "teste", ""

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.view_img and conf >= self.conf_threshold:  # Add bbox to image

                        label = '%s %.2f' % (names[int(cls)], conf)
                        bbox = [coord.item() for coord in xyxy]
                        bboxes.append(bbox)

                        ## get predictions for light_classifier
                        tic_light = time()
                        light_pred = light_run(im0, bbox).item()
                        toc_light = (time() - tic_light)*1000
                        if self.benchmark:
                            print(f"light: {toc_light} ms")

                        # plot the bboxes on image
                        if self.use_light:
                            if light_pred >= 0.5: ct_tr_light = "ct"
                            else: ct_tr_light = "tr"

                            label_light = f"{ct_tr_light}, {light_pred:3f}"
                            plot_one_box(xyxy, im0, label=label_light, color=colors[int(cls)], line_thickness=2)
                        else:
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                        bboxes.append(bbox)

                        if self.use_light:
                            if label[:2] != "ct" and label[:2] != "tr":
                                cv2.putText(im0,"nothing", (10,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            else:
                                if light_pred >= 0.5:
                                    ct_tr_light = "ct"
                                else: ct_tr_light = "tr"
                                cv2.putText(im0,f"yolo:{label[:2]}, light:{light_pred:1f} ({ct_tr_light})", (10,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    if self.use_light:
                        break #this break ensures only one bbox will be showed per viewport render (inference)
            # Stream results
            if self.view_img:
                #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                im0 = cv2.resize(im0, (1280,720))
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

        return bboxes