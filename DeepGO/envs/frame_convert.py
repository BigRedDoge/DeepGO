import cv2
import numpy as np
import copy

def frame(frame):
    try:
        if frame is not None:
            frame = copy.deepcopy(frame)
            frame = cv2.resize(frame, (256, 144))
            frame = np.asarray(frame).astype(np.uint8)  
            #cv2.imshow('DeepGO', frame)
            #cv2.waitKey(1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print("whoops")
            frame = np.zeros((256, 144, 3), dtype=np.uint8)
            return frame
    except Exception as e:
        print("whoops 2", e)
        return np.zeros((256, 144, 3), dtype=np.uint8)