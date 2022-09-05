import cv2
import numpy as np
import copy

def frame(frame):
    try:
        if frame is not None:
            frame = copy.deepcopy(frame)
            frame = frame.resize((512, 512))
            frame = np.asarray(frame).astype(np.uint8)  
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.zeros((512, 512, 3), dtype=np.uint8)
            return frame
    except:
        return np.zeros((512, 512, 3), dtype=np.uint8)