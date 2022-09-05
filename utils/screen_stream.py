#import d3dshot
import threading
import time
import numpy as np
from mss import mss
from PIL import Image
import cv2

class ScreenStream():

    """
    capture output can be either "numpy" or "pil"
    """
    def __init__(self, window_x=2560, window_y=1440, capture_output="numpy"):
        if capture_output not in ["numpy", "pil"]:
            raise ValueError("capture_output must be either 'numpy' or 'pil'")
        self.capture_output = capture_output
        
        self.stop_stream = False
        self.frame_lock = threading.Lock()

        self.frame = None
        self.stream = None
        self.stream_thread = None

        self.window_x = window_x
        self.window_y = window_y

    def start(self):
        self.stream_thread = threading.Thread(target=self.start_stream)
        self.stream_thread.start()
        print("Stream started")

    def start_stream(self):
        #self.stream = d3dshot.create(capture_output=self.capture_output)
        monitor = {"top": 0, "left": 0, "width": self.window_x, "height": self.window_y}
        #with self.frame_lock:
        while not self.stop_stream:
            with mss() as sct:
                # 1280 windowed mode for CS:GO, at the top left position of your main screen.
                # 26 px accounts for title bar. 
                img = sct.grab(monitor)
                #create PIL image
                img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                #imgarr = np.asarray(img)
                #imgarr = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
                self.frame = img

    def stop(self):
        self.stop_stream = True
        #self.stream.stop()
        
        print("Stream stopped")

    def get_frame(self):
        with self.frame_lock:
            return self.frame

if __name__ == '__main__':
    stream = ScreenStream()
    stream.start()
    import cv2
    for i in range(340):
        frame = stream.get_frame()
        if frame is not None:
            try:
                cv2.imshow("frame", np.array(frame))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass
        time.sleep(1/60)
    stream.stop()