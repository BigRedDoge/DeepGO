import d3dshot
import threading
import time
import numpy as np


class ScreenStream():

    """
    capture output can be either "numpy" or "pil"
    """
    def __init__(self, capture_output="numpy"):
        if capture_output not in ["numpy", "pil"]:
            raise ValueError("capture_output must be either 'numpy' or 'pil'")
        self.capture_output = capture_output
        
        self.stop_stream = False
        self.frame_lock = threading.Lock()

        self.frame = d3dshot.create(capture_output=self.capture_output)
        self.stream = None
        self.stream_thread = None

    def start(self):
        self.stream_thread = threading.Thread(target=self.start_stream)
        self.stream_thread.start()
        print("Stream started")

    def start_stream(self):
        self.stream = d3dshot.create(capture_output=self.capture_output)

        #with self.frame_lock:
        while not self.stop_stream:
            self.frame = self.stream.screenshot()
            if self.frame is None:
                break

    def stop(self):
        self.stop_stream = True
        self.stream.stop()
        
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