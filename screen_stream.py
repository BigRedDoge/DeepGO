import d3dshot
import threading
import cv2
import time


class ScreenStream():

    def __init__(self, width, height, top=0, left=0, frame_rate=30):
        self.width = width
        self.height = height
        self.top = top
        self.left = left
        self.frame_rate = frame_rate

        self.stop_stream = False
        self.frame_lock = threading.Lock()

        self.frame = None
        self.stream = None
        self.stream_thread = None

    def start(self):
        self.stream_thread = threading.Thread(target=self.start_stream)
        self.stream_thread.start()
        print("Stream started")

    def start_stream(self):
        self.stream = d3dshot.create(capture_output="numpy")

        with self.frame_lock:
            while not self.stop_stream:
                self.frame = self.stream.screenshot()

                if self.frame is None:
                    break

    def stop(self):
        self.stop_stream = True
        self.stream_thread.join()
        self.stream.stop()
        
        print("Stream stopped")


if __name__ == '__main__':
    stream = ScreenStream(width=3024, height=1964, top=0, left=0, frame_rate=2, record_path="test.mp4")
    stream.start()
    time.sleep(10)
    stream.stop()