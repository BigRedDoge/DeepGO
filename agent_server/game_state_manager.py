from player_classify import PlayerClassify
from map import Map
import numpy as np
import cv2
import threading
import time

class GameStateManager:

    def __init__(self, agent, classify=True):
        # ms to delay classification
        self.CLASSIFY_DELAY = 100
        self.agent = agent
        self.classify = classify
        if self.classify:
            self.player_classify = PlayerClassify()
        self.map = Map()

        self.classify_thread = None
        self.last_frame_time = int(time.time() * 1000)

    def update_state(self, state):
        time_classify = int(time.time() * 1000)
        if state.frame is not None:
            if self.classify and (time_classify - self.last_frame_time) > self.CLASSIFY_DELAY:
                self.agent.classification = self.player_classify.classify(state.frame)
                #self.classify_thread = threading.Thread(target=self.player_classify.classify, args=(state.frame,))
                #self.classify_thread.start()
                self.last_frame_time = time_classify

            frame = np.asarray(state.frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.agent.game_state = state.game_state
            self.agent.frame = frame
            self.agent.map = self.map.get_map(frame)

            #if self.classify_thread is not None:
            #    self.classify_thread.join()

    def classify_threaded(self, frame):
        self.agent.classification = self.player_classify.classify(frame)