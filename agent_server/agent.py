class Agent:

    def __init__(self):
        self.game_state = None
        self.frame = None
        self.classification = None
        self.map = None

    def get_classification(self):
        return self.classification

    def get_game_state(self):
        return self.game_state

    def get_frame(self):
        return self.frame

    def get_map(self):
        return self.map