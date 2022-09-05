class Message:
    def __init__(self):
        self.game_state = None
        self.frame = None

    def add_gsi(self, game_state):
        self.game_state = game_state

    def add_screen(self, frame):
        self.frame = frame