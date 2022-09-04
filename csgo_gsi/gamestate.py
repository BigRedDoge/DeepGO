from csgo_gsi.information import *

class GameState:
    def __init__(self):
        self.player = Player()
        self.map = Map()
        self.provider = Provider()
        self.phase_countdowns = PhaseCountdowns()
        self.bomb = Bomb()
        self.round = Round()
