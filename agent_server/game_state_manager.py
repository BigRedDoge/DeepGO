from player_classify import PlayerClassify
from map import Map 


class GameStateManager:

    def __init__(self, agent):
        self.agent = agent
        self.player_classify = PlayerClassify()
        self.map = Map()

    def update_state(self, state):
        self.agent.game_state = state.game_state
        self.agent.frame = state.frame
        self.agent.classification = self.player_classify.classify(state.frame)
        self.agent.map = self.map.get_map(state.frame)

