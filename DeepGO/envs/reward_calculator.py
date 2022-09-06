


class RewardCalculator:

    def __init__(self):
        pass

    def calculate_reward(self, state, previous_state):
        current_state = state.get_game_state()
        prev_state = previous_state.get_game_state()
        
        if None not in [current_state, prev_state]:
            #print(state.get_game_state().player.match_stats)
            current_kills = current_state.player.match_stats['kills']
            prev_kills = prev_state.player.match_stats['kills']
            if current_kills > prev_kills:
                return 1
        return 0