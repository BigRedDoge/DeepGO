import sys
import time
sys.path.append("agent_server")

from agent_server.agent_state import AgentState
from agent_server.agent_server import AgentServer
from agent_client.agent_client import AgentClient

class Agent:

    def __init__(self, classify=True):
        self.host = "127.0.0.1"
        self.port = 6000

        self.agent = AgentState()
        self.server = AgentServer(self.agent, self.host, self.port, classify)
        #self.client = AgentClient(host, port)

    def start_connection(self):
        self.server.start()
        time.sleep(1)
        self.client = AgentClient(self.host, self.port)
        self.client.start()

    def send_input(self, input):
        if self.server is not None:
            self.server.send_input(input)

    def get_state(self):
        return self.agent


if __name__ == '__main__':
    agent = Agent()
    agent.start_connection()
    while True:
        #print(agent.get_agent_state().get_frame())
        #print(agent.get_agent_state().get_game_state())
        #print(agent.get_agent_state().get_classification())
        #print(agent.get_agent_state().get_map())
        agent.send_input("test")
        time.sleep(1)