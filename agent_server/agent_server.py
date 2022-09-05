from multiprocessing.connection import Listener
import threading
import cv2
import time

#from agent_state import AgentState
from game_state_manager import GameStateManager

class AgentServer:

    def __init__(self, agent, host, port, classify=True):
        self.host = host
        self.port = port
        
        self.classify = classify
        self.agent = agent
        self.state_manager = GameStateManager(self.agent, classify)

        self.server_thread = None
        self.stop_server = False
        self.conn = None
        

    def start(self):
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.start()

    def start_server(self):
        address = (self.host, self.port)
        self.server = Listener(address)
        self.conn = self.server.accept()
        print("Connection accepted from", self.conn)

        while not self.stop_server:
            #print(self.conn.recv())
            if self.conn.poll():
                #print(self.conn.recv())
                self.handle_connection(self.conn)
            #if not run:
            #    break
    def stop_server(self):
        self.stop_server = True
        self.server_thread.join()
        self.conn.close()
        self.server.close()

    def handle_connection(self, conn):
        msg = conn.recv()
        #print(msg)
        if msg == 'close':
            conn.close()
            return False
        else:
            state_thread = threading.Thread(target=self.state_manager.update_state, args=(msg,))
            state_thread.start()
            #self.state_manager.update_state(msg)
            if self.agent.frame is not None:
                cv2.imshow("frame", self.agent.frame)
                cv2.waitKey(1)
            state_thread.join()
            return True
 
    def send_input(self, input):
        if self.conn is not None:
            self.conn.send(input)

if __name__ == '__main__': 
    server = AgentServer(host="127.0.0.1", port=6000)
    server.start()