from csgo_gsi.server import GSIServer
from utils.screen_stream import ScreenStream
from utils.message import Message
from multiprocessing.connection import Client
import threading
import time
import copy


class AgentClient:

    def __init__(self, host, port):
        # Start GSI server
        self.gsi_server = GSIServer(("127.0.0.1", 3000), "CSAI")
        self.gsi_server.start_server()

        # Start screen stream
        self.screen_stream = ScreenStream()
        self.screen_stream.start()

        # Socket info
        address = (host, port)
        self.client_thread = None

        self.conn = Client(address)
        self.stop_client = False

    def start(self):
        self.client_thread = threading.Thread(target=self.start_client)
        self.client_thread.start()

    def start_client(self):
        while not self.stop_client:
            try:
                data = Message()
                
                data.add_gsi(self.gsi_server.get_info())
                data.add_screen(self.screen_stream.get_frame())
                #print(data.game_state)
                # Get info from GSI server
                #if data.gamestate is not None and data.frame is not None:
                self.conn.send(data)
                
                if self.conn.poll():
                    self.handle_input(self.conn.recv())
                    #self.conn.close()
            except Exception as e:
                print(e)
                self.stop_client = True
                break

    def handle_input(sefl, input):
        print(input)

    def stop(self):
        self.stop_client = True
        self.client_thread.join()
        self.conn.send("close")
        self.conn.close()
        self.screen_stream.stop()
        self.gsi_server.stop_server()


if __name__ == '__main__':
    client = AgentClient("127.0.0.1", 6000)
    client.start()
    #time.sleep(15)
    #client.stop()


