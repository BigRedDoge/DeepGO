from ..csgo_gsi.server import GSIServer
from screen_stream import ScreenStream
from multiprocessing.connection import Client
import threading
import time


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
            data = Message()
            data.add_gsi(self.gsi_server.get_info())
            data.add_screen(self.screen_stream.get_frame())
            # Get info from GSI server
            if data:
                self.conn.send(data)
                #self.conn.close()


    def stop(self):
        self.stop_client = True
        self.client_thread.join()
        #self.conn.send("close")
        #self.conn.close()
        self.screen_stream.stop()
        self.gsi_server.stop_server()

class Message:
    def __init__(self):
        self.gamestate = None
        self.frame = None

    def add_gsi(self, gamestate):
        self.gamestate = gamestate

    def add_screen(self, frame):
        self.frame = frame

if __name__ == '__main__':
    client = AgentClient()
    client.start()
    time.sleep(5)
    client.stop()


