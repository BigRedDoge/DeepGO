from multiprocessing.connection import Listener
import threading


class AgentServer:

    def __init__(self, host, port):
        self.host = host
        self.port = port
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
            run = self.handle_connection(self.conn)
            if not run:
                break

    def handle_connection(self, conn):
        msg = conn.recv()
        print(msg)

        if msg == 'close':
            conn.close()
            return False
        return True