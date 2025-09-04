import socket
from time import sleep

import socket
from time import sleep

class SocketServer:
    def __init__(self, host_ip, port):
        self.host_ip = host_ip
        self.port = port
        self.conn = None
        self.addr = None
        self._init_server()

    def _init_server(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.host_ip, self.port))
        self.s.listen()
        print(f"Server listening on {self.host_ip}:{self.port}...")

    def accept_client(self):
        if self.conn is None:
            self.conn, self.addr = self.s.accept()
            print(f"Connected by {self.addr}")

    def send_msg(self, msg: str):
        for _ in range(3):  # 最多重試 3 次
            try:
                self.accept_client()
                self.conn.sendall(msg.encode())
                sleep(0.01)
                return
            except (ConnectionResetError, BrokenPipeError):
                print("Client disconnected, retrying send...")
                self.conn = None

        raise RuntimeError("Failed to send message after 3 retries")

    def wait_msg(self, delimiter=b'\n') -> str:
        for _ in range(3):  # 最多重試 3 次
            try:
                print('start')
                self.accept_client()
                data = self.conn.recv(8192)
                print('finsih')
                return data.decode()
            except (ConnectionResetError, BrokenPipeError):
                print("Client disconnected, retrying receive...")
                self.conn = None
                return ''
        raise RuntimeError("Failed to receive message after 3 retries")


if __name__ == "__main__":
    server = SocketServer(host_ip='127.0.0.1', port=12345)
    text = server.wait_msg()
    print(text)
    server.send_msg('你好')
