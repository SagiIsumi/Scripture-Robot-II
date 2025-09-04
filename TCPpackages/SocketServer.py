import socket
from time import sleep

class SocketServer():
    def __init__(self, host_ip, port) -> None:
        self.host_ip = host_ip
        self.port = port

    def send_msg(self, msg:str)->None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host_ip, self.port))
            print(f"Listening on {self.host_ip}:{self.port}...")
            s.listen()
            conn, addr = s.accept()
            conn.sendall(msg.encode())
            s.shutdown(socket.SHUT_RDWR)
            sleep(1)

    def wait_msg(self)->str:
            print('wait')
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host_ip, self.port))
                print(f"Listening on {self.host_ip}:{self.port}...")
                s.listen()
                conn, addr = s.accept()
                while True:
                    data = conn.recv(1024)
                    if data:
                        s.shutdown(socket.SHUT_RDWR)
                        sleep(0.05)
                        return data.decode()
                    else:
                        continue

if __name__ == "__main__":
    server = SocketServer(host_ip='127.0.0.1', port=8080)
    text = server.wait_msg()
    print(text)
    server.send_msg('你好')
