import socket

class SocketClient():
    def __init__(self, host_ip, port) -> None:
        self.host_ip = host_ip
        self.port = port

    def send_msg(self, msg:str)->None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host_ip, self.port))
            s.sendall(msg.encode())

    def wait_msg(self)->str:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.host_ip, self.port))
                    data = s.recv(1024)
                    if data:
                        return data.decode()
                    else:
                        continue
            except:
                continue

if __name__ == "__main__":
    client = SocketClient('127.0.0.1', 12345)
    client_recv = SocketClient('127.0.0.1', 4478)
    text = input('請輸入: ')
    text = '佳勳' + '@@' + text
    client.send_msg(text)
    print("wait for msg")
    r = client_recv.wait_msg()
    print('接收: ', r)
