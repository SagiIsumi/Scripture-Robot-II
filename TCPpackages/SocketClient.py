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
    client = SocketClient('140.112.14.225', 2468)
    text = input('請輸入: ')
    text = '佳勳' + '@@' + text
    client.send_msg(text)
    print("wait for msg")
    r = client.wait_msg()
    print('接收: ', r)
