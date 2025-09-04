import socket

class GPU_Client():
    def __init__(self, host_ip, port) -> None:
        self.host_ip = host_ip
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.connect((self.host_ip, self.port))  # 只連一次

    def send_msg(self, msg: str) -> None:
        try:
            self.s.sendall(msg.encode())
        except (BrokenPipeError, ConnectionResetError):
            # 連線斷掉時，可選擇重新 connect
            self._reconnect()
            self.s.sendall(msg.encode())

    def wait_msg(self)->str:
        try:
            data = self.s.recv(8192)
            print('finsih')
            return data.decode()
        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected, retrying receive...")
            self._reconnect()
            
    def _reconnect(self):
        self.s.close()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.connect((self.host_ip, self.port))

if __name__ == "__main__":
    client = GPU_Client('127.0.0.1', 12345)
    client_recv = GPU_Client('127.0.0.1', 4478)
    text = input('請輸入: ')
    text = '佳勳' + '@@' + text
    client.send_msg(text)
    print("wait for msg")
    r = client_recv.wait_msg()
    print('接收: ', r)
