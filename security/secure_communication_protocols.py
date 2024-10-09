import socket
import ssl

class SecureCommunicationProtocols:
    def __init__(self):
        self.context = ssl.create_default_context()
        self.context.load_cert_chain("cert.pem", "key.pem")

    def create_secure_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ssl_sock = self.context.wrap_socket(sock, server_side=False)
        return ssl_sock

    def send_secure_data(self, ssl_sock, data):
        ssl_sock.sendall(data.encode())

    def receive_secure_data(self, ssl_sock):
        data = ssl_sock.recv(1024)
        return data.decode()
