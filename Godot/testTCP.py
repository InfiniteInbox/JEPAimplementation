import socket
import json

HOST = "127.0.0.1"  
PORT = 25565

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"Server listening on {HOST}:{PORT}...")

try:
    conn, addr = server.accept()
    print(f"Connected by Godot client at: {addr}")
    
    while True:
        # Receive raw data up to 1024 bytes
        data = conn.recv(1024)
        message_str = data.decode("utf-8")
        print(f"Received from Godot: {message_str}")
        message = "Hello, Server!"
        encoded_data = message.encode("utf-8")
        conn.sendall(encoded_data)
        print(f"[CLIENT] Sent: {message}")
except:
    raise ValueError('e')
