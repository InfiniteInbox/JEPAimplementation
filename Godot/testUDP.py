import socket 
import json

HOST = "127.0.0.1" # Localhost
PORT = 25565 

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s: #(Aug 12)Using UDP for fast transfer 
    s.bind((HOST, PORT))

    print(f"listening on {HOST} and {PORT}")

    while True:
        data, addr = s.recvfrom(1024)
        msg = data.decode()
        print(msg)
        
