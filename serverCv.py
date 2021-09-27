import socket
import sys
import cv2
import pickle
import numpy as np
import struct 
import chaosMap

soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # Create a socket object
HOST = "127.0.0.1" # server ip
PORT = 5001 # Reserve a port for server.

soc.bind((HOST, PORT)) # Bind to the port
soc.listen(5) # Now wait for client connection.
print(f"[*] Listening as {HOST}:{PORT}")
# receive 4096 bytes each time

conn, addr = soc.accept() # Establish connection with client.
print(f"[+] Got a connection from {addr[0]}:{addr[1]}")

data = b""
payload_size = struct.calcsize(">L")

print("[!] payload_size: {}".format(payload_size))
while len(data) < payload_size:
    print("Recv: {}".format(len(data)))
    data += conn.recv(4096)

print("[!] Done Recv: {}".format(len(data)))
packed_msg_size = data[:payload_size]
data = data[payload_size:]
msg_size = struct.unpack(">L", packed_msg_size)[0]
print("[!] msg_size: {}".format(msg_size))


while len(data) < msg_size:
    data += conn.recv(4096)

img_data = data[:msg_size]
data = data[msg_size:]
img = pickle.loads(img_data, fix_imports=True, encoding="bytes")
#img = cv2.imdecode(img, cv2.IMREAD_COLOR)
key = 20#(0.1,0.1)
img = chaosMap.chaosDecryption(img,key,0)
cv2.imwrite('download.png',img)

soc.close()