import socket
import cv2
import pickle
import numpy as np
import struct 
import json
import chaosMap

print("[*] Creating socket")
soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # Create a socket object
HOST = "127.0.0.1" # server ip
PORT = 5001 # Reserve a port for server.
print("[+] Socket created")

soc.bind((HOST, PORT)) # Bind to the port
soc.listen(5) # Now wait for client connection.
print(f"[*] Listening as {HOST}:{PORT}")
# receive 4096 bytes each time

conn, addr = soc.accept() # Establish connection with client.
print(f"[+] Got a connection from {addr[0]}:{addr[1]}")

data = b"" # Initialize data
payload_size = struct.calcsize(">L") # Payload size

dataRecv = conn.recv(1024).decode() # recieve encryption details
print("[+] Received data: ", dataRecv)
dataRecv = json.loads(dataRecv) # convert to json
key = dataRecv['key'] # get key
type_ = dataRecv['type'] # get type
print("[+] Recieved decryption details from client")

print("[!] payload_size: {}".format(payload_size))
while len(data) < payload_size: # Get data of payload_size from client
    print("[/] Downloading : {}".format(len(data)))
    data += conn.recv(4096)

print("[!] Done Recv: {}".format(len(data)))
packed_img_size = data[:payload_size] # Get the image size
data = data[payload_size:] # get the remaining data
img_size = struct.unpack(">L", packed_img_size)[0]
print("[!] img_size: {}".format(img_size))


while len(data) < img_size: # Get data of image size from client
    data += conn.recv(4096)

print("[+] Image downloaded")

img_data = data[:img_size]
data = data[img_size:]
img = pickle.loads(img_data, fix_imports=True, encoding="bytes") # Convert image bytes to image
#img = cv2.imdecode(img, cv2.IMREAD_COLOR)

print("[*] Started Decryption ")
img = chaosMap.chaosDecryption(img,key,type_) # decrypt image
cv2.imwrite('download.png',img)
print("[+] Decryption Complete")

soc.close()