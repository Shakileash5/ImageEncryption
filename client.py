import socket
import os
import json
import tqdm

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # create socket object
BUFFER_SIZE = 4096
host = "127.0.0.1" # host to connect to
port = 5001 # port to connect to

print(f"[+] Connecting to {host}:{port}")
soc.connect((host, port)) # connect to host
print("[+] Connected.")

fileName = 'test.jpg'
fileSize = os.path.getsize(fileName)

data = fileName + ':' + str(fileSize)
soc.send(data.encode())
file_ = open(fileName, 'rb')

while True:
    # read the bytes from the file
    bytes_read = file_.read(BUFFER_SIZE)
    if not bytes_read:
        # file transmitting is done
        break
    # we use sendall to assure transimission in 
    # busy networks
    soc.sendall(bytes_read)

soc.close()
