import socket
import json
import os

soc = socket.socket() # Create a socket object
host = "127.0.0.1" # server ip
port = 5001 # Reserve a port for server.

soc.bind((host, port)) # Bind to the port
soc.listen(5) # Now wait for client connection.
print(f"[*] Listening as {host}:{port}")
# receive 4096 bytes each time
BUFFER_SIZE = 4096

conn, addr = soc.accept() # Establish connection with client.
print(f"[+] Got a connection from {addr[0]}:{addr[1]}")
fileData = conn.recv(1024).decode()
fileName, fileSize = fileData.split(":")
fileName = fileName.split(".")
fileName[0] = fileName[0]+ "_download"
fileName = ".".join(fileName)
filename = os.path.basename(fileName)

file_ = open(fileName, "wb")
while True:
    bytesReceived = conn.recv(BUFFER_SIZE)
    if not bytesReceived:
        break
    file_.write(bytesReceived)

conn.close()
print("[+] Download complete")
soc.close()

