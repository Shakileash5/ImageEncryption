import cv2
import numpy as np
import socket
import sys
import pickle
import struct 
import chaosMap

HOST = "127.0.0.1" # server ip
PORT = 5001 # Reserve a port for server.
sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM) # Create a socket object
sock.connect((HOST,PORT))

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

img = cv2.imread('HorizonZero.png')
key = 20#(0.1,0.1)
img = chaosMap.chaosEncryption(img,key,0)
#result, img = cv2.imencode('.jpg', img, encode_param)
data = pickle.dumps(img, 0)
size = len(data)
print("Size of the image : ",size)
sock.sendall(struct.pack(">L", size) + data)
sock.close()