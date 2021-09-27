import cv2
import numpy as np
import socket
import sys
import pickle
import struct ### new code


sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.connect(('localhost',8089))

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

frame=cv2.imread('test.jpg')
result, frame = cv2.imencode('.jpg', frame, encode_param)
data = pickle.dumps(frame, 0)
size = len(data)
print("{}: {}".format(img_counter, size))
sock.sendall(struct.pack(">L", size) + data)
sock.close()