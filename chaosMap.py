import cv2
from PIL import Image
import numpy as np
from math import log


def ArnoldCatTransform(img):
    """
    Function to perform the Arnaud-Cat transformation

    Args:
        img: The image to be encrypted

    Returns:
        The encrypted image
    """
    rows, cols, ch = img.shape
    n = rows
    img_arnold = np.zeros([rows, cols, ch])
    for x in range(0, rows):
        for y in range(0, cols):
            img_arnold[x][y] = img[(x+y)%n][(x+2*y)%n]  
    return img_arnold  

def ArnoldCatEncryption(img, key):
    """
    Encrypts the image using the Arnaud-Cat algorithm

    Args:
        img: The image to be encrypted
        key: The key to be used for encryption
    
    Returns:
        The encrypted image

    """
    for i in range (0,key):
        img = ArnoldCatTransform(img)
    return img

def ArnoldCatDecryption(img, key):
    """
    Decrypts the image using the Arnaud-Cat algorithm

    Args:
        img: The image to be decrypted
        key: The key to be used for decryption
    
    Returns:
        The decrypted image
    """
    rows, cols, ch = img.shape
    dimension = rows
    decrypt_it = dimension
    if (dimension%2==0) and 5**int(round(log(dimension/2,5))) == int(dimension/2):
        decrypt_it = 3*dimension
    elif 5**int(round(log(dimension,5))) == int(dimension):
        decrypt_it = 2*dimension
    elif (dimension%6==0) and  5**int(round(log(dimension/6,5))) == int(dimension/6):
        decrypt_it = 2*dimension
    else:
        decrypt_it = int(12*dimension/7)
    for i in range(key,decrypt_it):
        img = ArnoldCatTransform(img, i)
    return img

def getImageMatrix(imageName):
    """
    Function to open image and convert into matrix

    Args:
        imageName: The name of the image to be converted
    
    Returns:
        The image matrix
    """
    im = Image.open(imageName) 
    pix = im.load()
    color = 1
    print(pix[0,0])
    if type(pix[0,0]) == int:
      color = 0
    image_size = im.size 
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
                row.append((pix[width,height]))
        image_matrix.append(row)
    return image_matrix,image_size[0],color

def getImageMatrixCV(img):
    """
    Convert image into matrix

    Args:
        img: The image to be converted
    
    Returns:
        The image matrix
    """
    img = cv2.rotate(img[::-1], cv2.ROTATE_90_CLOCKWISE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    pix = Image.fromarray(img)
    color = 1
    print(img[0,0][::-1])
    if type(img[0,0]) == int:
      color = 0
    image_size = (img.shape[1],img.shape[0])
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
                row.append((img[width,height]))
        image_matrix.append(row)
    return image_matrix,image_size[0],color

def dec(bitSequence):
    """
    Function to convert bitSequence into decimal

    Args:
        bitSequence: The bitSequence to be converted
    
    Returns:
        The decimal value of the bitSequence
    """
    decimal = 0
    for bit in bitSequence:
        decimal = decimal * 2 + int(bit)
    return decimal

def genHenonMap(dimension, key):
    """
    Function to generate the Henon map

    Args:
        dimension: The dimension of the map
        key: The key to be used for encryption

    Returns:
        The Henon map
    """
    x = key[0]
    y = key[1]
    sequenceSize = dimension * dimension * 8 #Total Number of bitSequence produced
    bitSequence = []    #Each bitSequence contains 8 bits
    byteArray = []      #Each byteArray contains m( i.e 512 in this case) bitSequence
    TImageMatrix = []   #Each TImageMatrix contains m*n byteArray( i.e 512 byteArray in this case)
    for i in range(sequenceSize):
        xN = y + 1 - 1.4 * x**2
        yN = 0.3 * x

        x = xN
        y = yN

        if xN <= 0.4:
            bit = 0
        else:
            bit = 1

        try:
            bitSequence.append(bit)
        except:
            bitSequence = [bit]

        if i % 8 == 7:
            decimal = dec(bitSequence)
            try:
                byteArray.append(decimal)
            except:
                byteArray = [decimal]
            bitSequence = []

        byteArraySize = dimension*8
        if i % byteArraySize == byteArraySize-1:
            try:
                TImageMatrix.append(byteArray)
            except:
                TImageMatrix = [byteArray]
            byteArray = []
    return TImageMatrix

def HenonEncryption(img,key,flag = 0):
    """
    Encrypts the image using the Henon algorithm

    Args:
        img: The image to be encrypted
        key: The key to be used for encryption
        flag: The flag to be used for encryption(1 if image is in CV format,0 if image name is given)

    Returns:
        The encrypted image
    """
    if flag == 0:
        imageMatrix, dimension, color = getImageMatrix(img)
    else:
        imageMatrix, dimension, color = getImageMatrixCV(img)
    transformationMatrix = genHenonMap(dimension, key)
    resultantMatrix = []
    for i in range(dimension):
        row = []
        for j in range(dimension):
            try:
                if color:
                    row.append(tuple([transformationMatrix[i][j] ^ x for x in imageMatrix[i][j]]))
                else:
                    row.append(transformationMatrix[i][j] ^ imageMatrix[i][j])
            except:
                if color:
                    row = [tuple([transformationMatrix[i][j] ^ x for x in imageMatrix[i][j]])]
                else :
                    row = [transformationMatrix[i][j] ^ x for x in imageMatrix[i][j]]
        try:    
            resultantMatrix.append(row)
        except:
            resultantMatrix = [row]
    if color:
      im = Image.new("RGB", (dimension, dimension))
    else: 
      im = Image.new("L", (dimension, dimension)) # L is for Black and white pixels

    pix = im.load()
    for x in range(dimension):
        for y in range(dimension):
            pix[x, y] = resultantMatrix[x][y]
    #im.save("T_HenonEnc.png", "PNG")
    numpy_image=np.array(im)
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
    return opencv_image 

def HenonDecryption(img, key,flag=0):
    """
    Decrypt the image using the Henon algorithm

    Args:
        img: The image to be decrypted
        key: The key to be used for decryption
        flag: The flag to be used for decryption(1 if image is in CV format,0 if image name is given)
    
    Returns:
        The decrypted image
    """
    if flag == 0:
        imageMatrix, dimension, color = getImageMatrix(img)
    else:
        imageMatrix, dimension, color = getImageMatrixCV(img)
    transformationMatrix = genHenonMap(dimension, key)
    henonDecryptedImage = []
    for i in range(dimension):
        row = []
        for j in range(dimension):
            try:
                if color:
                    row.append(tuple([transformationMatrix[i][j] ^ x for x in imageMatrix[i][j]]))
                else:
                    row.append(transformationMatrix[i][j] ^ imageMatrix[i][j])
            except:
                if color:
                    row = [tuple([transformationMatrix[i][j] ^ x for x in imageMatrix[i][j]])]
                else :
                    row = [transformationMatrix[i][j] ^ x for x in imageMatrix[i][j]]
        try:
            henonDecryptedImage.append(row)
        except:
            henonDecryptedImage = [row]
    if color:
        im = Image.new("RGB", (dimension, dimension))
    else: 
        im = Image.new("L", (dimension, dimension)) # L is for Black and white pixels

    pix = im.load()
    for x in range(dimension):
        for y in range(dimension):
            pix[x, y] = henonDecryptedImage[x][y]
    #im.save( "TTT_HenonDec.png", "PNG")
    numpy_image=np.array(im)
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
    return opencv_image  

def LogisticEncryption(img, key,flag=0):
    """
    Encrypt the image using the Logistic algorithm

    Args:
        img: The image to be encrypted
        key: The key to be used for encryption
        flag: The flag to be used for encryption(1 if image is in CV format,0 if image name is given)
    
    Returns:
        The encrypted image
    """
    N = 256
    key_list = [ord(x) for x in key]
    G = [key_list[0:4] ,key_list[4:8], key_list[8:12]]
    g = []
    R = 1
    for i in range(1,4):
        s = 0
        for j in range(1,5):
            s += G[i-1][j-1] * (10**(-j))
        g.append(s)
        R = (R*s) % 1

    L = (R + key_list[12]/256) % 1
    S_x = round(((g[0]+g[1]+g[2])*(10**4) + L *(10**4)) % 256)
    V1 = sum(key_list)
    V2 = key_list[0]
    for i in range(1,13):
        V2 = V2 ^ key_list[i]
    V = V2/V1

    L_y = (V+key_list[12]/256) % 1
    S_y = round((V+V2+L_y*10**4) % 256)
    C1_0 = S_x
    C2_0 = S_y
    C = round((L*L_y*10**4) % 256)
    C_r = round((L*L_y*10**4) % 256)
    C_g = round((L*L_y*10**4) % 256)
    C_b = round((L*L_y*10**4) % 256)
    x = 4*(S_x)*(1-S_x)
    y = 4*(S_y)*(1-S_y)
    
    if flag == 0:
        imageMatrix, dimension, color = getImageMatrix(img)
    else:
        imageMatrix, dimension, color = getImageMatrixCV(img)
    LogisticEncryptionIm = []
    for i in range(dimension):
        row = []
        for j in range(dimension):
            while x <0.8 and x > 0.2 :
                x = 4*x*(1-x)
            while y <0.8 and y > 0.2 :
                y = 4*y*(1-y)
            x_round = round((x*(10**4))%256)
            y_round = round((y*(10**4))%256)
            C1 = x_round ^ ((key_list[0]+x_round) % N) ^ ((C1_0 + key_list[1])%N)
            C2 = x_round ^ ((key_list[2]+y_round) % N) ^ ((C2_0 + key_list[3])%N) 
            if color:
              C_r =((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ ((key_list[6]+imageMatrix[i][j][0]) % N) ^ ((C_r + key_list[7]) % N)
              C_g =((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ ((key_list[6]+imageMatrix[i][j][1]) % N) ^ ((C_g + key_list[7]) % N)
              C_b =((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ ((key_list[6]+imageMatrix[i][j][2]) % N) ^ ((C_b + key_list[7]) % N)
              row.append((C_r,C_g,C_b))
              C = C_r

            else:
              C = ((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ ((key_list[6]+imageMatrix[i][j]) % N) ^ ((C + key_list[7]) % N)
              row.append(C)

            x = (x + C/256 + key_list[8]/256 + key_list[9]/256) % 1
            y = (x + C/256 + key_list[8]/256 + key_list[9]/256) % 1
            for ki in range(12):
                key_list[ki] = (key_list[ki] + key_list[12]) % 256
                key_list[12] = key_list[12] ^ key_list[ki]
        LogisticEncryptionIm.append(row)

    im = Image.new("L", (dimension, dimension))
    if color:
        im = Image.new("RGB", (dimension, dimension))
    else: 
        im = Image.new("L", (dimension, dimension)) # L is for Black and white pixels
      
    pix = im.load()
    for x in range(dimension):
        for y in range(dimension):
            pix[x, y] = LogisticEncryptionIm[x][y]
    #im.save(imageName.split('.')[0] + "_LogisticEnc.png", "PNG")
    numpy_image=np.array(im)
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
    return opencv_image 


def LogisticDecryption(img, key, flag=0):
    """
    Decrypt the image using the Logistic algorithm

    Args:
        img: The image to be decrypted
        key: The key to be used for decryption
        flag: The flag to be used for decryption(1 if image is in CV format,0 if image name is given)
    
    Returns:
        The decrypted image
    """
    N = 256
    key_list = [ord(x) for x in key]

    G = [key_list[0:4] ,key_list[4:8], key_list[8:12]]
    g = []
    R = 1
    for i in range(1,4):
        s = 0
        for j in range(1,5):
            s += G[i-1][j-1] * (10**(-j))
        g.append(s)
        R = (R*s) % 1
    
    L_x = (R + key_list[12]/256) % 1
    S_x = round(((g[0]+g[1]+g[2])*(10**4) + L_x *(10**4)) % 256)
    V1 = sum(key_list)
    V2 = key_list[0]
    for i in range(1,13):
        V2 = V2 ^ key_list[i]
    V = V2/V1

    L_y = (V+key_list[12]/256) % 1
    S_y = round((V+V2+L_y*10**4) % 256)
    C1_0 = S_x
    C2_0 = S_y
    
    C = round((L_x*L_y*10**4) % 256)
    I_prev = C
    I_prev_r = C
    I_prev_g = C
    I_prev_b = C
    I = C
    I_r = C
    I_g = C
    I_b = C
    x_prev = 4*(S_x)*(1-S_x)
    y_prev = 4*(L_x)*(1-S_y)
    x = x_prev
    y = y_prev

    if flag == 0:
        imageMatrix, dimension, color = getImageMatrix(img)
    else:
        imageMatrix, dimension, color = getImageMatrixCV(img)

    henonDecryptedImage = []
    for i in range(dimension):
        row = []
        for j in range(dimension):
            while x <0.8 and x > 0.2 :
                x = 4*x*(1-x)
            while y <0.8 and y > 0.2 :
                y = 4*y*(1-y)
            x_round = round((x*(10**4))%256)
            y_round = round((y*(10**4))%256)
            C1 = x_round ^ ((key_list[0]+x_round) % N) ^ ((C1_0 + key_list[1])%N)
            C2 = x_round ^ ((key_list[2]+y_round) % N) ^ ((C2_0 + key_list[3])%N) 
            if color:
                I_r = ((((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ ((I_prev_r + key_list[7]) % N) ^ imageMatrix[i][j][0]) + N-key_list[6])%N
                I_g = ((((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ ((I_prev_g + key_list[7]) % N) ^ imageMatrix[i][j][1]) + N-key_list[6])%N
                I_b = ((((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ ((I_prev_b + key_list[7]) % N) ^ imageMatrix[i][j][2]) + N-key_list[6])%N
                I_prev_r = imageMatrix[i][j][0]
                I_prev_g = imageMatrix[i][j][1]
                I_prev_b = imageMatrix[i][j][2]
                row.append((I_r,I_g,I_b))
                x = (x +  imageMatrix[i][j][0]/256 + key_list[8]/256 + key_list[9]/256) % 1
                y = (x +  imageMatrix[i][j][0]/256 + key_list[8]/256 + key_list[9]/256) % 1  
            else:
                I = ((((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ ((I_prev+key_list[7]) % N) ^ imageMatrix[i][j]) + N-key_list[6])%N
                I_prev = imageMatrix[i][j]
                row.append(I)
                x = (x +  imageMatrix[i][j]/256 + key_list[8]/256 + key_list[9]/256) % 1
                y = (x +  imageMatrix[i][j]/256 + key_list[8]/256 + key_list[9]/256) % 1
            for ki in range(12):
                key_list[ki] = (key_list[ki] + key_list[12]) % 256
                key_list[12] = key_list[12] ^ key_list[ki]
        henonDecryptedImage.append(row)
    if color:
        im = Image.new("RGB", (dimension, dimension))
    else: 
        im = Image.new("L", (dimension, dimension)) # L is for Black and white pixels
    pix = im.load()
    for x in range(dimension):
        for y in range(dimension):
            pix[x, y] = henonDecryptedImage[x][y]
    #im.save(imageName.split('_')[0] + "_LogisticDec.png", "PNG")
    numpy_image=np.array(im)
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
    return opencv_image 


def chaosEncryption(img,key,algorithm=0):
    """
    Encrypt the image using the chaos encryption algorithms with the given key

    Args:
        img: The image to be encrypted
        key: The key to be used for encryption
        algorithm: The algorithm to be used for encryption. 0 for ArnoldCat , 1 for HenonMap, 2 for LogisticMap
    
    Returns:
        The encrypted image
    """
    if algorithm == 0:
        img = ArnoldCatEncryption(img, key)
    elif algorithm == 1:
        img = HenonEncryption(img, key, 1)
    else:
        img = LogisticEncryption(img, key, 1)
    
    return img

def chaosDecryption(img,key,algorithm=0):
    """
    Decrypt the image using the chaos encryption algorithms with the given key

    Args:
        img: The image to be decrypted
        key: The key to be used for decryption
        algorithm: The algorithm to be used for decryption. 0 for ArnoldCat , 1 for HenonMap, 2 for LogisticMap
    
    Returns:
        The decrypted image
    """
    if algorithm == 0:
        img = ArnoldCatDecryption(img, int(key))
    elif algorithm == 1:
        img = HenonDecryption(img, key, 1)
    else:
        img = LogisticDecryption(img, key, 1)
    
    return img
    
