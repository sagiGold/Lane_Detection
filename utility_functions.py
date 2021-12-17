import numpy as np
import matplotlib.pyplot as plt
import cv2

def import_frame(image_dir):
    frame =cv2.imread(image_dir)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def show_image(image,figsize):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    plt.show()


def convertVideo2Frames(video):
    vidcap = cv2.VideoCapture(video)
    success,frame = vidcap.read()
    cntr=1

    path = 'C:/Users/idano/Documents/HomeWork/3rd semester/computer vision/Line_Detection_Proj/frames'

    while success:
        cv2.imwrite(path+"\\frame%d.jpg" %cntr,frame)
        success,frame = vidcap.read()
        cntr+=1

def frameList2Video(list):
    h,w,d = list[0].shape
    out = cv2.VideoWriter('lane_vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30,(w,h))
    for frame in list:
        out.write(frame)

def createMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = np.array([(600,img.shape[0]), (200,img.shape[0]), (320,370), (480,370)])
    mask = np.zeros(img.shape,dtype=np.uint8)
    cv2.fillConvexPoly(mask,pts,255)
    return mask

def plotParameterSpace(lines):
    rho = []
    th = []

    for l in lines:
        r,t = l[0]
        rho.append(r)
        th.append(t)

    plt.scatter(rho,th)
    plt.xlabel('rho')
    plt.ylabel('theta')
    plt.show()
