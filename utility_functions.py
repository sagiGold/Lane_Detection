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
    