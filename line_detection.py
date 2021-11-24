import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    return 1



def convertVideo2Frames(video):
    vidcap = cv2.VideoCapture(video)
    success,frame = vidcap.read()
    cntr=1

    path = 'C:/Users/idano/Documents/HomeWork/3rd semester/computer vision/Line_Detection_Proj/frames'

    while success:
        cv2.imwrite(path+"\\frame%d.jpg" %cntr,frame)
        success,frame = vidcap.read()
        cntr+=1
    
    return


if __name__ == "__main__":
    main()
    print("hey")
    ##idan ?

