#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import utility_functions as util
import find_parallel_lines as fpl


def main():
    run('dash_cam.mp4')

def run(video):
    vidcap = cv2.VideoCapture(video)
    success,frame = vidcap.read()

    mask = util.createMask(frame)

    cntr=0
    frame_list=[]
    #path = 'C:/Users/idano/Documents/HomeWork/3rd semester/computer vision/Line_Detection_Proj/frames_new'
    # frame = util.import_frame("frame827.jpg")
    while success:
        frame_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        frame_list.append(fpl.frame_find_lanes(frame_gray,frame,mask))
        #cv2.imwrite(path+"\\frame%d.jpg" %cntr,frame_list[cntr])
        success,frame = vidcap.read()
        cntr+=1

    util.frameList2Video(frame_list)


def test():
    path = glob.glob("./frames/*.jpg")
    new_frame = fpl.frame_find_lanes(path[1083])
    util.show_image(new_frame,(10,10))
    return 1

if __name__ == "__main__":
    main()
    cv2.waitKey(0)
    print("hey")


# %%
