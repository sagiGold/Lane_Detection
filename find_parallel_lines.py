import numpy as np
import matplotlib.pyplot as plt
import cv2
import utility_functions as util

def frame_find_lanes(frame):
    cropped_img = frame[375:,:]
    # util.show_image(cropped_img,(10,10))
    # orig_image_coor = np.array([[0,0],[0,104],[359,0],[359,104]])
    # for x in orig_image_coor:
    #     cv2.circle(cropped_img,x,10,(0,255,0))
    #     util.show_image(cropped_img,(10,10))

    return cropped_img



#%%


#  Cordinates of the 4 corners of the original source image
def perspective_trasnform(cropped_img, source_img):
    height, width =  350,420

    # coordinates of the 4 corners of the target output
    new_img_coor = np.array([[0,0],[width,0],[0,height],[width,height]])

    # use the two sets of four points to compute the perspective transformation matrix
    compute_mat = cv2.getPerspectiveTransform(cropped_img, new_img_coor)

    return cv2.warpPerspective(source_img,compute_mat,(width,height))

     

    
