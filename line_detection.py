#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

import utility_functions as util
import find_parallel_lines as fpl

#%%
def main():
    util.import_frame(".frames/frame75.jpg")
    return 1

figsize = (10,10) 
img = util.import_frame('frame75.jpg')
util.show_image(img,figsize)

cropped_img = fpl.frame_find_lanes(img)
util.show_image(cropped_img,figsize)

#%%
cv2.circle(img,(100,400),5,(0,0,250),-1)
cv2.circle(img,(700,400),5,(0,0,250),-1)
cv2.circle(img,(0,470),5,(0,0,250),-1)
cv2.circle(img,(800,470),5,(0,0,250),-1)

util.show_image(img,figsize)


#%%
p_img = fpl.perspective_trasnform(img)
util.show_image(p_img,figsize)
# p_img = fpl.perspective_trasnform(cropped_img,img)
# util.show_image(p_img,figsize)

#%%
if __name__ == "__main__":
    main()
    print("hey")

#%%