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

#%%
mag_im = fpl.canny_image(img)
util.show_image(mag_im,figsize)

#%%
mask = np.zeros(mag_im.shape,dtype=np.uint8)
mask = cv2.rectangle(mask,(180,400),(550,mask.shape[0]),255,-1)
util.show_image(mask,figsize)
#%%
masked_im = cv2.bitwise_and(mag_im,mask)
util.show_image(masked_im,figsize)
#%%
fpl.find_lines(masked_im,masked_im,30)



#%%
# cropped_img = fpl.frame_find_lanes(img)
cropped_img = img[375:,120:734]
print(cropped_img.shape)
util.show_image(cropped_img,figsize)
#%%
# ret,im_th = cv2.threshold(cropped_img,200,255,cv2.THRESH_BINARY)
# util.show_image(im_th,figsize)

# #%%
# im_edge = cv2.Canny(im_th,0,255)
# util.show_image(im_edge,figsize)




# #%%
# r_step = 1
# t_step =np.pi/180
# TH = 40
# lines = cv2.HoughLines(im_edge,r_step,t_step,TH)
# print(lines)


# res = cropped_img.copy()

# res = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

# for r_t in lines:
#     rho = r_t[0, 0]
#     theta = r_t[0, 1]

#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))

#     res = cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)

# plt.imshow(res)
# plt.show()
#%%

# y = -0.554309x, y = 0.7535542x 
cv2.circle(img,(img.shape[0],0),5,(0,0,250),-1)
cv2.circle(img,(0,800),5,(0,0,250),-1)
cv2.circle(img,(400,132),5,(0,0,250),-1)
cv2.circle(img,(400,132),5,(0,0,250),-1)






cv2.circle(img,(100,400),5,(0,0,250),-1)
cv2.circle(img,(700,400),5,(0,0,250),-1)
cv2.circle(img,(0,400),5,(0,0,250),-1)
cv2.circle(img,(730,470),5,(0,0,250),-1)

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