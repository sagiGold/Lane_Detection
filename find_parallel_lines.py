import numpy as np
import matplotlib.pyplot as plt
import cv2
import utility_functions as util

def frame_find_lanes(frame):
    cropped_img = frame[375:,:700]
    # util.show_image(cropped_img,(10,10))
    # orig_image_coor = np.array([[0,0],[0,104],[359,0],[359,104]])
    # for x in orig_image_coor:
    #     cv2.circle(cropped_img,x,10,(0,255,0))
    #     util.show_image(cropped_img,(10,10))

    return cropped_img



#%%


#  Cordinates of the 4 corners of the original source image
def perspective_trasnform(source_img):

    cropped_img = np.float32([[100,400],[700,400],[0,470],[800,470]])
    height, width =  70,600

    # coordinates of the 4 corners of the target output
    new_img_coor = np.float32([[0,0],[width,0],[0,height],[width,height]])

    # use the two sets of four points to compute the perspective transformation matrix
    compute_mat = cv2.getPerspectiveTransform(cropped_img, new_img_coor)

    perspective = cv2.warpPerspective(source_img,compute_mat,(width,height))
    
    return perspective


# #  Cordinates of the 4 corners of the original source image
# def perspective_trasnform(cropped_img, source_img):
#     height, width =  100,850
#     # coordinates of the 4 corners of the target output
#     new_img_coor = np.array([[0,0],[width,0],[0,height],[width,height]])
#     print(new_img_coor)
#     # use the two sets of four points to compute the perspective transformation matrix
#     compute_mat = cv2.getPerspectiveTransform(cropped_img, new_img_coor)

#     return cv2.warpPerspective(source_img,compute_mat,(width,height))


def find_lines(src_im,result,TH):
    r_step = 1
    t_step =np.pi/180
    lines = cv2.HoughLines(src_im,r_step,t_step,TH)
    print(lines)


    res = result.copy()

    res = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

    for r_t in lines:
        rho = r_t[0, 0]
        theta = r_t[0, 1]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        res = cv2.line(res, (x1, y1), (x2, y2), (255, 255, 0), thickness=3)

        # print(str(a)+'x +'+str(b) +'y = '+str(rho))
        # print('y = '+str(-a/b)+'x +'+str(rho/b))
    res = cv2.line(res, (0, 400), (854, 400), (255 ,0 , 0), thickness=3)

    plt.imshow(res)
    plt.show()

def canny_image(src_im):
    figsize = (10,10)
    ret,im_th = cv2.threshold(src_im,200,255,cv2.THRESH_BINARY)
    im_edge = cv2.Canny(im_th,0,255)
    return im_edge
