import numpy as np
import matplotlib.pyplot as plt
import cv2
import utility_functions as util
import glob

def frame_find_lanes(frame):
    figsize = (10,10) 
    img = util.import_frame(frame)
    util.show_image(img,figsize)
    mag_im = canny_image(img)
    util.show_image(mag_im,figsize)

    mask = np.zeros(mag_im.shape,dtype=np.uint8)
    mask = cv2.rectangle(mask,(180,400),(550,mask.shape[0]),255,-1)
    #%%
    masked_im = cv2.bitwise_and(mag_im,mask)

    return find_lines(masked_im,img,30)

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


def find_lines(src_im,result,TH):
    r_step = 1
    t_step =np.pi/180
    lines = cv2.HoughLines(src_im,r_step,t_step,TH)
    print(lines)
    print("\n\n")
    lines_sorted = sorted(lines, key=lambda a_entry: a_entry[..., 0])
    # lines_sorted = np.sort(lines,axis=0)
    print(lines_sorted)


    res = result.copy()

    tl_line = tolerance_lines(lines_sorted)
    print(tl_line.shape)


    res = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

    for r_t in tl_line:
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
        res = cv2.line(res, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

    return res

def canny_image(src_im):
    figsize = (10,10)
    ret,im_th = cv2.threshold(src_im,200,255,cv2.THRESH_BINARY)
    im_edge = cv2.Canny(im_th,0,255)
    return im_edge



def tolerance_lines(sorted_lines):
    tl_lines= np.array([sorted_lines[0]])
    tol_r = 15
    tol_t = 0.1

    i = 0
    for idx, line in enumerate(sorted_lines):
        #checks if current line can be tolerated
        if(abs(tl_lines[i][0][0] - line[0][0]) < tol_r and abs(tl_lines[i][0][1]-line[0][1]) < tol_t):
            tl_lines[i][0][0] = (idx*tl_lines[i][0][0]+line[0][0])/(idx + 1)
            tl_lines[i][0][1] = (idx*tl_lines[i][0][1]+line[0][1])/(idx + 1)
        else:
            tl_lines = np.vstack([tl_lines,np.array([line])])
            i+=1

    return tl_lines


# cv2.circle(img,(100,400),5,(0,0,250),-1)
# cv2.circle(img,(700,400),5,(0,0,250),-1)
# cv2.circle(img,(0,400),5,(0,0,250),-1)
# cv2.circle(img,(730,470),5,(0,0,250),-1)

# util.show_image(img,figsize)




# #%%
# p_img = fpl.perspective_trasnform(img)
# util.show_image(p_img,figsize)
# p_img = fpl.perspective_trasnform(cropped_img,img)
# util.show_image(p_img,figsize)

