import numpy as np
import matplotlib.pyplot as plt
import cv2
import utility_functions as util
import glob


def frame_find_lanes(frame,src,mask,cropped):

    figsize = (10,10)
    # img = util.import_frame(frame)
    img = frame
    #util.show_image(img,figsize)
    mag_im = canny_image(img)
    # util.show_image(mag_im,figsize)

    # pts = np.array([(600,img.shape[0]), (200,img.shape[0]), (320,370), (480,370)])
    # mask = np.zeros(mag_im.shape,dtype=np.uint8)

    # cv2.fillConvexPoly(mask,pts,1)

    masked_im = cv2.bitwise_and(mag_im,mask)
    # util.show_image(masked_im,figsize)
    return find_lines(masked_im,src,cropped,30)

    # return find_lines(masked_im,src,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),30)

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


def find_lines(masked_img,result,cropped,TH):
    r_step = 1
    t_step =np.pi/180
    lines = cv2.HoughLines(masked_img,r_step,t_step,TH)

    res = result.copy()
    if lines is not None:
        #to find the lanes we use sorting by rho and then we tolerance to find the median parameters
        lines_sorted = sorted(lines, key=lambda a_entry: a_entry[..., 0])
        
        # function to plot the parameter space :
        # plotParameterSpace(lines_sorted)

        tl_line = tolerance_lines(lines_sorted)

        res = markLanes(tl_line,res,cropped)
    return res

def canny_image(src_im):
    figsize = (10,10)
    ret,im_th = cv2.threshold(src_im,220,255,cv2.THRESH_BINARY)
    im_edge = cv2.Canny(im_th,0,255)
    return im_edge


#returns the median parameter lines from list of lines
def tolerance_lines(sorted_lines):
    tl_lines= np.array([sorted_lines[0]])
    tol_r = 200
    tol_t = 0.7
    i = 0

    for idx, line in enumerate(sorted_lines):
        #checks if current line can be tolerated
        #checks if tl_line and line absulute diff is smaller than the tolerace
        if(abs(tl_lines[i][0][0] - line[0][0]) < tol_r and abs(tl_lines[i][0][1]-line[0][1]) < tol_t):
            tl_lines[i][0][0] = (idx*tl_lines[i][0][0]+line[0][0])/(idx + 1)
            tl_lines[i][0][1] = (idx*tl_lines[i][0][1]+line[0][1])/(idx + 1)
        else:
            #adds new line to the array
            tl_lines = np.vstack([tl_lines,np.array([line])])
            i+=1

    return tl_lines

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

def markLanes(param_lines, src, mask_bgr):

    canvas = np.ones((src.shape[0],src.shape[1],3),dtype=np.uint8)
    for r_t in param_lines:
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
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), thickness=10)
    
    canvas = cv2.bitwise_and(canvas,mask_bgr)

    return cv2.addWeighted(src,1,canvas,0.8,0)
    
    