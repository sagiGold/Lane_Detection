import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import utility_functions as util
import find_parallel_lines as fpl


def main():
    path = glob.glob("./frames/*.jpg")
    new_frame = fpl.frame_find_lanes(path[74])
    util.show_image(new_frame,(10,10))
    return 1



if __name__ == "__main__":
    main()
    cv2.waitKey(0)
    print("hey")

