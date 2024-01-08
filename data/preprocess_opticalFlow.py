import os
import cv2

if __name__ == "__main__":
    
    image1 = cv2.imread("./Competition_Image_preprocessed/000073.png")
    image2 = cv2.imread("./Competition_Image_preprocessed/000074.png")
    optical = image2 - image1
    cv2.imwrite("optical.png", optical)