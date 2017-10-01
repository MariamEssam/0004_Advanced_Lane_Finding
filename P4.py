import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from Funcs import *
from Classes import *
from moviepy.editor import VideoFileClip


def pipeline(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    undst,mtx = undistortimage(img,objpoints,imgpoints,nx,ny)
    Combined=CombinedImage(undst)
    warped_img,M,Minv=Transformimage(img, nx, ny, mtx, Combined)
    #plt.imshow(warped_img,cmap='gray')
    #plt.show()
    #left_fit, right_fit, left_lane_inds, right_lane_inds,nonzerox,nonzeroy=SlidingWindow(warped_img)
    left_fit,right_fit,left_lane_inds,right_lane_inds=SlidingWindow(warped_img)
    #FillVisualizer(warped_img,nonzerox,nonzeroy,left_lane_inds,right_lane_inds,left_fitx,right_fitx,ploty)
    #DrawLaneArea(img,undst,warped_img,left_fit,right_fit,Minv)
    CalcCurvature(warped_img,left_fit,right_fit,left_lane_inds,right_lane_inds)
    return warped_img,undst,M,Minv

def process_image_video(img):
    #Execute the pipeline on the new image
    warped_img,undst,M,Minv=pipeline(img)

    if left_line.detected==True and right_line.detected==True:
        left_fit,right_fit,left_lane_inds,right_lane_inds=Polyfit(warped_img,left_line.best_fit,right_line.best_fit)
    else:
         left_fit,right_fit,left_lane_inds,right_lane_inds=SlidingWindow(warped_img)
     
    #Sanity Check
    if left_fit is not None and right_fit is not None:
        h = img.shape[0]
        l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        x_int_diff = abs(r_fit_x_int-l_fit_x_int)
        #Check distance between the two lines
        if abs(350 - x_int_diff) > 100:
            l_fit = None
            r_fit = None
    rad_l, rad_r = CalcCurvature(warped_img,left_fit,right_fit,left_lane_inds,right_lane_inds)
    rad_l_K=rad_l/1000
    rad_r_K=rad_r/1000
    if rad_l_K>0.5 and rad_l_K<3 and rad_r_K<3 and rad_r_K>0.5:       
        left_line.add_fit(left_fit, left_lane_inds,rad_l)
        right_line.add_fit(right_fit, right_lane_inds,rad_r)
    else:
         l_fit = None
         r_fit = None
    # draw the current best fit if it exists
    if left_line.best_fit is not None and right_line.best_fit is not None:
        img_out=DrawLaneArea(img,undst,warped_img,left_line.best_fit,right_line.best_fit,Minv)
        text = 'Left Curve radius: ' + '{:04.2f}'.format(left_line.radius_of_curvature) + 'm'
        cv2.putText(img_out, text, (40,70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)
        text = 'Right Curve radius: ' + '{:04.2f}'.format(right_line.radius_of_curvature) + 'm'
        cv2.putText(img_out, text, (40,120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)
    else:
        img_out = img

    return img_out

nx=9
ny=6
#Calibrate the Camera
objpoints,imgpoints= CalibrateCamera(nx,ny)
# Test undistortion on an image
#img = cv2.imread('test_images/test3.jpg')
#mynew,undst,M,Minv=pipeline(img)
left_line=Line()
right_line=Line()
video_output = 'project_video_output.mp4'
video_input = VideoFileClip('project_video.mp4')
processed_video = video_input.fl_image(process_image_video)
processed_video.write_videofile(video_output, audio=False)


#Combined_C,Combined_img_F,Combined=CombinedImage(undst)
#warped_img,M=Transformimage(img, nx, ny, mtx, Combined_C,corners)
#warped_Original,M=Transformimage(img, nx, ny, mtx, undst,corners)
#Combined_img_F,M=Transformimage(img, nx, ny, mtx, Combined_img_F,corners)
#warped_final,M=Transformimage(img, nx, ny, mtx, Combined,corners)
### Visualize undistortion
#f, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,10))
#ax1.imshow(warped_Original)
#ax1.set_title('Original', fontsize=30)
#ax2.imshow(warped_img)
#ax2.set_title('Color Threshold', fontsize=30)
#ax3.imshow(Combined_img_F)
#ax3.set_title('Filters', fontsize=30)
#ax4.imshow(warped_final)
#ax4.set_title('Result', fontsize=30)
#plt.show()


