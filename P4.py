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
    undst,mtx = undistortimage(img,objpoints,imgpoints,nx,ny, mtx_Calibration, dist_Calibration)
    Combined=CombinedImage(undst)
    warped_img,M,Minv=Transformimage(img, nx, ny, mtx, Combined)
    #plt.imshow(warped_img)
    #plt.title(path)
    #plt.show()
    #left_fit, right_fit, left_lane_inds, right_lane_inds,nonzerox,nonzeroy=SlidingWindow(warped_img)
    left_fit,right_fit,left_lane_inds,right_lane_inds=SlidingWindow(warped_img)
    #FillVisualizer(warped_img,nonzerox,nonzeroy,left_lane_inds,right_lane_inds,left_fitx,right_fitx,ploty)
    DrawLaneArea(img,undst,warped_img,left_fit,right_fit,Minv)
    #CalcCurvature(warped_img,left_fit,right_fit,left_lane_inds,right_lane_inds)
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
            left_fit = None
            right_fit = None
        rad_l, rad_r,_ = CalcCurvature(warped_img,left_fit,right_fit,left_lane_inds,right_lane_inds)
        rad_l_K=rad_l/1000
        rad_r_K=rad_r/1000
        if rad_l_K>0.45 and rad_l_K<1.75 and rad_r_K<1.75 and rad_r_K>0.45:       
           left_line.add_fit(left_fit, left_lane_inds,rad_l)
           right_line.add_fit(right_fit, right_lane_inds,rad_r)
        else:
             left_fit = None
             right_fit = None

    # draw the current best fit if it exists
    if left_line.best_fit is not None and right_line.best_fit is not None:
        rad_l, rad_r,center_dist = CalcCurvature(warped_img,left_line.best_fit,right_line.best_fit,left_lane_inds,right_lane_inds)
        img_out=DrawLaneArea(img,undst,warped_img,left_line.best_fit,right_line.best_fit,Minv)
        text = 'Curve radius L: ' + '{:04.2f}'.format(rad_l) + 'm'+ '{:04.2f}'.format(rad_r) + 'm'
        cv2.putText(img_out, text, (40,70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)
        text = 'Center: ' + '{:04.2f}'.format(abs(center_dist)) + 'm'
        cv2.putText(img_out, text, (40,120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)
        #plt.imshow(img_out)
        #plt.show()
    else:
        img_out = img

    return img_out

nx=9
ny=6
#Calibrate the Camera
objpoints,imgpoints, mtx_Calibration, dist_Calibration= CalibrateCamera(nx,ny)

# Test undistortion on an image
#import glob
#pathes=glob.glob('test_images/ScreenHunter 12.jpg')
#for path in pathes:
#    img = cv2.imread(path)
#    mynew,undst,M,Minv=pipeline(img,path)
#undst,mtx = undistortimage(img,objpoints,imgpoints,nx,ny, mtx_Calibration, dist_Calibration)
#img = cv2.imread('camera_cal/calibration1.jpg')
#img_size = (img.shape[1], img.shape[0])
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
#dst = cv2.undistort(img, mtx, dist, None, mtx)

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
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#ax1.imshow(img)
#ax1.set_title('Original', fontsize=30)
#ax2.imshow(undst)
#ax2.set_title('Undistorted', fontsize=30)
#ax3.imshow(Combined_img_F)
#ax3.set_title('Filters', fontsize=30)
#ax4.imshow(warped_final)
#ax4.set_title('Result', fontsize=30)
#plt.show()


