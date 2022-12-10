import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc # For saving images as needed
import glob  # For reading in a list of images from a folder
import os

#########################################################################################################
############################# Functions needed for our realtime debugging mode ##########################
from perception import color_thresh
from perception import rover_coords
from perception import to_polar_coords
from perception import perspect_transform

 #########################################################################################################

cam = cv2.VideoCapture("../test_dataset/test_video/test_video.mp4")

# frame
currentframe = 0
  
while(True):
      
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        # if video is still left continue creating images
        name = '../test_dataset/IMG/frame' + str(currentframe) + '.jpg'
        
        # writing the extracted images
        cv2.imwrite(name, frame)
  
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()

def test_video_pipline():   

    path = '../test_dataset/IMG/*'
    img_list = glob.glob(path)
    
    
    
    # Grab a each image in a row
    i = 1
    for i in range(len(img_list)-1):
    
      
    
      dst_size = 5
      bottom_offset = 6
      
      #idx = np.random.randint(0, len(img_list)-1)
      #image = mpimg.imread(img_list[idx])
      
      
      image = mpimg.imread(img_list[i])
      
      
     
      
      
      
      source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
      destination = np.float32([
            [image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
            [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
            [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
            [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
            ])
    
      warped, mask = perspect_transform(image, source, destination)
      threshed = color_thresh(warped)
      
      

# Calculate pixel values in rover-centric coords and distance/angle to all pixels
      xpix, ypix = rover_coords(threshed)
      dist, angles = to_polar_coords(xpix, ypix)
      mean_dir = np.mean(angles)

# Do some plotting
      fig = plt.figure(figsize=(12,9))
      plt.subplot(221)
      plt.imshow(image)
      plt.subplot(222)
      plt.imshow(warped)
      plt.subplot(223)
      plt.imshow(threshed, cmap='gray')
      plt.subplot(224)
      plt.plot(xpix, ypix, '.')
      plt.ylim(-160, 160)
      plt.xlim(0, 160)
      arrow_length = 100
      x_arrow = arrow_length * np.cos(mean_dir)
      y_arrow = arrow_length * np.sin(mean_dir)
      plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
      
      fig.savefig("../test_dataset/IMG2/Image" + str(i) + ".png")  
    #  fig.savefig("../test_dataset/IMG2/Image.png")  
      
      plt.close(fig)
      
      
      
      
      
      
      #cv2.imwrite('../test_dataset/IMG2/PipeLining' + i + '.jpg', img)
      
      
       

             
test_video_pipline()

