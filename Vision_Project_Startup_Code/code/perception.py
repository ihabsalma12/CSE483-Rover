import numpy as np
import cv2
import keyboard # detect key presses

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
    
#threshed = color_thresh(warped)
#plt.imshow(threshed, cmap='gray')

 
# Define a function to convert from image coords to rover coords
# rover_coords
############With respect to rover############:
#The x-axis direction of the rover is is located at the center of the rover pointing in the forward #direction of the navigable train (with respect to the rover)


############With respect to Us############:
#the origin is at the bottom left and the x-axis positive direction is pointing towards the right #direction


############purpose of this function############:
#map the coordinates of the rover (that have different x-axis direction) into an image that have the #origin at the left and have x-axis pointing towards the right direction


############output of the function############:
#x-pixel , y-pixel are the new coordinates of the rover in the new image that have the origin at the left #and have x-axis pointing towards the right direction

############Implementation of the function############: 
#there are three operations
#1-mapping: the x-axis of the rover will be the y-axis of the new image
          #: the y-axis of the rover will be the x-axis of the new image
#2-Then  Use the equation of the Reflection

##NOTE:
# binary_img.shape[0]: represents the height the image(number of rows pixels)    
#(binary_img.shape[1]/2) represents the width of the image divided by 2(column pixels/2)
def rover_coords(binary_img):
    # Identify nonzero (white = navigable) pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
#rotate_pix function
############purpose of this function############:
#Map the axis of the rover x and the rover y to the world x and world y 
#this could be done by the rotation around the z-axis with the yaw rate of the robot

############output of the function############:
#the axis of the rover x and the rover y will be parallel to the axis of the world x and world y
#the function will return the x_pixel and y_pixel after rotation
  
############Implementation of the function############: 
#1- convert the yaw angle to radian by multiplying with (pi/180) 
#2- Multiply the position vector of the pixel with rotation matrix
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Rotation Matrix:
    #[x' y']T = [cos theta  -sin theta,  * [x y]T
    #            sin theta  cos theta]
    # theta will be yaw angle
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    # Translation and DIVIDING BY SCALE
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world
    



# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0])) #keep same size as input image (input y then x)
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]),M,(img.shape[1],img.shape[0]))
    return warped, mask

def find_rocks(img, levels=(110,110,50)):
    rockpix = ((img[:,:,0] > levels[0]) & (img[:,:,1] > levels[1]) & (img[:,:,2] < levels[2]))
    color_select = np.zeros_like(img[:,:,0])
    color_select[rockpix] = 1
    
    return color_select


class DebugListener():
    def __init__(self):
        self.debug_flag = False
        while True:  # making a loop
            try:  # used try so that if user pressed other than the given key error will not be shown
                if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                    print('You Pressed A Key!')
                    break  # finishing the loop
            except:
                break  # if user pressed a key other than the given key the loop will break
		    

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):

    debugger = DebugListener()
   
   
    """
    We will return the updated Rover. Update two params: Rover.vision_image and Rover.worldmap
    """   
   
   # We first calculate warped Rover perspective and extract the mask
   # Perspective is warped to bird eye view
   # We use 10 by 10 pixels to be destination size for one square meter of the grid
   # Later, we will map the 10 by 10 pixels to the worldmap

#perspect_transform
############purpose of this function############:
#1-Generate the mask of the bird eye view 
#2-apply the mask on the image to produce an image with the bird eye view

############Implementation of the function############: 
#Step1:
#M = cv2.getPerspectiveTransform(src, dst) 
#generation of the mask is done by having src points and dst points in order to estimate the value of the M

#Step2:
#warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
#apply the filter M on the image (img) and 
#the result will be a new image with dimention (img.shape[1], img.shape[0]) which is the same size of the input image

    dst_size = 5
    bottom_offset = 6
    image = Rover.img
    source = np.float32([[14, 140], # bottom left
                 [301 ,140], # bottom right
                 [200, 96], 
                 [118, 96]]) # arbitrarily clockwise or anti-clockwise square (same as destination)
    destination = np.float32([
            [image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
            [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
            [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
            [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
            ])
    
    warped, mask = perspect_transform(Rover.img, source, destination)
    # Color thresholding function will return B&W image, where navigable = 1
    threshed = color_thresh(warped) 
    # Using perspect_transform mask
    # Note: obs_map not yet worldmap. Needs transformation and mapping
    obs_map = np.absolute(np.float32(threshed)-1) * mask 
    
    
    Rover.vision_image[:,:,2] = threshed * 255 #b navigable
    Rover.vision_image[:,:,0] = obs_map * 255 #r obs
    #Rover.vision_image[:,:,1]  #g rock
         


    # Mapping the warped threshed image into the world map
    # Reflection and centering
    xpix, ypix = rover_coords(threshed)
    world_size = Rover.worldmap.shape[0] # init world_size so we can map to it
    scale = 2 * dst_size 
    # Scaling 10 by 10 pixels in front of robot to world size
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    
    obsxpix, obsypix = rover_coords(obs_map) 
    obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    
    Rover.worldmap[y_world, x_world,2] += 10
    Rover.worldmap[obs_y_world , obs_x_world , 0] +=1 
    
    dist, angles = to_polar_coords(xpix,ypix)
    Rover.nav_angles = angles 
    
    
    rock_map = find_rocks(warped, levels=(110,110,50))
    
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        rock_dist, rock_ang = to_polar_coords(rock_x,rock_y)
        rock_idx = np.argmin(rock_dist) #min dist rock pixel
        rock_xcen = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]
    
        Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
        Rover.vision_image[:,:,1] = rock_map * 255
    
    
    
    else:
        Rover.vision_image[:,:,1] = 0
       
    
 
    
    
    return Rover
