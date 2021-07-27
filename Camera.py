import numpy as np
import cv2
import glob
import pickle

# Prepare the object points as (0,0,0), (1,0,0).....(8,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Making arrays to store image points and object points from all the images
objpoints = []      # 3-D points in the real world
imgpoints = []      # 2-D points in the image plane

# Making a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Iterate through the list to search for the corners
for idx, image in enumerate (images):
	img = cv2.imread(image)

	#Grayscale conversion
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the Chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

	# If found
	if ret == True:
		print('Corners found in', image)
		objpoints.append(objp)
		imgpoints.append(corners)

		# Draw the chess corners
		cv2.drawChessboardCorners(img, (9,6), corners, ret)
		write_name = 'corners_found'+str(1+ idx)+'.jpg'
		cv2.imwrite(write_name, img)

# Load image
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])


# Camera Calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Saving the Camera Calibration results for later use (Dictionary)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./calibration_pickle.p","wb"))





