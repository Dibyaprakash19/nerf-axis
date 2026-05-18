import cv2
import numpy as np
import glob
import os

def calibrate_camera():
    CHECKERBOARD = (8, 5)
    
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    objpoints = []
    imgpoints = []
    
    print("Loading calibration images from 'calib' directory...")
    images = glob.glob('calib/*.jpg')
    
    if not images:
        print("No images found in 'calib' folder! Please capture checkerboard patterns.")
        return
        
    print(f"Found {len(images)} images.")
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Optional: draw and display corners
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(50)
            
    cv2.destroyAllWindows()
    
    if not objpoints:
        print("Failed to find checkerboard corners in any images.")
        return
        
    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )
    
    print("Camera Matrix (Intrinsic):")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)
    
    # Save the calibration
    np.savez("calibration.npz", mtx=mtx, dist=dist)
    print("Calibration saved to calibration.npz")
    
if __name__ == "__main__":
    if not os.path.exists('calib'):
        os.makedirs('calib')
        print("Created 'calib' directory. Please place checkerboard images there and run again.")
    else:
        calibrate_camera()
