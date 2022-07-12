import cv2
import time
import numpy as np
import shutil
import os
import glob
from PIL import Image
import math
import warnings
np.seterr(divide = 'ignore')
warnings.simplefilter('ignore', np.RankWarning)
import matplotlib.pyplot as plt

#Finding out the Fractal Dimensions
def fractal_dimension(Z, threshold):
    # Only for 2d image
    assert(len(Z.shape) == 2)

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    #n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.floor(np.log(p)/np.log(2)))
    # Extract the exponent
    #n = int(np.log(n)/np.log(2))
    
    if n==0:
    	return 0.05
    else:
    # Build successive box sizes (from 2**n down to 2**1)
    	sizes = 2**np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    	counts = []
    	for size in sizes:
    		counts.append(boxcount(Z, size))
    #Fit the successive log(sizes) with log (counts)
    	coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    	if math.isnan(coeffs[0]):
    		coeffs[0] = 0
    	return -coeffs[0]

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a,n//a
      

#Making image Patches----------------------
temp = 0
count = 0
#your src path here
orig_path ="../New_Spires/Tiles1/"
#orig_path = "../Dataset_2/Test/Not_Temple/"
for f in os.listdir(orig_path):
    img = cv2.imread(orig_path+f)
    print(img.shape)
    image_list = [[0]*(img.shape[1]) for i in range(img.shape[0])]
    image_list = np.array(image_list)
    print(len(image_list))
    print(len(image_list[0]))
    step = 8
    rows,cols,_= img.shape
    print(orig_path + f)
    print(rows,cols)
    cnt = 0
    flag=0
    for i in range(0, rows):
    	for j in range(0, cols):
    		#print(i,j)
    		if (flag ==0):
    			if (j+step > cols and (i+step < rows or i+step ==rows)):
    				break
    			elif (j+step > cols and i+step > rows):
    				flag = 1
    			cnt=cnt+1
    			I = cv2.cvtColor(img[i:i+step,j:j+step],cv2.COLOR_BGR2GRAY)
    			gradx = cv2.Sobel(I, cv2.CV_64F, 1,0, ksize=3)
    			grady = cv2.Sobel(I, cv2.CV_64F, 0,1, ksize=3)
    			I_s = np.sqrt(gradx**2 + grady**2)
    			Sedges =cv2.Canny(I_s.astype(np.uint8),50,150)
    			ms = np.mean(I_s)
    			mcan = np.mean(Sedges)
    			try:
    				temp = round(float(fractal_dimension(Sedges, mcan) *127))
    				if temp > 0 or temp ==0:
    					temp = temp + 1
    				else:
    					temp = 0
    				image_list[(2*i + step) //2, (2*j + step)//2] = temp
    				if round(float(fractal_dimension(Sedges, mcan) *127)) < 0:
    					image_list[(2*i + step) //2, (2*j + step)//2] = 0
    			
    			except ZeroDivisionError:
    				print("Division By zero Appending 0\n")
    				image_list[(2*i + step) //2, (2*j + step)//2] = 0
    			except TypeError:
    				print("this is the failure")
    				print(round(float(fractal_dimension(Sedges, mcan) *127)))
    				with open(r'../Sample/Type_Errors.txt', 'a') as fp1:
    					fp1.write("\n"+str(f)+"\n")
    			except ValueError:
                		print("ValueError Failure")
                		print(round(float(fractal_dimension(Sedges, mcan) *127)))
    		
    			cnt = cnt+1
    print(image_list)
    np.save('../Sample/fractal_dims', image_list)
    img= np.load('../Sample/fractal_dims.npy')
    plt.imsave(orig_path+"fd_"+f, img, cmap='Greys')
    fdimg = cv2.imread(orig_path+"fd_"+f)
    mainimg = cv2.imread(orig_path+f)
    fdimg = cv2.bitwise_not(fdimg)
    result = cv2.bitwise_and(fdimg, mainimg)
    cv2.imwrite(orig_path+"seg_"+f, result)
    count = count + 1
    print('image Generated')
