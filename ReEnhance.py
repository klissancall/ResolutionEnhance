import numpy as np # for doing most of our calculations
import matplotlib.pyplot as plt# for plotting
import cv2
import sys
import os

%matplotlib inline

%load_ext autoreload
%autoreload 2

x = []
#MPIf001_0r.rgb
for i in np.arange(1,101):
    if i < 10:
        fn = 'MPIf00' + str(i) + '_0r.rgb'
    elif i < 100:
        fn = 'MPIf0' + str(i) + '_0r.rgb'
    else:
        fn = 'MPIf' + str(i) + '_0r.rgb'
    #change the following path to the path of ur image files
    face_dir = '/Users/klissan/cs269/rgb/'
    img = plt.imread(face_dir+fn)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    a = np.asarray(gray)
    re_img = np.reshape(gray,(1, gray.shape[0]*gray.shape[1]))
    x.extend(re_img)
ref = np.reshape(x[0][:],(256,256))
for i in np.arange(2,101):
    img = np.reshape(x[i][:],(256,256))
    
print(x)
plt.imshow(gray)

print(np.array(x).shape)
ref = np.reshape(x[0][:],(256,256))
img = np.reshape(x[1][:],(256,256))
flow = np.array([])
corr = cv2.calcOpticalFlowFarneback(img, ref, None, 0.5, 3, 10, 10, 5, 1.1, 0)
corr = np.array(corr)
print(corr.shape)
#print(flow.shape)
#flow = corr
print(corr[:,:,0])
print(corr[:,:,1])
#plt.show(corr[:,:,0])