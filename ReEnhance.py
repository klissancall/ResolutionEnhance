import numpy as np # for doing most of our calculations
import matplotlib.pyplot as plt# for plotting
import cv2
import sys
import os
import pyflow


x = []
#MPIf001_0r.rgb
for i in np.arange(1,101):
    if i < 10:
        fn = 'MPIf00' + str(i) + '_0r.rgb'
    elif i < 100:
        fn = 'MPIf0' + str(i) + '_0r.rgb'
    else:
        fn = 'MPIf' + str(i) + '_0r.rgb'
    face_dir = '/Users/klissan/cs269/rgb/'
    img = plt.imread(face_dir+fn)
#img = cv2.imread(face_dir)
#img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    a = np.asarray(gray)
    re_img = np.reshape(gray,(1, gray.shape[0]*gray.shape[1]))
    x.extend(re_img)
ref = np.reshape(x[0][:],(256,256))
    
print(x)
plt.imshow(gray)


print(np.array(x).shape)
ref = np.reshape(x[0][:],(256,256,1))
img = np.reshape(x[1][:],(256,256,1))
flow = np.array([])
#corr = cv2.calcOpticalFlowFarneback(img, ref, None, 0.5, 3, 10, 10, 5, 1.1, 0)
#corr = np.array(corr)
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

u, v, im2W = pyflow.coarse2fine_flow(
    img, ref, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
print(flow.shape)
#print(flow.shape)
#flow = corr
print(flow[:,:,0])
print(flow[:,:,1])
plt.imshow(flow[:,:,0])
plt.imshow(flow[:,:,1])