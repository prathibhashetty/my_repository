# my_repository
1.Develop a program to display grayscale image using read and write operation

In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

To convert an image to grayscale in any of the Microsoft Office suite apps, right-click it and select Format Picture from the context menu . This will open an image editing panel on the right. Go to the Picture tab (the very last one). Expand the Picture Color options, and click the little dropdown next to the Presets for Color Saturation.

import cv2 image=cv2.imread('original.jpg') grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) cv2.imwrite('original.jpg',image) cv2.imshow("org",image) cv2.imshow("gimg",grey_image) cv2.waitKey(0) cv2.destroyAllWindows() output:

![image](https://user-images.githubusercontent.com/75006493/104893410-68ea8380-5999-11eb-8362-b3670f32e9cf.png)

2.Develop a program to perform linear transformation on image.

Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.

Image scaling is a computer graphics process that increases or decreases the size of a digital image. An image can be scaled explicitly with an image viewer or editing software, or it can be done automatically by a program to fit an image into a differently sized area.

Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. The input to an image rotation routine is an image, the rotation angle θ, and a point about which rotation is done.

Scaling: import cv2 import numpy as np src=cv2.imread('original.jpg',1) img=cv2.imshow('original.jpg',src) scale_p=500 width=int(src.shape[1]*scale_p/100) height=int(src.shape[0]*scale_p/100) dsize=(width,height) result=cv2.resize(src,dsize) cv2.imwrite('scaling.jpg',result) cv2.waitKey(0) output:

![image](https://user-images.githubusercontent.com/75006493/104893799-e3b39e80-5999-11eb-9c87-6967c6ad1fb5.png)
![image](https://user-images.githubusercontent.com/75006493/104893986-207f9580-599a-11eb-8ede-4e5977902877.png)

rotating: import cv2 import numpy as np src=cv2.imread('original.jpg') img=cv2.imshow('original.jpg',src) windowsname='image' image=cv2.rotate(src,cv2.ROTATE_90_CLOCKWISE) cv2.imshow(windowsname,image) c.waitKey(0) output

![image](https://user-images.githubusercontent.com/75006493/104894248-748a7a00-599a-11eb-9b7b-fb046c9fbaa4.png)
![image](https://user-images.githubusercontent.com/75006493/104894365-9dab0a80-599a-11eb-99ca-8636cdb3f9a1.png)

3.Create a program to find sum and mean of a set of image. In digital image processing, the sum of absolute differences (SAD) is a measure of the similarity between image blocks. It is calculated by taking the absolute difference between each pixel in the original block and the corresponding pixel in the block being used for comparison

Mean is most basic of all statistical measure. Means are often used in geometry and analysis; a wide range of means have been developed for these purposes. In contest of image processing filtering using mean is classified as spatial filtering and used for noise reduction. import cv2 import os path='C:\picture\images' imgs=[] dirs=os.listdir(path) for file in dirs: fpat=path+"\"+file imgs.append(cv2.imread(fpat)) i=0 sum_img=[] for sum_img in imgs: read_imgs=imgs[i] sum_img=sum_img+read_imgs #cv2.imshow(dirs[i],imgs[i]) i=i+1 print(i) cv2.imshow('sum',sum_img) print(sum_img) cv2.imshow('mean',sum_img/i) mean=(sum_img/i) print(mean) cv2.waitKey() cv2.destroyAllwindows() output

![image](https://user-images.githubusercontent.com/75006493/104894603-eb277780-599a-11eb-961b-040747a89be9.png)

4.Develop a program to convert image to binary image and gray scale. Binary images are images whose pixels have only two possible intensity values. Numerically, the two values are often 0 for black, and either 1 or 255 for white. The main reason binary images are particularly useful in the field of Image Processing is because they allow easy separation of an object from the background.

In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

import cv2 img = cv2.imread('original.jpg') cv2.imshow('Input',img) cv2.waitKey(0) grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) cv2.imshow('Grayscaleimage',grayimg) cv2.waitKey(0) ret, bw_img = cv2.threshold(img,127,255, cv2.THRESH_BINARY) cv2.imshow("Binary Image",bw_img) cv2.waitKey(0) cv2.destroyAllWindows() output

![image](https://user-images.githubusercontent.com/75006493/104894842-32156d00-599b-11eb-8816-c354c494de1e.png)
