# my_repository
1.Develop a program to display grayscale image using read and write operation

In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

To convert an image to grayscale in any of the Microsoft Office suite apps, right-click it and select Format Picture from the context menu . This will open an image editing panel on the right. Go to the Picture tab (the very last one). Expand the Picture Color options, and click the little dropdown next to the Presets for Color Saturation.

import cv2 image=cv2.imread('original.jpg') grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) cv2.imwrite('original.jpg',image) cv2.imshow("org",image) cv2.imshow("gimg",grey_image) cv2.waitKey(0) cv2.destroyAllWindows() output:

![image](https://user-images.githubusercontent.com/75006493/104893410-68ea8380-5999-11eb-8362-b3670f32e9cf.png)
