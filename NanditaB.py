import cv2
import sys
img=cv2.imread('Chess.jpg')
# gray=cv2.imread('Chess.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('Chess image',img)
# cv2.imshow('Chess image',gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#print(gray)
img=cv2.GaussianBlur(img,(3,3),0)

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Chess image',gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(gray)
window_name=('Optical Label')
scale=1
delta=0
ddepth=cv2.CV_16S

grad_x=cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta= delta, borderType=cv2.BORDER_DEFAULT)
grad_y=cv2.Sobel(gray, ddepth, 0, 1, ksize=5, scale=scale, delta= delta, borderType=cv2.BORDER_DEFAULT)

abs_grad_x=cv2.convertScaleAbs(grad_x)
abs_grad_y=cv2.convertScaleAbs(grad_y)

grad=cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0.5, 0)
cv2.imshow('Chess final image',grad)
cv2.waitKey(0)
cv2.destroyAllWindows()