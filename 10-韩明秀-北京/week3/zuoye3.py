import cv2
import os,sys
import numpy as np

def nearest_interpolation(im):
	h,w,_ = im.shape
	dest_im = np.zeros((800,800,3),dtype=np.uint8)
	scale_h = h*1.0/800
	scale_w = w*1.0/800
	for i in range(800):
		for j in range(800):
			src_h = int(scale_h * i+0.5)
			src_h = min(src_h,h-1)
			src_w = int(scale_w * j + 0.5)
			src_w = min(src_w,w-1)
			dest_im[i,j] = im[src_h,src_w]
	return dest_im

def bilinear_interpolation(im):
	h,w,_ = im.shape
	dest_im = np.zeros((800,800,3),dtype=np.uint8)
	scale_h = h*1.0/800
	scale_w = w*1.0/800
	for i in range(800):
		src_h = (i+0.5)*scale_h-0.5
		src_h_1 = int(src_h)
		src_h_2 = min(src_h_1+1,h-1)
		for j in range(800):
			src_w = (j+0.5)*scale_w-0.5
			src_w_1 = int(src_w)
			src_w_2 = min(src_w_1+1,w-1)
			mid_val_1 = im[src_h_1,src_w_1]*(src_w_2-src_w) + im[src_h_1,src_w_2]*(src_w-src_w_1)
			mid_val_2 = im[src_h_2,src_w_1]*(src_w_2-src_w) + im[src_h_2,src_w_2]*(src_w-src_w_1)
			dest_im[i,j] = mid_val_1 *(src_h_2-src_h) + mid_val_2 *(src_h-src_h_1)
	return dest_im

def histogram_equalization(im):
	gray_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	histogram = np.zeros((256),dtype=np.int32)
	h,w = gray_im.shape
	dest_im = np.zeros_like(gray_im,dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			histogram[gray_im[i,j]] += 1
	exist_gray_val_sets = np.where(histogram>0)
	prev_accu_gray_val = 0
	area = h * w
	for i in exist_gray_val_sets[0]:
		pix_val = i
		prev_accu_gray_val += histogram[pix_val]
		dest_val = int(prev_accu_gray_val*1.0/area * 256-1+0.5)
		dest_val = np.clip(dest_val,0,255)
		pix_val_coordinates = np.where(gray_im == pix_val)
		dest_im[pix_val_coordinates] = dest_val
	return dest_im

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("python zuoye3.py src_img")
	else:
		im = cv2.imread(sys.argv[1])
		dest_src = nearest_interpolation(im)
		cv2.imshow("nearest",dest_src)
		k = cv2.waitKey(0)&0xFF
		dest_src = bilinear_interpolation(im)
		cv2.imshow("bilinear",dest_src)
		cv2.waitKey(0)&0xFF
		dest_src = histogram_equalization(im)
		cv2.imshow("histogram",dest_src)
		cv2.waitKey(0)&0xFF
		cv2.destroyAllWindows()

