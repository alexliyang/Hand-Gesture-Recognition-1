import cv2
import os
import numpy as np

LOC = "/home/samiran/Desktop/suhit/Dataset-CS726/Custom/"


def preprocess(mat):
	
	r=mat.shape[0]
	c=mat.shape[1]
	xdif = r-80
	ydif = c-70
	if xdif==0 and ydif==0:
		return mat
	if xdif<0:
		for i in range(-xdif):
			mat = np.r_[mat,[np.zeros(c)]]
		r=mat.shape[0]
	elif xdif>0:
		mat = np.delete(mat,range(80,r),0)
		r=mat.shape[0]
	if ydif<0:
		for i in range(-ydif):
			mat = np.c_[mat, np.zeros(r)]
		c=mat.shape[1]
	elif ydif>0:
		mat = np.delete(mat,range(70,c),1)
		c=mat.shape[1]
	return mat

def load_data(loc):
	X=[]
	Y=np.array([])
	
	t1 = np.array([])
	t2 = np.array([])

	for root, dirs, files in os.walk(loc):
		
		for fn in files:
			im = cv2.imread(os.path.join(root,fn))
			gr_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			t1 = np.append(t1, gr_im.shape[0])
			t2 = np.append(t2, gr_im.shape[1])
			X += [gr_im]
			r = fn[0]
			posture = -1
			if 'o' in r:
				posture = 1
			if 'v' in r:
				posture = 2
			if 'g' in r:
				posture = 3
			if 's' in r:
				posture = 4
			if 'n' in r:
				posture = 5
			Y = np.append(Y, posture - 1)
	x = np.dstack(X)
	x = np.rollaxis(x,-1)

	return x,Y

if __name__ == "__main__":
	(X, Y) = load_data(LOC)

	print X.shape
	print Y

