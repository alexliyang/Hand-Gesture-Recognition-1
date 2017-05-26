import cv2
import os

class WebCam:

	def __init__(self):
		self.camera = cv2.VideoCapture(0)

	def get_image(self):
		retval, im = self.camera.read()
 		return im

	def captureImg(self):
		camera_capture = self.get_image()
		return camera_capture

	def saveImg(self,img,filename):
		cv2.imwrite(filename,img)

	def captureVideo(self,no_frames,fps,filename,root="."):
		self.camera.set(5,fps)
		for i in range(no_frames):
			img = self.get_image()
			cv2.imwrite(root+filename+str(i)+".jpg",img)
			cv2.waitKey(15);

	def getFramesFromVideo(self, num_, prefix_, loc_=".", ext_=".jpg", dim_=(128, 128), start_=0, delay_=1):
		for i in range(num_):
			img = self.get_image()
			cv2.imshow("Capture", img)
			filepath = loc_ + prefix_ + "-" + str(start_ + i) + ext_
			img = cv2.resize(img, dim_)
			cv2.imwrite(filepath, img)
			cv2.waitKey(delay_)

	def showVideo(self):
		while 1:
			img = self.get_image()
			cv2.imshow("image", img)
			if cv2.waitKey(1) == 0x1b:
				break

	def captureImgFromVid(self):
		while 1:
			img = self.get_image()
			cv2.imshow("image", img)
			if cv2.waitKey(1) & 0xFF == ord('c'):
				camera_capture = self.get_image()
				return camera_capture


def main():
	w = WebCam()
	#w.showVideo()
	classLabel = "Grasp"
	rootFolder = "/home/suhit/cs726Project/" + classLabel + "/"

	if not os.path.exists(rootFolder):
		os.mkdir(rootFolder)
	# w.captureVideo(100, 60, "test", rootFolder)
	w.getFramesFromVideo(500, classLabel.lower(), rootFolder, dim_=(128, 96), start_=500, delay_=300)

if __name__ == '__main__':
	main()
