import pyautogui as keypresser
import time
import threading
from subprocess import call

class myThread (threading.Thread):
	def __init__(self, function, arg1, arg2, seconds):
		threading.Thread.__init__(self)
		self.function = function
		self.arg1 = arg1
		self.arg2 = arg2
		self.seconds = seconds
	def run(self):
		time.sleep(self.seconds)
		decider(self.function, self.arg1, self.arg2)
def decider(function, arg1, arg2):
	if(function=="singleKey"):
		Action.singleKeyPress(arg1)
	else:
		Action.doubleKeyPress(arg1,arg2)



class Action:

	@staticmethod
	def singleKeyPress(key):
		keypresser.press(key)

	@staticmethod
	def doubleKeyPress(key1,key2):
		keypresser.keyDown(key1)
		keypresser.press(key2)
		keypresser.keyUp(key1)
	@staticmethod
	def singleKeyPressWithAsyncTimer(key,seconds):
		myThread("singleKey", key, "doesnt matter", seconds).start()

	@staticmethod
	def doubleKeyPressWithAsyncTimer(key1,key2,seconds):
		myThread("doubleKey", key1, key2, seconds).start()

	@staticmethod
	def play():
		call(["rhythmbox-client", "--play"])

	@staticmethod
	def pause():
		call(["rhythmbox-client", "--pause"])
