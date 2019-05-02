from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

import paho.mqtt.client as mqtt
import cv2
import numpy as np

def on_connect(client, userdata, flags, rc):
	print("Connected with result code "+str(rc))
	client.subscribe("camera/image") 

frame = None;
def on_message(client, userdata, msg):
	global frame
	print(msg.topic)
	frame = cv2.imdecode(np.fromstring(msg.payload, dtype='uint8'), -1)	

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("192.168.8.103", 1883, 60)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("MobileNetSSD.prototxt.txt", "MobileNetSSD.caffemodel")
fps = FPS().start()


while True:
	client.loop()
	if frame is None:
		continue

	frame = imutils.resize(frame, width=400)	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),	0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()

	for i in np.arange(0, detections.shape[2]):		
		confidence = detections[0, 0, i, 2]		
		if confidence > 0.3:			
			detection_id = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			if (CLASSES[detection_id] is 'person'):				
				#label = "{}: {:.2f}%".format(CLASSES[detection_id],	confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[detection_id], 2)				
				#y = startY - 15 if startY - 15 > 15 else startY + 15
				#cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[detection_id], 2)			
	#ret_code, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])     
	#client.publish("camera/image/inference",bytes(jpg_buffer),0) 			
	# show the output frame
	cv2.rectangle(frame, (0, 150), (400, 400), (0,255,0), 2)
	cv2.imshow("Frame", frame)
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
