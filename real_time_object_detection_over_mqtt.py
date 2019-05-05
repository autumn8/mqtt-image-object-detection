from imutils.video import VideoStream
from imutils.video import FPS
import json
import numpy as np
import argparse
import imutils
import time
import cv2
from detection_classes import detection_classes
from datetime import datetime
from utils import intersects

import paho.mqtt.client as mqtt
import cv2
import numpy as np

#vars
last_incident_time = 0
person_already_detected = False
detection_zone_color = (0,255,0)
detection_zone_box = [0, 170, 185, 400]
confidence_threshold = 0.3

def publish_mqtt_message(topic):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S") 
    data = json.dumps({"time": current_time , "location": "camera1"})
    client.publish(topic, data)

def on_connect(client, userdata, flags, rc):
	print("Connected with result code "+str(rc))
	client.subscribe("camera/frame/#") 

frame = None;
def on_message(client, userdata, msg):
	global frame
	print(msg.topic)
	frame = cv2.imdecode(np.fromstring(msg.payload, dtype='uint8'), -1)	

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("192.168.8.103", 1883, 60)

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
		if confidence > confidence_threshold:			
			detection_id = int(detections[0, 0, i, 1])
			detected_obj_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = detected_obj_box.astype("int")
			(zone_start_x, zone_start_y, zone_end_x, zone_end_y) = detection_zone_box
			# if is person
			if detection_classes[detection_id] is 'person':	
				# if person is within defined motion zone
				if intersects(detection_zone_box, detected_obj_box): #person detected and they are inside zone
					detected_obj_box_color = (0,0,255)	
					if person_already_detected == False:
						current_time = int(time.time())
						time_since_last_detection  = current_time - last_incident_time
							
						print("time_since_last_detection")
						print(time_since_last_detection)
						if (time_since_last_detection > 30): #only send mqtt messages again if over a minute since last notification
							#current_image_folder = get_current_image_folder()      
							print('person detected. Send mqtt message!')
							print("publishing ")
							publish_mqtt_message('camera/person-detected')                        
							person_already_detected = True
							last_incident_time = int(time.time())    				 

				else:
					detected_obj_box_color = (0,0,0)
				#label = "{}: {:.2f}%".format(CLASSES[detection_id],	confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), detected_obj_box_color, 1)				
				#y = startY - 15 if startY - 15 > 15 else startY + 15
				#cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)	
			else:
				# todo check multiple frames before resetting this as precaution
				person_already_detected = False		
	#ret_code, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])     
	#client.publish("camera/image/inference",bytes(jpg_buffer),0) 			
	# show the output frame
	
	cv2.rectangle(frame, (zone_start_x, zone_start_y), (zone_end_x, zone_end_y), (0,255,0), 1)
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
