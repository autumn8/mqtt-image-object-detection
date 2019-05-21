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
from utils import intersects, get_current_image_folder

import paho.mqtt.client as mqtt
import cv2
import numpy as np

#vars
last_incident_time = 0
min_time_since_last_incident = 30
person_already_detected = False
detection_zone_color = (0,255,0)
confidence_threshold = 0.3
object_detection_enabled = True
current_image_folder = None
count = 0
cameras = {}

def on_message(client, userdata, msg):		
	global object_detection_enabled	
	global cameras
	global frame

	if 'camera/connected/' in msg.topic:
		print('camera connected message')		
		payload = msg.payload.decode("utf-8")  
		print(payload)		
		if payload == 'True':
			print('camera connected true, sub to update')
			print(msg.topic.split('/')[-1])			
			camera_name = msg.topic.split('/')[-1]
			print(camera_name)
			client.subscribe("camera/settingsupdate/" + camera_name) 
					
	elif 'camera/settingsupdate/' in msg.topic:
		print('settings update received')
		payload = msg.payload.decode("utf-8")
		camera_name = msg.topic.split('/')[-1]
		camera_settings = json.loads(payload)
		print(camera_settings)
		cameras[camera_name] = camera_settings
		client.subscribe("camera/frame/" + camera_name) 
		# add frame subscription here rather

	elif 'camera/frame/' in msg.topic:		
		camera_name = msg.topic.split('/')[-1]		
		# TODO: add frame data and camera name to frame object		
		frame = {}
		frame['data'] = cv2.imdecode(np.fromstring(msg.payload, dtype='uint8'), -1)	
		frame['camera'] = cameras[camera_name]		

def publish_mqtt_message(topic):
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S") 
	data = json.dumps({"time": current_time , "location": "camera1"})
	client.publish(topic, data)

def on_connect(client, userdata, flags, rc):
	print("Connected with result code "+str(rc))
	client.subscribe("camera/connected/#") 	

frame = None;	

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("192.168.8.104", 1883, 60)

net = cv2.dnn.readNetFromCaffe("MobileNetSSD.prototxt.txt", "MobileNetSSD.caffemodel")
fps = FPS().start()

while True:
	client.loop()	
	if frame is None:
		continue	
	frameData = frame['data']
	camera_settings = frame['camera']
	frameData = imutils.resize(frameData, width=400)	
	(h, w) = frameData.shape[:2]

	if camera_settings['isDetectionEnabled'] == True:		
		blob = cv2.dnn.blobFromImage(cv2.resize(frameData, (300, 300)),	0.007843, (300, 300), 127.5)
		net.setInput(blob)
		detections = net.forward()	
		for i in np.arange(0, detections.shape[2]):		
			confidence = detections[0, 0, i, 2]		
			if confidence > confidence_threshold:			
				detection_id = int(detections[0, 0, i, 1])
				detected_obj_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = detected_obj_box.astype("int")
				print(camera_settings['zoneWidth'] * w )
				zone_start_x = int(camera_settings['zoneX'] * w)
				zone_start_y = int(camera_settings['zoneY'] * h)
				zone_end_x = int(camera_settings['zoneWidth'] * w)
				zone_end_y = int(camera_settings['zoneHeight'] * h)
				detection_zone_box = [zone_start_x, zone_start_y, zone_end_x, zone_end_y]
				#(zone_start_x, zone_start_y, zone_end_x, zone_end_y) = detection_zone_box				
				# if is person
				if detection_classes[detection_id] is 'person':	
					# if person is within defined motion zone
					if intersects(detection_zone_box, detected_obj_box): #person is inside detection zone
						detected_obj_box_color = (0,0,255)	
						if person_already_detected is False:  #only send notification at beginning of new detection incident
							current_time = int(time.time())
							time_since_last_detection  = current_time - last_incident_time
							if (time_since_last_detection > 60): 
								current_image_folder = get_current_image_folder()      
								print('person detected. Send mqtt message!')
								print("publishing ")
								publish_mqtt_message('camera/person-detected')                        
								person_already_detected = True	
						last_incident_time = int(time.time())    				 

					else:
						detected_obj_box_color = (0,0,0)
					#label = "{}: {:.2f}%".format(CLASSES[detection_id],	confidence * 100)
					#cv2.imwrite(current_image_folder +  '/frame%d.jpg' %count, frameData) 
					cv2.rectangle(frameData, (startX, startY), (endX, endY), detected_obj_box_color, 1)						
					count = count + 1 			
					#y = startY - 15 if startY - 15 > 15 else startY + 15
					#cv2.putText(frameData, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)	
				else:
					# todo check multiple frames before resetting this as precaution
					person_already_detected = False		
			
			#ret_code, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])     
			#client.publish("camera/image/inference",bytes(jpg_buffer),0) 			
			# show the output frame
	
			cv2.rectangle(frameData, (zone_start_x, zone_start_y), (zone_end_x, zone_end_y), (0,255,0), 1)

	cv2.imshow("Frame", frameData)
	
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
