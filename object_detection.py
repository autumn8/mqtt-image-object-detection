# USAGE
# python pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2
#import paho.mqtt.client as mqtt

""" def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc)) 
    client.subscribe("$SYS/#")

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
mqtt_broker = '192.168.8.101'
mqtt_broker_port = '1883'
client.connect(mqtt_broker, mqtt_broker_port, 60)
print("connecting to broker ",broker)
client.subscribe("house/bulb1")#subscribe
"""

is_person_detected = False

def classify_frame(net, inputQueue, outputQueue):
    while True:        
        if not inputQueue.empty():
            frame = inputQueue.get()
            frame = cv2.resize(frame, (300, 300))
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            outputQueue.put(detections)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

inputQueue = Queue(maxsize=100)
outputQueue = Queue(maxsize=100)
detections = None

p = Process(target=classify_frame, args=(net, inputQueue, outputQueue,))
p.daemon = True
p.start()

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True, framerate=2, resolution=(300,300)).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # mqttc.loop()
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (fH, fW) = frame.shape[:2]
    inputQueue.put(frame)

    if not outputQueue.empty():
        detections = outputQueue.get()

    if detections is not None:
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            detection_id = int(detections[0, 0, i, 1])
            if (CLASSES[detection_id] is 'person'):
                if is_person_detected is not True:
                    print('person detected. Send mqtt message!')
                    print("publishing ")
                    # client.publish("camera/person_detected", true)
                    is_person_detected = True
                dims = np.array([fW, fH, fW, fH])
                box = detections[0, 0, i, 3:7] * dims
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY),(endX, endY), COLORS[detection_id], 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
