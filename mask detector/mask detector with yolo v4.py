import numpy as np
import argparse
import time
import cv2
import tensorflow as tf
from tensorflow import keras
import os

ap=argparse.ArgumentParser()
# ap.add_argument("-i",'--image',required=True,help='path to input image')
# ap.add_argument("-y",'--yolo',required=True,help='base path to YOLO directory')
ap.add_argument("-c",'--confidence',type=float,default=0.5,help='minimum probability to filter weak detections')
ap.add_argument("-t",'--threshold',type=float,default=0.2,help='threshold when applying non-maxima suppression')
args=vars(ap.parse_args())

#labelsPath=os.path.sep.join([args['yolo'],'coco.names'])
labels=open('./coco.names').read().strip().split('\n')

np.random.seed(42)
colors=np.random.randint(0,255,size=(len(labels),3),dtype='uint8')

weightspath="./yolov4-tiny.weights"
configpath='./yolo4-tiny.cfg'

net=cv2.dnn.readNetFromDarknet(configpath,weightspath)
ln=net.getLayerNames()

ln=[ln[i-1] for i in net.getUnconnectedOutLayers()]
model=keras.models.load_model('./mask detector.h5')
vc=cv2.VideoCapture(0)
writer=None
(W,H)=(None,None)

if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False
while rval:
    cv2.imshow('test',frame)

    (rval,frame)=vc.read()
    if not rval:
        break
    if W is None or H is None:
        (H,W)=frame.shape[:2]
    masks_labels={0:'Mouth Mask 1',1:' Nose Mask',2:'Mask',3:'No Mask'}
    blob=cv2.dnn.blobFromImage(frame,1/255.,(280,280),swapRB=True,crop=False)
    net.setInput(blob)
    layerOutputs=net.forward(ln)
    boxes=[]
    confidences=[]
    classIDs=[]
    # loop over each of the layer outputs
    # print(len(layerOutputs))

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:

                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class ID
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)
                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in colors[classIDs[i]]]

            if str((labels[classIDs[i]]))=='person':
                cropped_img = frame[y:y + height, x:x + width]
                if cropped_img.size==0:
                    break
                cropped_img = cv2.resize(cropped_img, (300, 300))
                cropped_img = np.reshape(cropped_img, [1, 300, 300, 3]) / 255.
                # img = tf.expand_dims(crop_img, 0)

                pred = model.predict(cropped_img)
                pred = pred[0]
                if pred[2]>=max(pred) :
                    mask='{}:{:.4f}'.format(masks_labels[2],pred[2])
                    #text = "{}: {:.4f}".format(labels[classIDs[i]],confidences[i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

                    cv2.putText(frame, mask, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif pred[1]>=max(pred) :
                    mask='{}:{:.4f}'.format(masks_labels[1],pred[1])
                    #text = "{}: {:.4f}".format(labels[classIDs[i]],confidences[i])
                    cv2.putText(frame, mask, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif pred[3]>=max(pred) :
                    mask='{}:{:.4f}'.format(masks_labels[3],pred[3])
                    #text = "{}: {:.4f}".format(labels[classIDs[i]],confidences[i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

                    cv2.putText(frame, mask, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    mask='{}:{:.4f}'.format(masks_labels[0],pred[0])
                    #text = "{}: {:.4f}".format(labels[classIDs[i]],confidences[i])
                    cv2.putText(frame, mask, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break