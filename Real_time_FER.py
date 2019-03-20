import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from torch import optim
from PIL import Image
import os
import torch.nn as nn
from torch.autograd import Variable
import cv2
import imutils
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 7

class new_model(nn.Module):
    def __init__(self):
        super(new_model, self).__init__()
        self.features = nn.Sequential(*list(vgg.features.children()))
        input_dim = 10*10*512
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5))
        self.fc4 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.5))
        self.fc5 = nn.Sequential(nn.Linear(64, num_classes))
        
    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        return out

vgg = models.vgg16(pretrained=False)
FEN = new_model()
FEN.load_state_dict(torch.load('VGG-Adam-LR-3-5-Aug.pth', map_location='cpu'))
FEN.eval()



# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EMOTIONS = ["angry" ,"disgust","fear", "happy", "neutral", "sad", "surprised"]


# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
pred = np.array([0,0,0,0,0,0,0])
label = 'a'
fX = 0
fY=0
fW=0
fH=0
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (350, 350))
        #print(roi.shape)
        roi = roi.astype("float") / 255.0
        #roi = np.array(roi).transpose(2,0,1)
        inp = [roi,roi,roi]
        inp = np.expand_dims(inp, axis=0)
        X_test = Variable(torch.tensor(inp)).to(device).float()
        pred=FEN(X_test)

        pred = pred[0]
        emotion = pred.argmax().item()
        label = EMOTIONS[emotion]
        #print(label)
    #cv2.waitKey(0)

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, pred)):
                # construct the label text
	    text = "{}: {:.2f}%".format(emotion, prob.item() * 100)
	    w = int(prob.item() * 300)
	    cv2.rectangle(canvas, (7, (i * 35) + 5),
	    (w, (i * 35) + 35), (0, 0, 255), -1)
	    cv2.putText(canvas, text, (10, (i * 35) + 23),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
	    (255, 255, 255), 2)
    cv2.putText(frameClone, label, (fX, fY - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                  (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    #cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()