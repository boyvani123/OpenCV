import cv2
import numpy as np
import os

from PIL import Image

#the following are to do with this interactive notebook code
%matplotlib inline 
from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import pylab # this allows you to control figure size 
pylab.rcParams['figure.figsize'] = (10.0, 8.0) # this controls figure size in the notebook

img_scr_dir = "/content/drive/MyDrive/lab3/raw/Thảo Nguyên"
img_des_dir = "/content/drive/MyDrive/lab3/dataset" 

face_detector = cv2.CascadeClassifier('/content/drive/MyDrive/lab3/Cascades/haarcascade_frontalface_default.xml')

imagePaths = [os.path.join(img_scr_dir, f) for f in os.listdir(img_scr_dir)]
id = 1
for imagePath in imagePaths:
  #PIL_img = Image.open(imagePath).convert('L')
  #img_numpy = np.array(PIL_img, 'uint8')
  img=cv2.imread(imagePath)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #cv2.imwrite("/content/images/gray/thuat.jpg", img);
  faces = face_detector.detectMultiScale(gray, 1.3, 5)
  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2);
    cv2.imwrite(img_des_dir + "/user.5."+ str(id) +".jpg",  gray[y:y+h, x:x+w]);
    id = id +1
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    plt.imshow(cv2.cvtColor(gray[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
    plt.show()
