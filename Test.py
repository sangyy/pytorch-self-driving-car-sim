from datetime import datetime
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import cv2
from PIL import Image
from io import BytesIO
from flask import Flask
import numpy as np
import eventlet
import socketio
import shutil
import base64
import os
import matplotlib.image as mpimg
print('Setting UP')

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from tensorflow.keras.models import load_model
# FOR REAL TIME COMMUNICATION BETWEEN CLIENT AND SERVER
sio = socketio.Server()
# FLASK IS A MICRO WEB FRAMEWORK WRITTEN IN PYTHON
app = Flask(__name__)  # '__main__'

maxSpeed = 10

transformations = transforms.Compose(
    [transforms.Lambda(lambda x: (x / 255.0))])


def preProcess(img):

    img = img[60:135, :, :]
    # img = img[65:-25, :, :]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    # img = cv2.resize(img, (320, 70))

    return img


global num
num = 0
sw = True


@sio.on('telemetry')
def telemetry(sid, data):
    global num, sw
    if data:
        speed = float(data['speed'])
        image_original = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image_original)
        image = preProcess(image)
        image = transformations(image)
        image = torch.Tensor(image)

        # image = np.array([image])
        # steering = float(model.predict(image))

        # print(image.shape)
        image = image.view(1, 3, 66, 200)
        # image = image.view(1, 3, 70, 320)

        image = Variable(image)
        # print(image.shape)
        steering = model(image).view(-1).data.numpy()[0]

        throttle = 1.0 - speed / maxSpeed
        print(f'{steering}, {throttle}, {speed}')
        sendControl(steering, throttle)
        # save frame
        if record_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(
                record_folder, '{}.jpg'.format(timestamp))
            num = num + 1
            # image = np.array(image)
            # image.save('{}.jpg'.format(image_filename))
            if num % 300 == 0:
                img = np.asarray(image_original)
                # img = img[60:135,:,:]
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

                # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

                # img = cv2.GaussianBlur(img,  (3, 3), 0)
                # img.save(image_filename)
                img = preProcess(img)
                # cv2.imwrite(image_filename, img)
                mpimg.imsave(image_filename, img)
            # print(image_filename)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    record_folder = ''
    # record_folder = '/Users/sangyy/Documents/beta_simulator_mac/dataset/testDriveRecord'
    # model = torch.load('model.h5')
    checkpoint = torch.load(
        'model.h5', map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    # cpu -> cpu或者gpu -> gpu:
    # checkpoint = torch.load('model.h5')
    # model = checkpoint['model']

    """
    2. cpu -> gpu 1

    torch.load('modelparameters.pth', map_location=lambda storage, loc: storage.cuda(1))
    3. gpu 1 -> gpu 0

    torch.load('modelparameters.pth', map_location={'cuda:1':'cuda:0'})
    4. gpu -> cpu

    torch.load('modelparameters.pth', map_location=lambda storage, loc: storage)
    """
    app = socketio.Middleware(sio, app)
    # LISTEN TO PORT 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
