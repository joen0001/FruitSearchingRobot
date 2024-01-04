from cmath import inf
import os 
import time

import cmd_printer
import numpy as np
import torch
from args import args
from res18_skip import Resnet18Skip
from torchvision import transforms
import cv2

class Detector:
    def __init__(self, ckpt, use_gpu=False):
        self.args = args
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',path = ckpt)
        self.model.conf = 0.8
        if torch.cuda.torch.cuda.device_count() > 0 and use_gpu:
            self.use_gpu = True
            self.model = self.model.cuda()
        else:
            self.use_gpu = False
        #self.load_weights(ckpt)
        self.model = self.model.eval()
        cmd_printer.divider(text="warning")
        print('This detector uses "RGB" input convention by default')
        print('If you are using Opencv, the image is likely to be in "BRG"!!!')
        cmd_printer.divider()
        self.colour_code = [(237, 26, 26), (222, 185, 53), (250, 139, 2), (83, 201, 146), (219, 24, 76), (0, 0, 255)]
        self.color = (255, 0, 0)
        self.legend = ['apple', 'lemon', 'orange', 'pear', 'strawberry']

    def detect_single_image(self, np_img):
        # path = "mult_fruit.jpg"
        # image = cv2.imread(path)    
        # torch_img = self.np_img2torch(np_img)
        tick = time.time()
        with torch.no_grad():
            pred = self.model.forward(np_img)
        dt = time.time() - tick
        colour_map = self.drawBoundingBoxes(np_img,pred)
        print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r") 
        out = pred.xyxy[0].numpy()
        return out, colour_map

    def drawBoundingBoxes(self, imageData, inferenceResults):
        """
        Draw bounding boxes on an image.
        imageData: image data in numpy array format
        imageOutputPath: output image file path
        inferenceResults: inference results array off object (l,t,w,h)
        colorMap: Bounding box color candidates, list of RGB tuples.
        """
        if range(len(inferenceResults.xyxy[0])) == range (0,0):
            output = 0
        else:
            for i in range(len(inferenceResults.xyxy[0])):
                left = int(inferenceResults.xyxy[0][i][0])
                top = int(inferenceResults.xyxy[0][i][3])
                right = int(inferenceResults.xyxy[0][i][2])
                bottom = int(inferenceResults.xyxy[0][i][1])
                label = self.legend[int(inferenceResults.xyxy[0][i][5])]

                # print(left.shape)
                imgHeight, imgWidth, _ = imageData.shape
                thick = int((imgHeight + imgWidth) // 400)

                output = cv2.rectangle(imageData,(left, top), (right, bottom), self.colour_code[int(inferenceResults.xyxy[0][i][5])], thick)
                output = cv2.putText(imageData, str(label), (left, top+int(4e-1*(right-left))), 0, 2e-2 * (right-left), self.colour_code[int(inferenceResults.xyxy[0][i][5])], thick)
        return output

    def visualise_output(self, nn_output):
       # r = np.zeros_like(nn_output).astype(np.uint8)
       # g = np.zeros_like(nn_output).astype(np.uint8)
       # b = np.zeros_like(nn_output).astype(np.uint8)
       # for class_idx in range(0, self.args.n_classes + 1):
       #     idx = nn_output == class_idx
       #     r[idx] = self.colour_code[class_idx, 0]
       #     g[idx] = self.colour_code[class_idx, 1]
       #     b[idx] = self.colour_code[class_idx, 2]
       # colour_map = np.stack([r, g, b], axis=2)
       # colour_map = cv2.resize(colour_map, (320, 240), cv2.INTER_NEAREST)
       # w, h = 10, 10
       # pt = (10, 160)
       # pad = 5
       # labels = ['apple', 'lemon', 'pear', 'orange', 'strawberry']
       # font = cv2.FONT_HERSHEY_SIMPLEX 
       # for i in range(1, self.args.n_classes + 1):
       #     c = self.colour_code[i]
       #     colour_map = cv2.rectangle(colour_map, pt, (pt[0]+w, pt[1]+h),
       #                     (int(c[0]), int(c[1]), int(c[2])), thickness=-1)
       #     colour_map  = cv2.putText(colour_map, labels[i-1],
       #     (pt[0]+w+pad, pt[1]+h-1), font, 0.4, (0, 0, 0))
       #     pt = (pt[0], pt[1]+h+pad)
       #     
        
        colour_map = cv2.cvtColor(nn_output.imgs[0], cv2.COLOR_BGR2RGB) # Because of OpenCV reading images as BGR
        return colour_map

    def load_weights(self, ckpt_path):
        ckpt_exists = os.path.exists(ckpt_path)
        if ckpt_exists:
            ckpt = torch.load(ckpt_path,
                              map_location=lambda storage, loc: storage)

            self.model.load_state_dict(ckpt['model'].state_dict())
        else:
            print(f'checkpoint not found, weights are randomly initialised')
            
    @staticmethod
    def np_img2torch(np_img, use_gpu=False, _size=(192, 256)):
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=_size),
                                        # transforms.ColorJitter(brightness=0.4, contrast=0.3,
                                        #                         saturation=0.3, hue=0.05),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        img = preprocess(np_img)
        img = img.unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        return img
