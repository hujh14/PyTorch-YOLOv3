from __future__ import division

from models import Darknet
from utils.utils import *
from utils.datasets import ImageFolder

import os
import sys
import time
import argparse
from visualize import visualize

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

class YOLODetector:
    def __init__(self, opt):
        cuda = torch.cuda.is_available() and opt.use_cuda
        # Set up model
        model = Darknet(opt.config_path, img_size=opt.img_size)
        model.load_weights(opt.weights_path)
        if cuda:
            model.cuda()
        model.eval() # Set in evaluation mode

        self.model = model
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def detect(self, input_imgs):
        # Configure input
        input_imgs = Variable(input_imgs.type(self.Tensor))
        # Get detections
        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, 80, self.opt.conf_thres, self.opt.nms_thres)
            return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
    parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    opt = parser.parse_args()
    print(opt)

    detector = YOLODetector(opt)

    dataset = ImageFolder(opt.image_folder, img_size=opt.img_size)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index
    print ('\nPerforming object detection:')
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        start_time = time.time()
        detections = detector.detect(input_imgs)
        print ('\t+ Batch %d, Inference Time: %s' % (batch_i, time.time() - start_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    os.makedirs('output', exist_ok=True)
    classes = load_classes(opt.class_path) # Extracts class labels from file
    visualize(imgs, img_detections, classes, opt.img_size)