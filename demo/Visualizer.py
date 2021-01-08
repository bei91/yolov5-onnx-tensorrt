import cv2
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt

from classes import coco

class Visualizer():
    def __init__(self):
        self.color_list = self.gen_colors(coco)
    
    def gen_colors(self, classes):
        """
            generate unique hues for each class and convert to bgr
            classes -- list -- class names (80 for coco dataset)
            -> list
        """
        hsvs = []
        for x in range(len(classes)):
            hsvs.append([float(x) / len(classes), 1., 0.7])
        random.seed(1234)
        random.shuffle(hsvs)
        rgbs = []
        for hsv in hsvs:
            h, s, v = hsv
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgbs.append(rgb)

        bgrs = []
        for rgb in rgbs:
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            bgrs.append(bgr)
        return bgrs


    def draw_results(self, img, boxes, confs, classes):
        window_name = 'final results'
        cv2.namedWindow(window_name)
        overlay = img.copy()
        final = img.copy()
        print(classes)
        for box, conf, cls in zip(boxes, confs, classes):
            # draw rectangle
            x1, y1, x2, y2 = box
            conf = conf[0]
            cls_name = coco[cls]
            color = self.color_list[cls]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), thickness=1, lineType=cv2.LINE_AA)
            # draw text
            cv2.putText(overlay, '%s %f' % (cls_name, conf), org=(x1, int(y1-10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255))
        # cv2.addWeighted(overlay, 0.5, final, 1 - 0.5, 0, final)
        cv2.imshow(window_name, overlay)
        cv2.waitKey(0)

    def draw_results_batch(self, index, img, boxes, confs, classes):
        for box, conf, cls in zip(boxes, confs, classes):
            # draw rectangle
            x1, y1, x2, y2 = box
            conf = conf[0]
            cls_name = coco[cls]
            color = self.color_list[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), thickness=1, lineType=cv2.LINE_AA)
            # draw text
            cv2.putText(img, '%s %f' % (cls_name, conf), org=(x1, int(y1-10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255))
        # cv2.addWeighted(overlay, 0.5, final, 1 - 0.5, 0, final)
        cv2.namedWindow("res{}".format(index), cv2.WINDOW_NORMAL)
        cv2.imshow("res{}".format(index), img)
        cv2.waitKey(0)
