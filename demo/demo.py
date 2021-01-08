import os
import random

import cv2
import argparse

from Processor import Processor
from Visualizer import Visualizer

def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='coco_yolov5_16.trt', help='tensorrt engine file', required=False)
    parser.add_argument('-i', '--image', default='imgs', help='image file path', required=False)
    args = parser.parse_args()
    return {'model': args.model, 'image': args.image}

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.rectangle(img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [0, 255, 255],
            #thickness=tf,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
def main():
    # parse arguments
    args = cli()
    # setup processor and visualizer
    processor = Processor(model=args['model'])
    visualizer = Visualizer()

    # fetch input
    print('image arg', args['image'])
    # img = cv2.imread('inputs/{}'.format(args['image']))
    input_image_paths = []
    folder_path = args['image']
    if os.path.isdir(folder_path):
        ls = os.listdir(folder_path)
        for file_name in sorted(ls, key=lambda x: str(x.split('.jpg')[0])):
            input_image_paths.append(os.path.join(folder_path, file_name))
    for input_image_path in input_image_paths:
        img = cv2.imread(input_image_path)

        # inference
        output = processor.detect(img)

        # final results
        boxes, confs, classes = processor.post_process(output, conf_thres=0.3, iou_thres=0.4, origin_w=img.shape[1], origin_h=img.shape[0])
        visualizer.draw_results(img, boxes, confs, classes)


if __name__ == '__main__':
    main()   
