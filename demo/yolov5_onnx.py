"""Exports a YOLOv5 *.pt model to ONNX formats
Usage:
    $ python yolov5_onnx.py --weights coco_yolov5_80.pt --img 640 --batch batch-size
    onnx model input size must be src pth size
    when use batch_size inference and when yolov5 convert onnx , the batch_size must set batch-size value.
"""

import argparse

import torch
import torch.nn as nn

from networks.common import Conv
from networks.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import set_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default='coco_yolov5_80.pt', help='weights path')  # from yolov5/models/
    #onnx model input size must be src pth size
    parser.add_argument('-is', '--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('-bs', '--batch-size', type=int, default=4, help='batch size') #yolov5 the value is batch_size
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection
    print(img.shape)
    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        if isinstance(m, Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation
        # if isinstance(m, Detect):
        #    m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run
    try:
        import onnx
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '_640.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=True, opset_version=10, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'], export_params=True)
        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
