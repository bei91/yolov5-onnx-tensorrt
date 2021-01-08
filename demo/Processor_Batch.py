import cv2 
import sys
import os 
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import math
import time
import torch

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
class Processor():
    def __init__(self, model):
        self.INPUT_W = 608
        self.INPUT_H = 608
        # load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        TRTbin = '{0}'.format(model)
        print('trtbin', TRTbin)
        with open(TRTbin, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        stream = cuda.Stream()

        # allocate memory
        inputs, outputs, bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream

        # post processing config
        self.cls_nums = 80
        self.filters = (self.cls_nums + 5) * 3
        self.output_filter = self.cls_nums + 5
        self.output_shapes = [
            (1, 3, 80, 80, self.output_filter),
            (1, 3, 40, 40, self.output_filter),
            (1, 3, 20, 20, self.output_filter)
        ]

        self.strides = np.array([8., 16., 32.])
        anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ])
        self.nl = len(anchors)
        self.nc = 80  # classes
        self.no = self.nc + 5  # outputs per anchor
        self.na = len(anchors[0])
        a = anchors.copy().astype(np.float32)
        a = a.reshape(self.nl, -1, 2)
        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)

    def detect(self, img_batch):
        patch_ratio = 4  # when tensorrt7 is 4, version < tensorrt7 is 1
        resized, image_raw_batch = self.preprocess_image(img_batch)
        outputs = self.inference(resized)
        # reshape from flat to (1, 3, x, y, 85)
        reshaped_batch_size = []
        reshaped_batch_size_out = []
        for i in range(0, len(outputs)):
            for j in range(0, len(outputs[i]), int(len(outputs[i]) / patch_ratio * len(img_batch))):  # batch size is 4
                if j * patch_ratio < int(len(outputs[i])):
                    out = outputs[i][j:(int(len(outputs[i]) / patch_ratio * len(img_batch)) + j)]
                    out = out.reshape(self.output_shapes[i])
                    reshaped_batch_size.append(out)
        for k in range(0, len(img_batch)):
            out = []
            out.append(reshaped_batch_size[k])
            out.append(reshaped_batch_size[k + len(img_batch)])
            out.append(reshaped_batch_size[k + len(img_batch)*2])
            reshaped = []
            for output, shape in zip(out, self.output_shapes):
                reshaped.append(output.reshape(shape))
            reshaped_batch_size_out.append(reshaped)
        # return reshaped, image_raw
        return reshaped_batch_size_out, image_raw_batch

    def preprocess_image(self, image_raw_batch):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            image_raw: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        imgs_infer = []
        image_raw_batch_size = []
        for img in image_raw_batch:
            image_raw = cv2.imread(img)
            image_raw_batch_size.append(image_raw)
            h, w, c = image_raw.shape
            #image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            image = image_raw[:, :, ::-1]
            # Calculate widht and height and paddings
            r_w = self.INPUT_W / w
            r_h = self.INPUT_H / h
            if r_h > r_w:
                tw = self.INPUT_W
                th = int(r_w * h)
                tx1 = tx2 = 0
                ty1 = int((self.INPUT_H - th) / 2)
                ty2 = self.INPUT_H - th - ty1
            else:
                tw = int(r_h * w)
                th = self.INPUT_H
                tx1 = int((self.INPUT_W - tw) / 2)
                tx2 = self.INPUT_W - tw - tx1
                ty1 = ty2 = 0
            # Resize the image with long side while maintaining ratio
            image = cv2.resize(image, (tw, th))
            # Pad the short side with (128,128,128)
            image = cv2.copyMakeBorder(
                image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
            )
            image = image.astype(np.float32)
            # Normalize to [0,1]
            image /= 255.0
            # HWC to CHW format:
            image = np.transpose(image, [2, 0, 1])
            # CHW to NCHW format
            image = np.expand_dims(image, axis=0)
            # Convert the image to row-major order, also known as "C order":
            #image = np.ascontiguousarray(image)
            imgs_infer.append(image)
        if len(image_raw_batch) == 1:
            images_batch = imgs_infer[0]
        else:
            images_batch = imgs_infer[0]
            for i in range(len(imgs_infer) - 1):
                images_batch = np.concatenate((images_batch, imgs_infer[i + 1]))
        image = np.ascontiguousarray(images_batch, dtype=np.float32)
        return image, image_raw_batch_size

    def inference(self, img):
        # copy img to input memory
        self.inputs[0].host = np.ravel(img)
        # transfer data to the gpu
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

        # run inference
        start = time.time()
        self.context.execute_async(batch_size=4,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # synchronize stream
        self.stream.synchronize()
        end = time.time()
        #print(len(self.outputs)) #3
        print('execution time:', end - start)
        return [out.host for out in self.outputs]

    def extract_object_grids(self, output):
        """
        Extract objectness grid 
        (how likely a box is to contain the center of a bounding box)
        Returns:
            object_grids: list of tensors (1, 3, nx, ny, 1)
        """
        object_grids = []
        for out in output:
            probs = self.sigmoid_v(out[..., 4:5])
            object_grids.append(probs)
        return object_grids

    def extract_class_grids(self, output):
        """
        Extracts class probabilities
        (the most likely class of a given tile)
        Returns:
            class_grids: array len 3 of tensors ( 1, 3, nx, ny, 80)
        """
        class_grids = []
        for out in output:
            object_probs = self.sigmoid_v(out[..., 4:5])
            class_probs = self.sigmoid_v(out[..., 5:])
            obj_class_probs = class_probs * object_probs
            class_grids.append(obj_class_probs)
        return class_grids

    def extract_boxes(self, output, conf_thres=0.3):
        """
        Extracts boxes (xywh) -> (x1, y1, x2, y2)
        """
        scaled = []
        grids = []
        for out in output:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

            out[..., 5:] = out[..., 4:5] * out[..., 5:]
            out = out.reshape((1, 3 * width * height, self.output_filter))
            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        boxes = self.xywh2xyxy(pred[:, :4])
        return boxes

    def post_process(self, outputs, conf_thres=0.3, iou_thres=0.3,  image_raw_batch=None):
        """
        Transforms raw output into boxes, confs, classes
        Applies NMS thresholding on bounding boxes and confs
        Parameters:
            output: raw output tensor
        Returns:
            boxes: x1,y1,x2,y2 tensor (dets, 4)
            confs: class * obj prob tensor (dets, 1) 
            classes: class type tensor (dets, 1)
        """
        res_boxes = []
        res_scores = []
        res_classid = []
        for i in range(0, len(outputs)):
            origin_h, origin_w, c = image_raw_batch[i].shape
            scaled = []
            grids = []
            for out in outputs[i]:
                out = self.sigmoid_v(out)
                _, _, width, height, _ = out.shape
                grid = self.make_grid(width, height)
                grids.append(grid)
                scaled.append(out)
            z = []

            for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
                _, _, width, height, _ = out.shape
                out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
                out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

                out = out.reshape((1, 3 * width * height, self.output_filter))
                z.append(out)
            pred = np.concatenate(z, 1)
            xc = pred[..., 4] > conf_thres
            pred = pred[xc]
            boxes, confs, classes = self.nms(pred, iou_thres, origin_w, origin_h)
            res_boxes.append(boxes)
            res_scores.append(confs)
            res_classid.append(classes)
        return res_boxes, res_scores, res_classid
    
    def make_grid(self, nx, ny):
        """
        Create scaling tensor based on box location
        Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        Arguments
            nx: x-axis num boxes
            ny: y-axis num boxes
        Returns
            grid: tensor of shape (1, 1, nx, ny, 80)
        """
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)
    def exponential_v(self, array):
        return np.exp(array)
    
    def non_max_suppression(self, boxes, confs, classes, iou_thres=0.3):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
        order = confs.flatten().argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where( ovr <= iou_thres)[0]
            order = order[inds + 1]
        boxes = boxes[keep]
        confs = confs[keep]
        classes = classes[keep]
        return boxes, confs, classes

    def nms(self, pred, iou_thres=0.6, origin_w=0, origin_h=0):
        boxes = self.xywh2xyxy(pred[..., 0:4], origin_w, origin_h)
        # best class only
        confs = np.amax(pred[:, 5:], 1, keepdims=True)
        classes = np.argmax(pred[:, 5:], axis=-1)
        return self.non_max_suppression(boxes, confs, classes, iou_thres)

    def xywh2xyxy(self, x, origin_w=0, origin_h=0):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.INPUT_W / origin_w
        r_h = self.INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y