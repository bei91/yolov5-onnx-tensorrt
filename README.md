## yolov5-onnx-tensorrt 
This Repos contains how to run yolov5 model using TensorRT.  
The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5).  
Convert pytorch to onnx and tensorrt yolov5 model to run on a Jetson AGX Xavier.  
Support to infer an image .  
Support to infer multi images simultaneously.

## Requirements 
Please use torch>=1.6.0 + onnx==1.8.0 + TensorRT 7.0.0.11 to run the code

## Code structure 
`networks` code is network  
`demo` code runs tensorrt implementation on Jetson AGX Xavier
```
├── networks
├── utils
├── models
├── demo
│   ├── demo.py
|   ├── demo_batch.py
|   ├── Processor.py
|   ├── Processor_Batch.py
|   ├── Visualizer.py
|   ├── classes.py
|   ├── yolov5_onnx.py
│   └── onnx_tensorrt.py
```

- [x] convert yolov5 pytorch model to onnx
- [x] convert yolov5 onnx model to tensorrt
- [x] pre-process image 
- [x] run inference against input using tensorrt engine
- [x] post process output (forward pass)
- [x] apply nms thresholding on candidate boxes
- [x] visualize results

___
## Compile pytorch model to onnx
```
python yolov5_onnx.py --help
usage: yolov5_onnx.py [-h] [-w WEIGHTS] [-is IMG-SIZE] [-bs BATCH-SIZE]

compile Pytorch model to ONNX

optional arguments:
  -h, --help            show this help message and exit
  -m WEIGHTS, --weights WEIGHTS
                        pytorch model file location
  -is IMG-SIZE, --img-size IMG-SIZE
                        image size
  -bs BATCH-SIZE, --batch-size BATCH-SIZE
                        batch size
```
## Compile onnx model to tensorrt

```
python onnx_tensorrt.py --help
usage: onnx_tensorrt.py [-h] [-m MODEL] [-fp FLOATINGPOINT] [-o OUTPUT]

compile Onnx model to TensorRT

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        onnx file location
  -fp FLOATINGPOINT, --floatingpoint FLOATINGPOINT
                        floating point precision. 16 or 32
  -o OUTPUT, --output OUTPUT
                        name of trt output file
```

## Run demo to infer an image

```
python demo.py -image=./path -model=./path/model.trt
```
## Run demo_batch to infer multi images simultaneously

```
python demo_batch.py -image=./path -model=./path/model.trt
```
___

## Thanks
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
* [https://github.com/SeanAvery/yolov5-tensorrt](https://github.com/SeanAvery/yolov5-tensorrt)

