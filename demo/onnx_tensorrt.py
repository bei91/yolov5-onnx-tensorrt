import tensorrt as trt
import sys
import argparse

"""
takes in onnx model
converts to tensorrt
tensorrt model input size must be src pth input size 
"""

def cli():
    desc = 'Compile Onnx model to TensorRT'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='', help='onnx file location')
    parser.add_argument('-fp', '--floatingpoint', type=int, default=16, help='floating point precision. 16 or 32')
    parser.add_argument('-o', '--output', default='', help='name of trt output file')
    args = parser.parse_args()
    model = 'coco_yolov5.onnx'
    fp = args.floatingpoint
    if fp != 16 and fp != 32:
        print('floating point precision must be 16 or 32')
        sys.exit()
    output = 'coco_yolov5-{}.trt'.format(fp)
    return {
        'model': model,
        'fp': fp,
        'output': output
    }

if __name__ == '__main__':
    args = cli()
    batch_size = 4
    model = '{}'.format(args['model'])
    output = '{}'.format(args['output'])
    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # trt7
    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = batch_size
        if args['fp'] == 16:
            builder.fp16_mode = True
        with open(model, 'rb') as f:
            print('Beginning ONNX file parsing')
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("ERROR", parser.get_error(error))
        print("num layers:", network.num_layers)
        network.get_input(0).shape = [batch_size, 3, 608, 608]  # trt7
        # last_layer = network.get_layer(network.num_layers - 1)
        # network.mark_output(last_layer.get_output(0))
        # reshape input from 32 to 1
        engine = builder.build_cuda_engine(network)
        with open(output, 'wb') as f:
            f.write(engine.serialize())
        print("Completed creating Engine")
