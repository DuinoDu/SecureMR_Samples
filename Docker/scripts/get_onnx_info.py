import onnx
import argparse

parser = argparse.ArgumentParser(description='get onnx info')
parser.add_argument('onnxfile', type=str, help='input onnx file')
args = parser.parse_args()
model = onnx.load(args.onnxfile)
for input in model.graph.input:
    input_name = input.name
    shape = [int(dim.dim_value) for dim in input.type.tensor_type.shape.dim]
    res = f"-d {input_name} {shape}"
    print(res)
