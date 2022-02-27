import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft
import tensorflow as tf
import argparse

def get_nn(spec):
    if spec.WhichOneof("Type") == "neuralNetwork":
        return spec.neuralNetwork
    elif spec.WhichOneof("Type") == "neuralNetworkClassifier":
        return spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
        return spec.neuralNetworkRegressor
    else:
        raise ValueError("MLModel does not have a neural network")


def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    parser.add_argument("model_path", type=str,
                        help='Path to the .h5 model')

    parser.add_argument("out_path", type=str,
                        help='Output path')

    parser.add_argument("--image_size", type=int, default=512)

    args = parser.parse_args()

    model = ct.convert(args.model_path)

    model.author = "Leon Jungemeyer"
    model.version = '1.0'
    model.short_description = "Model to enhance dark sky images."

    spec = model.get_spec()
    input_names = [inp.name for inp in spec.description.input]
    ct.utils.rename_feature(spec, input_names[0], 'image')

    input = spec.description.input[0]
    input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    input.type.imageType.height = args.image_size
    input.type.imageType.width = args.image_size
    
    nn = get_nn(spec)

    # Scale the input image
    preprocessing = nn.preprocessing.add()
    preprocessing.scaler.blueBias = -1
    preprocessing.scaler.greenBias = -1
    preprocessing.scaler.redBias = -1
    preprocessing.scaler.channelScale = (1 / 127.5)

    new_layer = nn.layers.add()
    new_layer.name = "add"
    params = ct.proto.NeuralNetwork_pb2.AddLayerParams
    new_layer.add.alpha = 1.0

    new_layer.output.append(nn.layers[-2].output[0])
    nn.layers[-2].output[0] = nn.layers[-2].name + "_output"
    new_layer.input.append(nn.layers[-2].output[0])

    new_layer = nn.layers.add()
    new_layer.name = "scale"
    params = ct.proto.NeuralNetwork_pb2.MultiplyLayerParams
    new_layer.multiply.alpha = 127.5

    new_layer.output.append(nn.layers[-2].output[0])
    nn.layers[-2].output[0] = nn.layers[-2].name + "_output"
    new_layer.input.append(nn.layers[-2].output[0])

    output = spec.description.output[0]
    output.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    output.type.imageType.height = args.image_size
    output.type.imageType.width = args.image_size

    ct.models.utils.save_spec(spec, args.out_path)


if __name__ == "__main__":
    main()