from onnxruntime.quantization import quantize_dynamic, QuantType
from sr_mobile_pytorch.model import AnchorBasedPlainNet
import torch


def main():
    model_checkpoint = "./experiments/generator_minecraft_x4_v2/model.pth"
    onnx_model_name = model_checkpoint.replace("pth", "onnx")
    quantized_model_name = model_checkpoint.replace("pth", "quant.onnx")

    weights = torch.load(model_checkpoint, map_location=torch.device("cpu"))
    model = AnchorBasedPlainNet()
    model.load_state_dict(weights, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 160, 100, requires_grad=True)

    with torch.no_grad():
        output_tensor = model(dummy_input)
        print(output_tensor.shape)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    quantize_dynamic(
        onnx_model_name, quantized_model_name, weight_type=QuantType.QUInt8
    )


if __name__ == "__main__":
    main()
