import torch
import torch.nn as nn
from onnxruntime.quantization import quantize_dynamic, QuantType

from sr_mobile_pytorch.model import AnchorBasedPlainNet


class AnchorBasedPlainNetChannelLast(nn.Module):
    def __init__(self, model_checkpoint: str):
        super(AnchorBasedPlainNetChannelLast, self).__init__()
        self.weights = torch.load(model_checkpoint, map_location=torch.device("cpu"))
        self.abpn = AnchorBasedPlainNet()
        self.abpn.load_state_dict(self.weights, strict=True)
        self.abpn.eval()

    def forward(self, x):
        output = torch.permute(
            self.abpn(torch.permute(x[:, :, :, :3], (0, 3, 1, 2))), (0, 2, 3, 1)
        )
        n, h, w, c = output.shape
        alpha = torch.zeros(1, h, w, 1)
        return torch.concat((output, alpha), dim=3).type(torch.uint8)


def main():
    model_checkpoint = (
        "./experiments/generator_minecraft_x4_ios/model_minecraft_x4_ios.pth"
    )
    onnx_model_name = model_checkpoint.replace("pth", "onnx")
    quantized_model_name = model_checkpoint.replace("pth", "quant.onnx")

    model = AnchorBasedPlainNetChannelLast(model_checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 160, 100, 4, requires_grad=True)

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
            "input": {0: "batch_size", 1: "height", 2: "width"},
            "output": {0: "batch_size", 1: "height", 2: "width"},
        },
    )

    quantize_dynamic(
        onnx_model_name, quantized_model_name, weight_type=QuantType.QUInt8
    )


if __name__ == "__main__":
    main()
