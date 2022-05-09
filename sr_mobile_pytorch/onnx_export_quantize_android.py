import torch
import torch.nn as nn
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import warnings

warnings.filterwarnings("ignore")

from sr_mobile_pytorch.model import AnchorBasedPlainNet


class AnchorBasedPlainNetChannelLast(nn.Module):
    def __init__(self, model_checkpoint: str):
        super(AnchorBasedPlainNetChannelLast, self).__init__()
        self.weights = torch.load(model_checkpoint, map_location=torch.device("cpu"))
        self.abpn = AnchorBasedPlainNet()
        self.abpn.load_state_dict(self.weights, strict=True)
        self.abpn.eval()

    def forward(self, bitmap: torch.Tensor):
        # bitmap to floatarray
        # x = torch.permute(bitmap.repeat(1, 3, 1, 1), (0, 2, 3, 1))
        # mask = torch.tensor([16, 8, 0], dtype=torch.int32)
        # rs = x.__rshift__(mask).type(torch.uint8)
        # copy = torch.zeros_like(rs, dtype=torch.uint8).copy_(rs).type(torch.float32)
        # input_tensor = torch.permute(copy, (0, 3, 1, 2))

        output = self.abpn(bitmap)

        # floatarray to bitmap
        output = torch.permute(output, (0, 2, 3, 1)).to(torch.int32)
        mask = torch.tensor([16, 8, 0], dtype=torch.int32)
        ls = output.__lshift__(mask)
        bitmap = torch.flatten(torch.sum(ls, dim=-1)).type(torch.int32) - 16777216
        return bitmap


def main():
    model_checkpoint = (
        "./experiments/generator_minecraft_x4_android/model_minecraft_x4_android.pth"
    )
    onnx_model_name = model_checkpoint.replace("pth", "onnx")
    quantized_model_name = model_checkpoint.replace("pth", "quant.onnx")

    model = AnchorBasedPlainNetChannelLast(model_checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 3, 160, 100, requires_grad=True) * 255.0

    with torch.no_grad():
        output_bitmap = model(dummy_input)
        print(output_bitmap)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {0: "num_pixels"},},
    )

    ort_session = onnxruntime.InferenceSession(onnx_model_name)
    ort_inputs = {
        "input": dummy_input.detach().numpy(),
    }
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0])

    # quantize_dynamic(
    #     onnx_model_name, quantized_model_name, weight_type=QuantType.QUInt8
    # )


if __name__ == "__main__":
    main()
