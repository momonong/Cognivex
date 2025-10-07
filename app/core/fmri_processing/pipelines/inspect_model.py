import io
import re
import torch
from torchsummary import summary
from contextlib import redirect_stdout


def parse_summary_text(summary_text: str):
    layers = []
    for line in summary_text.split("\n"):
        match = re.match(r"\s*(\w+-\d+)\s+([-a-zA-Z0-9,\[\] ]+)\s+([0-9,]+)", line)
        if match:
            name, output_shape, params = match.groups()
            layers.append(
                {
                    "name": name,
                    "output_shape": output_shape.strip(),
                    "params": int(params.replace(",", "")),
                }
            )
    return layers


def inspect_torch_model(model, input_shape: tuple, device: str) -> list[dict]:
    """
    Return layer information from PyTorch model using named_modules, matched with torchsummary.
    :param model: PyTorch model instance
    :param input_shape: tuple of input shape (excluding batch size)
    :param device: device to use for inspection
    :return: list of layers with name, class_name, output_shape (from summary), and params
    """

    # Ensure model is on the specified device and device is compatible with torchsummary
    original_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    # torchsummary only accepts 'cuda' or 'cpu', not 'mps'
    summary_device = "cuda" if device == "cuda" else "cpu"
    
    model = model.to(summary_device)
    
    try:
        # Get summary string
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary(model, input_size=input_shape, device=summary_device)
        summary_text = buffer.getvalue()
    finally:
        # Restore original device
        model = model.to(original_device)
    parsed_layers = parse_summary_text(summary_text)

    # Get named modules
    module_info = []
    for idx, (name, module) in enumerate(model.named_modules()):
        if name == "":  # skip the top-level model
            continue
        module_info.append({
            "index": idx,
            "name": name,
            "class_name": module.__class__.__name__,
            "instance": module
        })

    # Combine both: align by index
    combined = []
    for mod, parsed in zip(module_info, parsed_layers):
        combined.append({
            "layer_name": parsed["name"],  # e.g. Conv3d-1
            "model_path": mod["name"],     # e.g. capsnet.conv1
            "layer_type": mod["class_name"],
            "output_shape": parsed["output_shape"],
            "params": parsed["params"],
        })

    return combined



if __name__ == "__main__":
    # Example usage
    from scripts.capsnet.model import CapsNetRNN

    MODEL = CapsNetRNN()
    input_shape = (
        1,
        1,
        91,
        91,
        109,
    )  # for instances [batch_size, channels, height, width]

    print(inspect_torch_model(MODEL, input_shape))
