import io
from torchsummary import summary
from contextlib import redirect_stdout

def inspect_torch_model(MODEL_CLASS, input_shape: tuple) -> str:
    """
    Returns PyTorch model architecture summary as a string.
    :param MODEL: PyTorch model instance
    :param input_shape: Input shape tuple, excluding batch size
    :return: Summary string
    """
    model = MODEL_CLASS.to("cpu")
    
    # 使用 StringIO 來捕捉輸出
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        summary(model, input_size=input_shape, device="cpu")
    
    output = buffer.getvalue()
    return output


if __name__ == "__main__":
    # Example usage
    from scripts.capsnet.model import CapsNetRNN  
    MODEL = CapsNetRNN()
    input_shape = (1, 1, 91, 91, 109)  # for instances [batch_size, channels, height, width]
    
    print(inspect_torch_model(MODEL, input_shape))