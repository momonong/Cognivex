import torch

# 存 activation 的全域變數（你也可以用 class 包裝）
activations = {}

def get_activation(name):
    """建立 forward hook 並儲存到 activations"""
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

def clear_activations():
    activations.clear()

def get_saved_activation(name):
    return activations.get(name, None)
