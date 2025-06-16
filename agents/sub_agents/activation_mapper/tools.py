def extract_activation(model, layer_name, input_tensor):
    activation = {}

    def hook_fn(module, input, output):
        activation["value"] = output.detach().cpu()

    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)
    _ = model(input_tensor)
    handle.remove()

    return activation["value"]
