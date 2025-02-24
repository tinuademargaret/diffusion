def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            param.grad.detach()
            param.grad.zero()
