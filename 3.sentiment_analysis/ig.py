def integrated_gradients_sst(s):
    x=embed_sentence(s)
    x_dash = torch.zeros_like(x)
    sum_grad = None
    grad_array = None
    x_array = None
    m = 300
    for k in range(m):
        step_input = x_dash + k * (x - x_dash) / m
        step_output = model(step_input,embed=False)
        step_grad = torch.autograd.grad(step_output[0,torch.max(step_output, 1)[1].item()], x, retain_graph=True)[0]
        if sum_grad is None:
            sum_grad = step_grad
            grad_array = step_grad
            x_array = step_input
        else:
            sum_grad += step_grad
            grad_array = torch.cat([grad_array, step_grad])
            x_array = torch.cat([x_array, step_input])
    sum_grad = sum_grad / m
    sum_grad = sum_grad * (x - x_dash)
    sum_grad = sum_grad.sum(dim=2)
    relevances = sum_grad.detach().cpu().numpy()
    return relevances

def integrated_gradients_imdb(s):
    x=embed_sentence(s)
    x_dash = torch.zeros_like(x)
    sum_grad = None
    grad_array = None
    x_array = None
    m = 300
    for k in range(m):
        step_input = x_dash + k * (x - x_dash) / m
        step_output = torch.sigmoid(model(step_input,embed=False))
        step_grad = torch.autograd.grad(step_output, x, retain_graph=True)[0]
        if sum_grad is None:
            sum_grad = step_grad
            grad_array = step_grad
            x_array = step_input
        else:
            sum_grad += step_grad
            grad_array = torch.cat([grad_array, step_grad])
            x_array = torch.cat([x_array, step_input])
    sum_grad = sum_grad / m
    sum_grad = sum_grad * (x - x_dash)
    sum_grad = sum_grad.sum(dim=2)
    relevances = sum_grad.detach().cpu().numpy()
    return relevances
