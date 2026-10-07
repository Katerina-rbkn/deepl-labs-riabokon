import torch

torch.set_default_dtype(torch.float64)

def torch_reference(p, X, y):
    lin1, lin2 = torch.nn.Linear(4, 8), torch.nn.Linear(8, 3)
    with torch.no_grad():
        lin1.weight.copy_(torch.from_numpy(p["W1"].T.copy()))  # (out, in)!
        lin1.bias.copy_(torch.from_numpy(p["b1"].copy()))
        lin2.weight.copy_(torch.from_numpy(p["W2"].T.copy()))
        lin2.bias.copy_(torch.from_numpy(p["b2"].copy()))
    logits = lin2(torch.relu(lin1(torch.from_numpy(X))))
    loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y).long())
    loss.backward()
    grads = {"W1": lin1.weight.grad.T.numpy(), "b1": lin1.bias.grad.numpy(),
             "W2": lin2.weight.grad.T.numpy(), "b2": lin2.bias.grad.numpy()}
    return loss.item(), grads