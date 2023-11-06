import torch
def fn(x, y):
    a = torch.cos(x).cuda()
    b = torch.sin(y).cuda()
    return a + b
new_fn = torch.compile(fn, backend="inductor")
input_tensor = torch.randn(10000).to(device="cuda:0")
a = new_fn(input_tensor, input_tensor)

print(new_fn)