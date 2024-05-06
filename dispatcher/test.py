import torch

torch.ops.load_library("build/libdispatcher.so")
a, b = torch.rand(2,2), torch.rand(2,2)
print(a)
print(b)
print(a + b)
print(torch.ops.myops.myadd(a, b))
