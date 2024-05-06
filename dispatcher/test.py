import torch

torch.ops.load_library("build/libdispather.so")
print(torch.ops.myops.myadd(torch.rand(2,2), torch.rand(2,2)))
