import torch
import torch._dynamo as dynamo
import torch.fx
import logging
import torch._dynamo
import dis
from typing import List

torch._logging.set_logs(dynamo=logging.DEBUG)
torch._logging.set_logs(all=logging.DEBUG)


def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    gm.graph.print_tabular()
    print(dis.Bytecode(gm.forward.__code__).dis())
    return gm.forward


def add_raw(a, b):
    c = a + b
    return c


@dynamo.optimize(custom_backend)
def add(a, b):
    c = a + b
    return c


x = torch.randn(10, requires_grad=False)
y = torch.randn(10, requires_grad=False)

z = add(x, y)
print(f"res: {z}")