import torch
from torchvision.models import densenet121
from typing import List

def init_model():
    return densenet121().to(torch.float32).cuda()

def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# Reset since we are using a different backend.
torch._dynamo.reset()

opt_model = torch.compile(init_model(), backend=custom_backend)
opt_model(generate_data(16)[0])


def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

opt_bar = torch.compile(bar, backend=custom_backend)
inp1 = torch.randn(10)
inp2 = torch.randn(10)
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)

torch._dynamo.reset()
explain_output = torch._dynamo.explain(bar)(torch.randn(10), torch.randn(10))
print(explain_output)