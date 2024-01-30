import torch
import torch.fx as fx
# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.relu(self.linear(x + self.param).clamp(min=0.0, max=1.0))

module = MyModule()

from torch.fx import symbolic_trace
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

# High-level intermediate representation (IR) - Graph representation
print(symbolic_traced.graph)
"""
graph():
    %x : [num_users=1] = placeholder[target=x]
    %param : [num_users=1] = get_attr[target=param]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
    %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
    %clamp : [num_users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
    return clamp
"""

# Code generation - valid Python code
print(symbolic_traced.code)
"""
def forward(self, x):
    param = self.param
    add = x + param;  x = param = None
    linear = self.linear(add);  add = None
    clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
    return clamp
"""

# Transform
def replace_activation(m: torch.nn.Module,
                       tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    for node in graph.nodes:
        if node.op == 'call_function' and node.target == torch.relu:
            # Specifies the insertion point. Any nodes added to the
            # Graph within this scope will be inserted after `node`
            with graph.inserting_after(node):
                # Insert a new `call_function` node calling `gelu`
                new_node = graph.call_function(torch.nn.functional.gelu, node.args)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
    graph.lint()
    return fx.GraphModule(m, graph)


replaced_module = replace_activation(MyModule())
print(replaced_module)

# onnx_program = torch.onnx.dynamo_export(symbolic_traced, torch.rand(3,4))
