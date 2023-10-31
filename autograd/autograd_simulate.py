import numpy as np
from typing import List, NamedTuple, Callable, Dict, Optional

class Tape(NamedTuple):
    inputs : List[str]
    outputs : List[str]
    # apply chain rule
    propagate : 'Callable[List[Variable], List[Variable]]'

grad_tape = []

unique_id = 1
def unique_name():
    global unique_id;
    name = f"v_{unique_id}"
    unique_id += 1
    return name

class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or unique_name()

    def __repr__(self):
        return repr(self.value)

    @staticmethod
    def constant(value, name = None):
        var = Variable(value, name)
        print(f'{var.name} = {value}')
        return var

    def __mul__(self, other):
        return ops_mul(self, other)

    def __add__(self, other):
        return ops_add(self, other)

    def __sub__(self, other):
        return ops_sub(self, other)

    @staticmethod
    def sin(x):
        return ops_sin(x)

    @staticmethod
    def log(x):
        return ops_log(x)

def ops_mul(self, other):
    # forward
    x = Variable(self.value * other.value)
    print(f"forward: {x.name} = {self.name} * {other.name}")

    # backward
    def backward(dl_doutputs):
        dl_dx, = dl_doutputs
        dx_dself = other
        dx_dother = self
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs

    tape = Tape(inputs = [self.name, other.name], outputs = [x.name], propagate = backward)
    grad_tape.append(tape)
    return x

def ops_add(self, other):
    # forward
    x = Variable(self.value + other.value)
    print(f"forward: {x.name} = {self.name} + {other.name}")

    # backward
    def backward(dl_doutputs):
        dl_dx, = dl_doutputs
        return [dl_dx * Variable(1.), dl_dx * Variable(1.)]

    tape = Tape(inputs = [self.name, other.name], outputs = [x.name], propagate = backward)
    grad_tape.append(tape)
    return x

def ops_sub(self, other):
    # forward
    x = Variable(self.value - other.value)
    print(f"forward: {x.name} = {self.name} - {other.name}")

    # backward
    def backward(dl_doutputs):
        dl_dx, = dl_doutputs
        return [dl_dx * Variable(1.), dl_dx * Variable(-1.)]

    tape = Tape(inputs = [self.name, other.name], outputs = [x.name], propagate = backward)
    grad_tape.append(tape)
    return x

def ops_sin(self):
    # forward
    x = Variable(np.sin(self.value))
    print(f"forward: sin({self.name})")

    # backward
    def backward(dl_doutputs):
        dl_dx, = dl_doutputs
        return [dl_dx * Variable(np.cos(self.value))]

    tape = Tape(inputs = [self.name], outputs = [x.name], propagate = backward)
    grad_tape.append(tape)
    return x

def ops_log(self):
    # forward
    x = Variable(np.log(self.value))
    print(f"forward: log({self.name})")

    # backward
    def backward(dl_doutputs):
        dl_dx, = dl_doutputs
        return [dl_dx * Variable(1 / self.value)]

    tape = Tape(inputs = [self.name], outputs = [x.name], propagate = backward)
    grad_tape.append(tape)
    return x

def backward(l, result):
    dl_d = {}
    dl_d[l.name] = Variable.constant(1.)

    for entry in reversed(grad_tape):
        dl_doutputs = [dl_d[out] if out in dl_d else None for out in entry.outputs]
        dl_dinputs = entry.propagate(dl_doutputs)

        for input, dl_dinput in zip(entry.inputs, dl_dinputs):
            if input in dl_d:
                dl_d[input] += dl_dinput
            else:
                dl_d[input] = dl_dinput
    for name, value in dl_d.items():
        print(f"{name} has grad: {value}")

    return [dl_d[out.name] if out.name in dl_d else None for out in result]

def main():
    x = Variable.constant(2.)
    y = Variable.constant(5.)
    f = Variable.log(x) + x * y - Variable.sin(y)
    print(f)

    grads = backward(f, [x, y])
    print(grads)

if __name__ == "__main__":
    main()
