import torch

# fp32 = scale * (int8 - zero_point)
# int8 = fp32 / scale + zero_point
# scale = (fp32_max - fp32_min) / (int8_max - int8_min)
# zero_point = int8_max - (fp32_max / scale)

def quantize(inp, min=None, max=None):
    if min is None or max is None:
        min, max = torch.min(inp), torch.max(inp)
    scale = (max - min) / 255
    zero_point = int(255 - (max / scale))
    return (inp / scale + zero_point).to(torch.uint8), scale, zero_point

def dequantize(inp, scale, zero_point):
    result = scale * (inp - zero_point)
    return result.to(torch.float32)

def quantized_mul(x, y):
    quantized_x, x_scale, x_zero_point = quantize(x)
    quantized_y, y_scale, y_zero_point = quantize(y)
    # NOTE: z_min/z_max should be kown value before inference
    quantized_z, z_scale, z_zero_point = quantize(x * y)

    result = (quantized_x.to(torch.int) - x_zero_point) * (quantized_y.to(torch.int) - y_zero_point) * (x_scale * y_scale / z_scale) + z_zero_point

    return dequantize(result, z_scale, z_zero_point)

if __name__ == "__main__":
    x = torch.rand(3, 3) * 300
    y = torch.rand(3, 3) * 300

    res = quantized_mul(x, y)
    print(res)
    print(x * y)


