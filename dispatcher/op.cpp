#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

#include <ATen/NamedTensorUtils.h>

using torch::Tensor;
using torch::DeviceType;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;

Tensor myadd(const Tensor& self, const Tensor& other) {
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("myops::myadd", "")
    .typed<decltype(myadd)>();
  return op.call(self, other);
}

TORCH_LIBRARY(myops, m) {
  m.def("myadd(Tensor self, Tensor other) -> Tensor");
}

Tensor myadd_cpu(const Tensor& self_, const Tensor& other_) {
  TORCH_CHECK(self_.sizes() == other_.sizes());
  TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(other_.device().type() == DeviceType::CPU);
  Tensor self = self_.contiguous();
  Tensor other = other_.contiguous();
  Tensor result = torch::empty(self.sizes(), self.options());

  const auto* self_ptr = self.data_ptr<float>();
  const auto* other_ptr = other.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  for (int64_t i = 0; i < self.numel(); ++i) {
    result_ptr[i] = self_ptr[i] + other_ptr[i];
  }
  return result;
}

TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("myadd", myadd_cpu);
}

Tensor myadd_cuda(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(0, "Not implemented yet.");
}

TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("myadd", myadd_cuda);
}

// FIXME: output tensor's shape is not correct when enable following code.
// class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
// public:
//   static Tensor forward(
//     AutogradContext *ctx, torch::Tensor self, torch::Tensor other
//   ) {
//     // at::AutoNonVariableTypeMode g;
//     at::AutoDispatchBelowADInplaceOrView guard;
//     return myadd(self, other);
//   }
//
//   static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
//     auto grad_output = grad_outputs[0];
//     return {grad_output, grad_output};
//   }
// };
//
// Tensor myadd_autograd(const Tensor& self, const Tensor& other) {
//   return MyAddFunction::apply(self, other)[0];
// }
//
// TORCH_LIBRARY_IMPL(myops, Autograd, m) {
//   m.impl("myadd", myadd_autograd);
// }
