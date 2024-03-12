# Pytorch Dispatch from add exmaple

## Add example

The discussion will based on a simple add test case.

```
import torch

a = torch.randn(3,3)
b = torch.randn(3,3)

c = a + b
```



## Code generation



## Backtrace analysis

`torch/include/ATen/ops/add_ops.h` generated `add__Tensor`

```c++
struct TORCH_API add_Tensor {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::add")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Tensor")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
  static at::Tensor call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
};
```

Find `OperatorHandle` for `add`
```c++
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add__Tensor::name, add__Tensor::overload_name)
      .typed<add__Tensor::schema>();
```

Get `KernelFunction` for `add`

```c++
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
```



### CPP call stack

#### `/torch/csrc/autograd/generated/python_variable_methods.cpp`

- `__add__` is binding to `THPVariable_add`

```c++
PyMethodDef variable_methods[] = {
  // These magic methods are all implemented on python object to wrap NotImplementedError
  {"__add__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
};
```

- `THPVariable_add`

```c++
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::add(Tensor self, Scalar alpha, Tensor other) -> Tensor
      
      auto dispatch_add = [](const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_add = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha); // <------- CALLED HERE 2nd
      };
      return wrap(dispatch_add(self, _r.tensor(0), _r.scalar(1))); // <------- CALLED HERE 1st
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
```

#### `build/aten/src/ATen/core/TensorBody.h`

```c++
// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
inline at::Tensor Tensor::add(const at::Tensor & other, const at::Scalar & alpha) const {
    return at::_ops::add_Tensor::call(const_cast<Tensor&>(*this), other, alpha); // <------- CALLED HERE
}
```

#### `build/aten/src/ATen/Operators_2.cpp`

```c++
// aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<add__Tensor::schema> create_add__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add__Tensor::name, add__Tensor::overload_name)
      .typed<add__Tensor::schema>();
}

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    
    static auto op = create_add_Tensor_typed_handle();
    return op.call(self, other, alpha); // <------- CALLED HERE
}
```

#### `aten/src/ATen/core/dispatch/Dispatcher.h` 

- `TypedOperatorHandle`

```c++
template<class FuncType>
class TypedOperatorHandle final {
  static_assert(guts::false_t<FuncType>(), "FuncType in OperatorHandle::typed<FuncType> was not a valid function type");
};
template<class Return, class... Args>
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle {
public:
  TypedOperatorHandle(TypedOperatorHandle&&) noexcept = default;
  TypedOperatorHandle& operator=(TypedOperatorHandle&&) noexcept = default;
  TypedOperatorHandle(const TypedOperatorHandle&) = default;
  TypedOperatorHandle& operator=(const TypedOperatorHandle&) = default;

  // See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
  C10_ALWAYS_INLINE Return call(Args... args) const {
    return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...); // <------- CALLED HERE
  }

  // See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
  C10_ALWAYS_INLINE Return redispatch(DispatchKeySet currentDispatchKeySet, Args... args) const {
    return c10::Dispatcher::singleton().redispatch<Return, Args...>(*this, currentDispatchKeySet, std::forward<Args>(args)...);  // <------- CALLED HERE later
  }

private:
  explicit TypedOperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : OperatorHandle(operatorIterator) {}
  friend class OperatorHandle;
};
```

- `Dispatcher::call`

```c++
template<class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...); // <------- CALLED HERE
}
```

- `Dispatcher::redispatch`

```c++
// See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
template<class Return, class... Args>
inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  // do not use RecordFunction on redispatch
  const KernelFunction& kernel = op.operatorDef_->op.lookup(currentDispatchKeySet);
  return kernel.template call<Return, Args...>(op, currentDispatchKeySet, std::forward<Args>(args)...); // <------- CALLED HERE later
}
```

#### `aten/src/ATen/core/boxing/KernelFunction_impl.h`

- `KernelFunction::call`

```c++
template<class Return, class... Args>
C10_ALWAYS_INLINE Return KernelFunction::call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const {
    // note: Args above is intentionally not Args&&. We don't want perfect
    // forwarding, which would require Args to be deduced, but instead we
    // want callers to explicitly specify the Args.

    if constexpr (std::disjunction_v<has_symint<Args>...>) {
      if (sym_unboxed_kernel_func_ != nullptr) {
        auto *functor = boxed_kernel_func_.getFunctor();
        return callUnboxedKernelFunction<Return, Args...>(
            sym_unboxed_kernel_func_, functor, dispatchKeySet, std::forward<Args>(args)...);
      }

      if (unboxed_kernel_func_ != nullptr) {
        auto *functor = boxed_kernel_func_.getFunctor();
        return callUnboxedKernelFunction<Return, typename remove_symint<Args>::type...>(
            unboxed_kernel_func_, functor, dispatchKeySet, unpackSymInt<Args>(args)...);
      }
    } else {
      if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
        auto *functor = boxed_kernel_func_.getFunctor();
        return callUnboxedKernelFunction<Return, Args...>(
            unboxed_kernel_func_, functor, dispatchKeySet, std::forward<Args>(args)...); // <------- CALLED HERE
      }
    }

    return impl::BoxedKernelWrapper<Return(Args...)>::call(
        boxed_kernel_func_,
        opHandle,
        dispatchKeySet,
        std::forward<Args>(args)...
    );
}
```

- `callUnboxedKernelFunction`

```c++
template<class Return, class... Args>
inline Return callUnboxedKernelFunction(void* unboxed_kernel_func, OperatorKernel* functor, DispatchKeySet dispatchKeySet, Args&&... args) {
    using ActualSignature = Return (OperatorKernel*, DispatchKeySet, Args...);
    ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func);
    return (*func)(functor, dispatchKeySet, std::forward<Args>(args)...); // <------- CALLED HERE
}
```

#### ` aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h`

```c++
  // This specialization is for kernels with a first argument of type DispatchKeySet
  template<class KernelFunctor, class ReturnType, class... ParameterTypes>
  struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(DispatchKeySet, ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
      "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<DispatchKeySet, ParameterTypes...>, typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
      "Parameter types mismatch");

    // See [Note: Argument forwarding in the dispatcher] for why ParameterTypes doesn't use &&
    static ReturnType call(OperatorKernel* functor, DispatchKeySet dispatchKeySet, ParameterTypes... args) {
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      // We're explicitly taking in a dispatchKeySet and forwarding it to the registered kernel.
      // See Note [Plumbing Keys Through The Dispatcher 2] for details.
      return (*functor_)(dispatchKeySet, std::forward<ParameterTypes>(args)...); // <------- CALLED HERE
    }
  };
```

#### ` aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h`

```c++
    template<class FuncPtr, class ReturnType, class ParameterList> class WrapFunctionIntoFunctor_ {};
    template<class FuncPtr, class ReturnType, class... Parameters>
    class WrapFunctionIntoFunctor_<FuncPtr, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
    public:
      C10_ALWAYS_INLINE decltype(auto) operator()(Parameters... args) {
        return (*FuncPtr::func_ptr())(std::forward<Parameters>(args)...); // <------- CALLED HERE
      }
    };
```

#### ` torch/csrc/autograd/generated/VariableType_2.cpp`

```c++
at::Tensor add_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, other );
  
  [[maybe_unused]] auto _any_has_forward_grad_result = (isFwGradDefined(self) || isFwGradDefined(other));
  std::shared_ptr<AddBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddBackward0>(new AddBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
    grad_fn->other_scalar_type = other.scalar_type();
    grad_fn->self_scalar_type = self.scalar_type();
  }
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::add(ks & c10::after_autograd_keyset, self_, other_, alpha); // <------- CALLED HERE 2nd
  })(); // <------- CALLED HERE 1st
  auto result = std::move(_tmp);
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  c10::optional<at::Tensor> result_new_fw_grad_opt = c10::nullopt;
  if (_any_has_forward_grad_result && (result.defined())) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_tensor = toNonOptTensor(self);
      auto self_t = (self_t_raw.defined() || !self_tensor.defined())
        ? self_t_raw : at::_efficientzerotensor(self_tensor.sizes(), self_tensor.options());
      auto other_t_raw = toNonOptFwGrad(other);
      auto other_tensor = toNonOptTensor(other);
      auto other_t = (other_t_raw.defined() || !other_tensor.defined())
        ? other_t_raw : at::_efficientzerotensor(other_tensor.sizes(), other_tensor.options());
      result_new_fw_grad_opt = self_t + maybe_multiply(other_t, alpha);
  }
  if (result_new_fw_grad_opt.has_value() && result_new_fw_grad_opt.value().defined() && result.defined()) {
    // The hardcoded 0 here will need to be updated once we support multiple levels.
    result._set_fw_grad(result_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ false);
  }
  return result;
}
```

#### `build/aten/src/ATen/RedispatchFunctions.h`

```c++
    // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    inline at::Tensor add(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1) {
        return at::_ops::add_Tensor::redispatch(dispatchKeySet, self, other, alpha); // <------- CALLED HERE
    }
```

#### `build/aten/src/ATen/Operators_2.cpp`

```c++
// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    
    static auto op = create_add_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha); // <------- CALLED HERE
}
```

#### [`aten/src/ATen/core/dispatch/Dispatcher.h`](#`aten/src/ATen/core/dispatch/Dispatcher.h`)

#### [`aten/src/ATen/core/boxing/KernelFunction_impl.h`](#`aten/src/ATen/core/boxing/KernelFunction_impl.h`)

#### `aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h`

```c++
  // This specialization is for kernels with a first argument that is NOT of type DispatchKeySet
  // This includes kernels with 0 arguments.
  template<class KernelFunctor, class ReturnType, class... ParameterTypes>
  struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
      "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<ParameterTypes...>, typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
      "Parameter types mismatch");

    // See [Note: Argument forwarding in the dispatcher] for why ParameterTypes doesn't use &&
    static ReturnType call(OperatorKernel* functor, DispatchKeySet, ParameterTypes... args) {
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      // Note [Plumbing Keys Through The Dispatcher 2]
      // See Note [Plumbing Keys Through The Dispatcher] for the background.
      // This functor explicitly takes in a dispatchKeySet and drops it on the floor- it does not forward it to the registered kernel.
      //
      // This is due to the calling convention within the dispatcher, which expects all registered kernels to have a first argument of type
      // DispatchKeySet.
      // This is not the case for pretty much all manually written kernels, however- this functor serves to separate the calling convention
      // of the dispatcher from the calling convention of manually written kernels.
      return (*functor_)(std::forward<ParameterTypes>(args)...); // <------- CALLED HERE
    }
  };
```

#### [` aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h`](#` aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h`)

#### `build/aten/src/ATen/RegisterCPU.cpp`

```c++
at::Tensor wrapper_CPU_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
structured_ufunc_add_CPU_functional op;
op.meta(self, other, alpha);
op.impl(self, other, alpha, op.outputs_[0]); // <------- CALLED HERE
return std::move(op.outputs_[0]);
}
```

#### ` build/aten/src/ATen/UfuncCPU_add.cpp`

```c++
TORCH_IMPL_FUNC(ufunc_add_CPU)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {
  add_stub(device_type(), *this, alpha); // <------- CALLED HERE
}
```

#### ` aten/src/ATen/native/DispatchStub.h`

```c++
template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

private:
  FnPtr get_call_ptr(c10::DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
      )
    );
  }
public:
  template <typename... ArgTypes>
  rT operator()(c10::DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...); // <------- CALLED HERE
  }
  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }
  void set_mps_dispatch_ptr(FnPtr fn_ptr) {
    impl.mps_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_privateuse1_dispatch_ptr(FnPtr fn_ptr) {
    impl.privateuse1_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  static TORCH_API FnPtr DEFAULT;
private:
  DispatchStubImpl impl;
};

```

#### `build/aten/src/ATen/UfuncCPUKernel_add.cpp`

```c++
void add_kernel(TensorIteratorBase& iter, const at::Scalar & alpha) {
  AT_DISPATCH_SWITCH(iter.common_dtype(), "add_stub",
    AT_DISPATCH_CASE(at::ScalarType::Float,
      [&]() {
    
      auto _s_alpha = alpha.to<scalar_t>();
      auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
      cpu_kernel_vec(iter,
        [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
        [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha);           }
        );
      }
)                    
);
}
```

#### ` aten/src/ATen/native/cpu/Loops.h`

```c++
template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU, but some kernels (like Fill)
  // explicitly dynamic_cast, so we give the opt-out of checking.
  if constexpr (check_dynamic_cast) {
    TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
  }

  iter.for_each(make_vectorized_loop2d(op, vop), grain_size); // <------- CALLED HERE
  iter.cast_outputs();
}
```

#### ` aten/src/ATen/TensorIterator.cpp`

- `for_each`

```c++
void TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  int64_t numel = this->numel();
  if (numel == 0) {
    return;
  } else if (numel < grain_size || at::get_num_threads() == 1) {
    return serial_for_each(loop, {0, numel}); // <------- CALLED HERE
  } else {
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}
```

- `serial_for_each`

```c++
void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) {
    return;
  }

  const auto ntensors = this->ntensors();
  const auto ndim = this->ndim();

  c10::SmallBuffer<char*, 4> ptrs(ntensors);
  c10::SmallBuffer<int64_t, 8> strides(ntensors * static_cast<size_t>(std::max(ndim, 2)));

  at::get_base_ptrs(ptrs.data(), operands_);
  at::get_strides(strides.data(), operands_, ndim);
  at::internal::serial_for_each(
      shape_, strides, ptrs.data(), ptrs.size(), loop, range); // <------- CALLED HERE
}
```

#### ` aten/src/ATen/TensorIteratorInternal.h`

```c++
inline void serial_for_each(
    IntArrayRef shape,
    IntArrayRef strides,
    char** base_ptrs,
    size_t ntensors,
    typename TensorIteratorBase::loop2d_t loop,
    Range range) {
  const auto ndim = shape.size();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      strides.size() == ntensors * std::max(size_t{2}, ndim));

  if (ndim <= 1) {
    if (range.begin == 0) {
      loop(base_ptrs, strides.data(), range.size(), 1); // <------- CALLED HERE
    } else {
      c10::SmallBuffer<char*, 4> ptrs(ntensors);
      get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, {range.begin});
      loop(ptrs.data(), strides.data(), range.size(), 1);
    }
  } else {
    c10::SmallBuffer<char*, 4> ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    while (!counter.is_done()) {
      get_data_ptrs(
          ptrs.data(), {base_ptrs, ntensors}, strides, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
    }
  }
}
```

#### ` pytorch/c10/util/FunctionRef.h`

```c++
template <typename Fn>
class function_ref;

template <typename Ret, typename... Params>
class function_ref<Ret(Params...)> {
  Ret (*callback)(intptr_t callable, Params... params) = nullptr;
  intptr_t callable{};

  template <typename Callable>
  static Ret callback_fn(intptr_t callable, Params... params) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Params>(params)...); // <------- CALLED HERE 2nd
  }

 public:
  function_ref() = default;
  function_ref(std::nullptr_t) {}

  template <typename Callable>
  function_ref(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      Callable&& callable,
      std::enable_if_t<
          !std::is_same_v<std::remove_reference_t<Callable>, function_ref>>* =
          nullptr,
      std::enable_if_t<std::is_convertible_v<
          typename std::invoke_result_t<Callable, Params...>,
          Ret>>* = nullptr)
      : callback(callback_fn<std::remove_reference_t<Callable>>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}

  Ret operator()(Params... params) const {
    return callback(callable, std::forward<Params>(params)...); // <------- CALLED HERE 1st
  }

  operator bool() const {
    return callback;
  }
};
```

#### ` aten/src/ATen/native/cpu/Loops.h`

- `VectorizedLoop2d`

```c++
template <typename op_t, typename vop_t>
struct VectorizedLoop2d {
  op_t op;
  vop_t vop;

  using traits = function_traits<op_t>;
  static constexpr int ntensors = traits::arity + 1;
  using data_t = std::array<char*, ntensors>;

  VectorizedLoop2d(const op_t &op, vop_t vop):
    op(op), vop(std::move(vop)) {}

  static void advance(data_t &data, const int64_t *outer_strides) {
    for (const auto arg : c10::irange(data.size())) {
      data[arg] += outer_strides[arg];
    }
  }

  void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
    data_t data;
    std::copy_n(base, ntensors, data.data());
    const int64_t *outer_strides = &strides[ntensors];

    if (is_contiguous<traits>(strides)) {
      for (const auto i C10_UNUSED : c10::irange(size1)) {
        vectorized_loop(data.data(), size0, 0, op, vop); // <------- CALLED HERE 1st
        advance(data, outer_strides);
      }
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          for (const auto i C10_UNUSED : c10::irange(size1)) {
            vectorized_loop(data.data(), size0, idx, op, vop);
            advance(data, outer_strides);
          }
        } else {
          for (const auto i C10_UNUSED : c10::irange(size1)) {
            basic_loop(data.data(), strides, 0, size0, op);
            advance(data, outer_strides);
          }
        }
      });
    }
  }
};
```

- `vectorized_loop`

```c++
template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
  using traits = function_traits<vec_func_t>;
  using scalar_t = typename function_traits<func_t>::result_type;
  using Vec = Vectorized<scalar_t>;
  constexpr int ntensors = traits::arity + 1;

  char* C10_RESTRICT data[ntensors];
  for (const auto arg : c10::irange(ntensors)) {
    data[arg] = data_[arg];
  }

  Vec opt_scalar = Vec(S > 0 ? *(scalar_t*)data[S] : scalar_t(0));
  int64_t i = 0;
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    auto args1 = dereference_vec<traits>(&data[1], opt_scalar, S, i);
    auto args2 = dereference_vec<traits>(&data[1], opt_scalar, S, i + Vec::size());
    auto out1 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args1));
    auto out2 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args2));
    out1.store(data[0] + i * sizeof(scalar_t));
    out2.store(data[0] + (i + Vec::size()) * sizeof(scalar_t));
  }
  if (i < n) {
    int64_t strides[ntensors];
    for (const auto arg : c10::irange(ntensors)) {
      strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(scalar_t);
    }
    basic_loop(data, strides, i, n, std::forward<func_t>(op)); // <------- CALLED HERE 2nd
  }
}
```

- `basic_loop`

```c++
template <typename func_t>
static inline void
basic_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  constexpr int ntensors = traits::arity + 1;

  // Copying strides to temporary array helps auto vectorization in older GCC
  // versions.
  int64_t strides[ntensors];
  for (const auto arg : c10::irange(ntensors)) {
    strides[arg] = strides_[arg];
  }

  execute_op(data, strides, i, n, std::forward<func_t>(op)); // <------- CALLED HERE 3rd
}
```

- `execute_op`

```c++
template <typename func_t,
    typename std::enable_if<!std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
static inline void
execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  using result_type = typename traits::result_type;
  for (; i < n; i++) {
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    *out_ptr = c10::guts::apply(std::forward<func_t>(op), dereference<traits>( // <------- CALLED HERE 4th
        &data[1],
        &strides[1],
        i));
  }
}
```

#### `c10/util/C++17.h`

```c++
template <class F, class Tuple>
C10_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t)); // <------- CALLED HERE
}
```

#### [`build/aten/src/ATen/UfuncCPUKernel_add.cpp`](#`build/aten/src/ATen/UfuncCPUKernel_add.cpp`)

#### `aten/src/ATen/native/ufunc/add.h`

```c++
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
  return self + alpha * other; // <------- CALLED HERE
}
```

