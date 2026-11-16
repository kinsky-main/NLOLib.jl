# Julia Binding Guide

The Julia package exposes the native wrapper through the exported `NLOLib`
module.

## Install

```julia
using Pkg
Pkg.add("NLOLib")
```

The native library and its runtime dependencies arrive as a Julia artifact, so
nothing is compiled on install and no Vulkan SDK is needed. On Linux the Vulkan
loader (`libvulkan.so.1`) must be present because it is a plain shared-library
dependency; on Windows it is delay-loaded and optional. Without a Vulkan driver
the CPU backend runs unchanged.

`NLOLib.load()` resolves the library from an explicit path, then
`ENV["NLOLIB_LIBRARY"]`, then the installed artifact, then a build tree. Release
and registration details are in [release.md](release.md).

## Minimal Math-to-API Example

This example uses the quadratic GLSE mapping

\f[
D(\omega)=i\left(\frac{\beta_2}{2}\right)\omega^2-\frac{\alpha}{2},
\qquad
N(A)=i\gamma A|A|^2
\f]

so the operators are written as Julia functions of the runtime symbols:

```julia
using NLOLib

beta2 = -0.02
alpha = 0.10
gamma = 1.20

phys = NLOLib.physics_config(
    dispersion_factor_fn = (A, w) -> im * (beta2 / 2) * w^2 - alpha / 2,
    nonlinear_fn = (A, I) -> im * gamma * A * I,
)
```

Each function is traced once into the compiled operator program, so captured
numbers are inlined and the body may use `+ - * / ^`, unary minus and
`exp log sqrt sin cos`. `NLOLib.runtime_context_symbols(context)` lists the
symbols each slot accepts. The high-level `propagate(pulse, linear, nonlinear)`
takes the same kind of functions, or an `OperatorSpec(fn = f, params = Dict(...))`
to give named values their own runtime constant slot.

## Build-Tree Usage

After a core build with this binding checked out (see
[development.md](development.md)):

```bash
cmake --build build --config Release --target julia_stage
```

Then in Julia:

```julia
using Pkg
Pkg.activate("build/julia_package")
using NLOLib
NLOLib.load()
```

## Public API Entry Points

- `NLOLib.NLolib`
- `NLOLib.RuntimeOperators`
- `NLOLib.PulseSpec`
- `NLOLib.physics_config`
- `NLOLib.query_runtime_limits`
- `NLOLib.propagate`
- `NLOLib.propagate!`

## Not Exposed Here: the Perf-Profile Counters

The C API, Python and MATLAB can read runtime performance counters through
`nlolib_perf_profile_read`, including the two that report whether the Vulkan
backend compiled each operator program to a GPU kernel or fell back to the
interpreter (`operator_jit_programs` and `operator_jit_fallbacks`). This package
wraps no part of that surface, so there is no Julia equivalent to show; see
[GPU backend behaviour](../../docs/gpu_backend.md) for what the counters mean and
the Python guide for the shape of the call.

## Related Reference

- @ref julia_binding
