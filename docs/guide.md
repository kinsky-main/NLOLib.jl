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

`load()` resolves the library from `ENV["NLOLIB_LIBRARY"]`, then the installed
artifact, then a build tree; `load(path)` uses that file only. Release and
registration details are in [release.md](release.md).

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
nt, dt = 1024, 0.01
t = ((0:nt-1) .- nt / 2) .* dt

result = propagate(
    PulseSpec(samples = ComplexF64.(exp.(-(t ./ 0.2) .^ 2)), delta_time = dt),
    (A, w) -> im * (beta2 / 2) * w^2 - alpha / 2,
    (A, I) -> im * gamma * A * I;
    propagation_distance = 1.0,
    num_records = 128,
)
```

Each function is traced once into the compiled operator program, so captured
numbers are inlined and the body may use `+ - * / ^`, unary minus and
`exp log sqrt sin cos`. The binding writes the linear operator to the C
`dispersion_factor_expr` for a temporal run and to `linear_factor_expr` for a
tensor run, so one `(A, w)` function serves both. `potential_operator` (tensor
runs only) and `linear_step_operator` fill the other two slots, and
`constant_bindings = Dict("gamma" => gamma)` gives a named value its own
runtime constant slot.

The symbols each operator may take:

| Operator | Temporal run | Tensor run |
|---|---|---|
| `linear_operator` | `A, w` | `A, wt, kx, ky, x, y, t` |
| `linear_step_operator` | `A, D, h, w, V` | `A, D, h, wt, kx, ky, x, y, t, V` |
| `potential_operator` | not allowed | `x, y, t` |
| `nonlinear_operator` | `A, I, V` | `A, I, V` |

`w` and `wt` are interchangeable in the linear operators. Any other parameter
name binds to the next unused symbol in that order, and a symbol from the wrong
column is an `ArgumentError`.

## Results and errors

`propagate` returns a `PropagationResult` with `records` (samples x records),
`z_axis`, `final`, `step_history` (with `capture_step_history = true`),
`storage_result` (with `storage`) and `meta`, a `Dict` holding `"preset"`,
`"records_requested"`, `"records_written"`, `"backend_requested"`,
`"coupled"` and `"status"`.

A bad pulse, operator or option value is an `ArgumentError`, including an
enum string such as `backend_type = "gpu"`; an unknown option is Julia's own
`MethodError`. A library failure is an `NLOLibError` whose `status` is the C
status as a string (`"invalid_argument"`, `"allocation_failed"`, ...) and
`nothing` when the library could not be loaded. A `progress_callback` that
returns `false` aborts the run with `PropagationAbortedError`, whose `result`
holds what was recorded.

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

## Perf-Profile Counters

`perf_profile_set_enabled(true)` starts accumulating runtime performance
counters, `perf_profile_read()` returns them as a `PerfProfileSnapshot` and
`perf_profile_reset()` zeroes them. `operator_jit_programs` and
`operator_jit_fallbacks` report whether the Vulkan backend compiled each
operator program to a GPU kernel or fell back to the interpreter, and they
accumulate even while profiling is off; see
[GPU backend behaviour](../../docs/gpu_backend.md) for what the counters mean.

## Related Reference

- @ref julia_binding
