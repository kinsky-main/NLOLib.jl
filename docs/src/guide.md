# Julia Binding Guide

The Julia package exposes the native wrapper through the exported `NonlinearOptics`
module.

## Install

```julia
using Pkg
Pkg.add("NonlinearOptics")
```

The native library and its runtime dependencies arrive as a Julia artifact, so
nothing is compiled on install and no Vulkan SDK is needed. On Linux the Vulkan
loader (`libvulkan.so.1`) must be present because it is a plain shared-library
dependency; on Windows it is delay-loaded and optional. Without a Vulkan driver
the CPU backend runs unchanged.

`load()` resolves the library from `ENV["NLOLIB_LIBRARY"]`, then the installed
artifact, then a build tree; `load(path)` uses that file only. Release and
registration details are in [release.md](https://github.com/kinsky-main/NonlinearOptics.jl/blob/main/docs/release.md).

## Minimal Math-to-API Example

This example uses the quadratic GLSE mapping

```math
D(\omega)=i\left(\frac{\beta_2}{2}\right)\omega^2-\frac{\alpha}{2},
\qquad
N(A)=i\gamma A|A|^2
```

so the operators are written as Julia functions of the runtime symbols:

```julia
using NonlinearOptics

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

Parameters named after one of these symbols bind to it first, wherever they
stand; the rest then take the unused symbols in the order shown, so
`(B, A) -> ...` binds `A` to `A` and `B` to `w`. `w` and `wt` are one symbol in
the linear operators. A symbol from the wrong column, two parameters naming
the same symbol (such as `w` and `wt`), or more parameters than symbols is an
`ArgumentError`. A free name in the body is always a captured constant.

## Results and errors

`propagate` returns a `PropagationResult` with `records` (samples x records),
`z_axis`, `final`, `step_history` (with `capture_step_history = true`),
`storage_result` (with `storage`) and `meta`, a `Dict` holding `"preset"`,
`"records_requested"`, `"records_written"`, `"backend_requested"`,
`"coupled"` and `"status"`.

`final` is the last returned record, `records[:, end]`, not a separate
snapshot of the field at `propagation_distance`. With `num_records` the last
record is at `propagation_distance`, so the two agree; with a `record_z` whose
last position is earlier, `final` is the field at that position. A single
`record_z` position returns the field at exactly that position.

A bad pulse, operator or option value is an `ArgumentError`, including an
enum string such as `backend_type = "gpu"`; an unknown option is Julia's own
`MethodError`. A library failure is an `NLOLibError` whose `status` is the C
status as a string (`"invalid_argument"`, `"allocation_failed"`, ...) and
`nothing` when the library could not be loaded. A `progress_callback` that
returns `false` aborts the run with `PropagationAbortedError`, whose `result`
holds what was recorded.

## Build-Tree Usage

After a core build with this binding checked out (see
[development.md](https://github.com/kinsky-main/NonlinearOptics.jl/blob/main/docs/development.md)):

```bash
cmake --build build --config Release --target julia_stage
```

Then in Julia:

```julia
using Pkg
Pkg.activate("build/julia_package")
using NonlinearOptics
NonlinearOptics.load()
```

## Perf-Profile Counters

`perf_profile_set_enabled(true)` starts accumulating runtime performance
counters, `perf_profile_read()` returns them as a `PerfProfileSnapshot` and
`perf_profile_reset()` zeroes them. `operator_jit_programs` and
`operator_jit_fallbacks` report whether the Vulkan backend compiled each
operator program to a GPU kernel or fell back to the interpreter, and they
accumulate even while profiling is off; see
[GPU backend behaviour](https://github.com/kinsky-main/nlo-rk4ip-code/blob/main/docs/gpu_backend.md) for what the counters mean.

## Upgrading from 1.x

2.0.0 moves the package to the API shared with the Python and MATLAB
bindings and breaks 1.x code.

- One exported entry point, `propagate(pulse, linear_operator,
  nonlinear_operator; ...)`. Both operators are required and must be Julia
  functions; string operators and `OperatorSpec` are gone.
- No `NLolib` client struct: every `f(::NLolib, ...)` method is now a plain
  function, and `load(path)` selects a library.
- Every option is a named keyword, so a misspelled one is a `MethodError`.
  `t_span`, `first_step`, `max_step`, `min_step`, `rtol`, `dense_output` and
  `events` are removed.
- Enum-style values are lowercase strings (`backend_type = "cpu"`,
  `set_log_level("warn")`, `meta["status"] == "ok"`). The integer constants
  (`VECTOR_BACKEND_*`, `FFT_BACKEND_*`, `NLOLIB_STATUS_*`, ...) and `NT_MAX`
  are no longer exported; `query_runtime_limits` replaces `NT_MAX`.
- Records: `records=`/`num_recorded_samples=` become `num_records`, and
  `t_eval`/`explicit_record_z` become `record_z`; `output`/`output_mode` are
  removed. See "Which records come back" in the core's binding vocabulary.
- `propagate!`, `propagate(config, input)`, `propagate(sim, phys, input)`,
  `prepare_sim_config` and the C-layout structs are private.

| 1.x | 2.0.0 |
|---|---|
| `linear_factor_fn`, `nonlinear_fn`, `potential_fn`, `linear_fn` | `linear_operator`, `nonlinear_operator`, `potential_operator`, `linear_step_operator` |
| `dispersion_factor_fn`, `dispersion_fn` | removed; the binding picks the C context by run type |
| `auto_capture_constants`, raw `constants` | removed; capture is always on, `constant_bindings` names values |
| six storage keywords | `storage = StorageOptions(...)` |
| `default_execution_options(...)` | `ExecutionOptions(; ...)` |
| `storage_options`, `default_storage_options`, `physics_config`, `PreparedValue`, `wrap_records`, `final_record` | private |
| `PropagateResult` | `PropagationResult`; `step_history` and `storage_result` are fields, not `meta` keys |
| `query_runtime_limits(config, ...)` | `query_runtime_limits(; pulse, exec_options)` |
| `tensor_record_view(records, index, nt, nx, ny)` | `tensor_record_view(record, nt, nx, ny)` |
| `set_log_file(path, append)` | `set_log_file(path; append = false)` |
| `set_progress_stream(; stream_mode)` | `set_progress_stream(stream_mode)` |
| `error(...)` on a status failure | `NLOLibError` with a `status` string |
| `NLOComplex`, `SimulationConfig`, `PhysicsConfig`, `Tensor3DDesc`, the other C structs | private |
| (new) | `perf_profile_*`, `PerfProfileSnapshot`, `StepHistory`, `loaded_library_path` |

Behaviour changes:

- A tensor run with an `nx*ny` `potential_grid` is now broadcast along `t`.
  1.x tiled the grid across the volume instead, so such runs were wrong.
- The number of records a run may keep in memory now follows the host's (and
  device's) available memory and the run's grid instead of a fixed count;
  `query_runtime_limits(pulse = ...)` reports it for a given pulse.
- A single `record_z` position returns the field at that position, with
  `z_axis == [p]`, not the field at `propagation_distance`.
- `result.final` is the last returned record, which is the field at
  `propagation_distance` only when the records reach it.

## Related Reference

- [API reference](api.md)
