# NLOLib.jl

Julia package for the nlolib nonlinear-optics runtime: a split-step (RK4IP)
propagator for pulses and beams in nonlinear media, with a CPU backend and an
optional Vulkan GPU backend.

## Install

```julia
using Pkg
Pkg.add("NLOLib")
```

The native library and its runtime dependencies arrive as a Julia artifact, so
nothing is compiled on install and no Vulkan SDK is needed. Linux x86-64 and
Windows x86-64 are supported, on Julia 1.10 or later. On Linux the Vulkan
loader (`libvulkan.so.1`) must be present because it is a plain shared-library
dependency; on Windows it is optional. Without a Vulkan driver the CPU backend
runs unchanged.

## Quick start

Operators are plain Julia functions of the field and grid symbols. This
propagates a second-order soliton (`D(w) = i*(beta2/2)*w^2`,
`N(A) = i*gamma*A*|A|^2`) over one soliton period, after which it returns to its
input shape:

```julia
using NLOLib

beta2, gamma = -1.0, 1.0
nt, dt = 1024, 0.02
t = ((0:nt-1) .- nt / 2) .* dt
pulse = PulseSpec(samples = ComplexF64.(2 ./ cosh.(t)), delta_time = dt)

result = propagate(pulse,
                   (A, w) -> im * (beta2 / 2) * w^2,  # linear (dispersion) operator
                   (A, I) -> im * gamma * A * I;      # nonlinear operator, I = |A|^2
                   propagation_distance = pi / 2,
                   num_records = 1)  # 1 record: the final field only

println("peak power in:  ", maximum(abs2, pulse.samples))
println("peak power out: ", maximum(abs2, result.final))
```

Each function is traced once, at setup, into the compiled operator program, so
its body may use captured numbers and ordinary arithmetic (`+ - * / ^`, `exp`,
`log`, `sqrt`, `sin`, `cos`) but cannot branch on the field. The linear operator
takes `(A, w)` for a temporal run and `(A, wt, kx, ky, x, y, t)` for a tensor
run, where `w` and `wt` name the same axis; the nonlinear operator takes
`(A, I, V)`. Arguments bind by name, so `(w, A) -> ...` works, and a symbol the
run does not provide is an `ArgumentError`.

Every option of `propagate` is a named keyword, and a misspelled one is a
`MethodError`. Records come back at `record_z` when given, otherwise as
`num_records` evenly spaced records, where `num_records = 1` is the final
field only. `result.final` is the last returned record, so it is the field at
`propagation_distance` only when the records reach it: with a `record_z` that
ends earlier it is the field at the last listed position. Enum-style options
are lowercase strings:

```julia
result = propagate(pulse, linear, nonlinear;
                   propagation_distance = 1.0,
                   preset = "accuracy",
                   num_records = 64,
                   exec_options = ExecutionOptions(backend_type = "cpu"),
                   storage = StorageOptions(sqlite_path = "run.sqlite"))
```

The package exports `propagate`, the data types `PulseSpec`,
`ExecutionOptions`, `StorageOptions`, `PropagationResult`, `StepHistory`,
`StorageResult`, `ProgressInfo`, `RuntimeLimits` and `PerfProfileSnapshot`,
the errors `NLOLibError` and `PropagationAbortedError`, and the library
functions `load`, `loaded_library_path`, `query_runtime_limits`,
`storage_is_available`, `tensor_record_view`, the `set_log_*` /
`read_log_buffer` / `clear_log_buffer` / `set_progress_*` controls and the
`perf_profile_*` counters. Each has a docstring.

`load()` resolves the library from `ENV["NLOLIB_LIBRARY"]`, then the installed
artifact, then a build tree; `load(path)` uses that file only.

## Examples

The [`examples/`](examples) directory is a Julia project of its own holding the
example set shared with the Python and MATLAB bindings: `quickstart`,
`linear_drift`, `spm`, `second_order_soliton`, `conservation_checks`,
`raman_scattering`, `tensor_dispersion_3d`, `transverse_grin_beam`,
`spatiotemporal_kerr_bullet`, `grin_soliton_fission`,
`high_order_grin_soliton` and `sqlite_storage`. Each prints its validation
metrics as `label: value` and writes its figures to `examples/output/<name>/`.

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/quickstart.jl
```

## Documentation

- [Full documentation and API reference](https://kinsky-main.github.io/nlo-rk4ip-code/)
- [Binding guide](docs/guide.md)
- [Working on the package itself](docs/development.md) and
  [releasing](docs/release.md)

## Links

- Source and issues: https://github.com/kinsky-main/NLOLib.jl
- Core C library: https://github.com/kinsky-main/nlo-rk4ip-code

## License

GPL-2.0-or-later.
