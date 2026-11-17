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
                   output = "final")

println("peak power in:  ", maximum(abs2, pulse.samples))
println("peak power out: ", maximum(abs2, result.final))
```

Each function is traced once, at setup, into the compiled operator program, so
its body may use captured numbers and ordinary arithmetic (`+ - * / ^`, `exp`,
`log`, `sqrt`, `sin`, `cos`) but cannot branch on the field. The linear operator
may take `A`, `w`, `wt`; tensor runs use `(A, wt, kx, ky)`; the nonlinear
operator takes `A`, `I`, `V`. Arguments bind by name, so `(w, A) -> ...` works.

The main entry points are `NLOLib.NLolib`, `NLOLib.RuntimeOperators`,
`NLOLib.PulseSpec`, `NLOLib.physics_config`, `NLOLib.query_runtime_limits`,
`NLOLib.propagate` and `NLOLib.propagate!`.

`NLOLib.load()` resolves the library from an explicit path, then
`ENV["NLOLIB_LIBRARY"]`, then the installed artifact.

## Examples

The [`examples/`](examples) directory is a Julia project of its own with
runnable scripts: second-order and high-order GRIN solitons, Raman scattering,
and 3D tensor dispersion.

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/second_order_soliton_rk4ip.jl
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
