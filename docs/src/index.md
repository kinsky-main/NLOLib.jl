# NonlinearOptics.jl

Julia package for the nlolib nonlinear-optics runtime: a split-step (RK4IP)
propagator for pulses and beams in nonlinear media, with a CPU backend and an
optional Vulkan GPU backend.

## Install

```julia
using Pkg
Pkg.add("NonlinearOptics")
```

The native library comes from `nlolib_jll`, so nothing is compiled on install.
Without a Vulkan driver the CPU backend runs unchanged.

## Quick start

Operators are plain Julia functions of the field and grid symbols. This
propagates a second-order soliton (``D(\omega) = i(\beta_2/2)\omega^2``,
``N(A) = i\gamma A|A|^2``) over one soliton period, after which it returns to
its input shape:

```jldoctest quickstart
julia> using NonlinearOptics

julia> set_log_level("error"); set_progress_options(enabled = false)

julia> beta2, gamma = -1.0, 1.0;

julia> nt, dt = 1024, 0.02;

julia> t = ((0:nt-1) .- nt / 2) .* dt;

julia> pulse = PulseSpec(samples = ComplexF64.(2 ./ cosh.(t)), delta_time = dt);

julia> result = propagate(pulse,
                          (A, w) -> im * (beta2 / 2) * w^2,  # linear (dispersion) operator
                          (A, I) -> im * gamma * A * I;      # nonlinear operator, I = |A|^2
                          propagation_distance = pi / 2,
                          num_records = 1,                   # the final field only
                          exec_options = ExecutionOptions(backend_type = "cpu"));

julia> round(maximum(abs2, result.final), digits = 2)  # peak power in: 4.0
4.0
```

The [Guide](guide.md) covers the operator symbols, records, results and
errors, and the changes from 1.x; the [API reference](api.md) documents every
exported name.

## Links

- Source and issues: <https://github.com/kinsky-main/NonlinearOptics.jl>
- Core C library: <https://github.com/kinsky-main/nlo-rk4ip-code>
