"""
    RuntimeOperators(; linear_factor_fn, linear_fn, potential_fn, dispersion_factor_fn, dispersion_fn, nonlinear_fn, ...)

Runtime operators for `prepare_sim_config`. Each operator is a Julia function of
its context's symbols, traced once at setup into the compiled operator VM; see
`NLOLib.runtime_context_symbols` for the symbols each slot accepts.
`dispersion_factor_fn` / `dispersion_fn` are aliases of `linear_factor_fn` /
`linear_fn` that also accept `w`. `constant_bindings` optionally maps names to
values that should each occupy a runtime constant slot.
"""
Base.@kwdef struct RuntimeOperators
    linear_factor_fn::Any = nothing
    linear_fn::Any = nothing
    potential_fn::Any = nothing
    dispersion_factor_fn::Any = nothing
    dispersion_fn::Any = nothing
    nonlinear_fn::Any = nothing
    nonlinear_model::Int = Int(NONLINEAR_MODEL_EXPR)
    nonlinear_gamma::Float64 = 0.0
    raman_fraction::Float64 = 0.0
    raman_tau1::Float64 = 0.0122
    raman_tau2::Float64 = 0.0320
    shock_omega0::Float64 = 0.0
    raman_response_time::Any = nothing
    constant_bindings::Union{Nothing, Dict{String, Float64}} = nothing
end

Base.@kwdef mutable struct PulseSpec
    samples::Vector{ComplexF64}
    delta_time::Float64
    pulse_period::Union{Nothing, Float64} = nothing
    frequency_grid::Union{Nothing, Vector{ComplexF64}} = nothing
    tensor_nt::Union{Nothing, Int} = nothing
    tensor_nx::Union{Nothing, Int} = nothing
    tensor_ny::Union{Nothing, Int} = nothing
    tensor_layout::Int = Int(TENSOR_LAYOUT_XYT_T_FAST)
    delta_x::Float64 = 1.0
    delta_y::Float64 = 1.0
    spatial_frequency_grid::Union{Nothing, Vector{ComplexF64}} = nothing
    potential_grid::Union{Nothing, Vector{ComplexF64}} = nothing
end

"""
    OperatorSpec(; fn, params = nothing)

A high-level operator: `fn` is a Julia function of the context's symbols and
`params` an optional `name => value` mapping bound as runtime constants.
"""
Base.@kwdef struct OperatorSpec
    fn::Any = nothing
    params::Any = nothing
end

struct ProgressInfo
    event_type::Int
    step_index::Int
    reject_attempt::Int
    z::Float64
    z_end::Float64
    percent::Float64
    step_size::Float64
    next_step_size::Float64
    error::Float64
    elapsed_seconds::Float64
    eta_seconds::Float64
end

struct PropagationResult
    records::Matrix{ComplexF64}
    z_axis::Vector{Float64}
    final::AbstractVector{ComplexF64}
    meta::Dict{String, Any}
end

struct PropagateResult
    records::Matrix{ComplexF64}
    z_axis::Vector{Float64}
    final::AbstractVector{ComplexF64}
    meta::Dict{String, Any}
    status::Int
    message::String
    t_events::Vector{Vector{Float64}}
    y_events::Vector{Matrix{ComplexF64}}
    sol::Any
end

struct PropagationAbortedError <: Exception
    result::PropagationResult
end

Base.showerror(io::IO, err::PropagationAbortedError) =
    print(io, "nlolib_propagate aborted by progress callback")
