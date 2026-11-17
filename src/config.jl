function _complex_vector_or_nothing(values)
    values === nothing && return nothing
    return ComplexF64.(collect(values))
end

function _default_frequency_grid(num_time_samples::Integer, delta_time::Real)
    two_pi = 2.0 * π
    omega_step = two_pi / (Float64(num_time_samples) * Float64(delta_time))
    positive_limit = (Int(num_time_samples) - 1) ÷ 2
    out = Vector{ComplexF64}(undef, Int(num_time_samples))
    for i in eachindex(out)
        idx = i - 1
        omega = idx <= positive_limit ? Float64(idx) * omega_step : -(Float64(num_time_samples - idx) * omega_step)
        out[i] = ComplexF64(omega, 0.0)
    end
    return out
end

_is_tensor_run(pulse::PulseSpec) = pulse.tensor_nt !== nothing

"""
    _normalize_pulse_spec(pulse)

Check `pulse` and return it; every inconsistency is an `ArgumentError`.
"""
function _normalize_pulse_spec(pulse)
    pulse isa PulseSpec || throw(ArgumentError("pulse must be a PulseSpec; got $(typeof(pulse))"))
    isempty(pulse.samples) && throw(ArgumentError("pulse.samples must be non-empty"))
    pulse.delta_time > 0.0 || throw(ArgumentError("pulse.delta_time must be > 0"))
    total = length(pulse.samples)
    shape = (pulse.tensor_nt, pulse.tensor_nx, pulse.tensor_ny)
    if any(!isnothing, shape)
        all(!isnothing, shape) ||
            throw(ArgumentError("pulse.tensor_nt, tensor_nx and tensor_ny must be set together"))
        all(>(0), shape) || throw(ArgumentError("pulse.tensor_nt, tensor_nx and tensor_ny must be > 0"))
        prod(shape) == total ||
            throw(ArgumentError("length(pulse.samples) must equal tensor_nt*tensor_nx*tensor_ny"))
    end
    nt, nx, ny = _is_tensor_run(pulse) ? shape : (total, 1, 1)
    check(name, values, lengths) = values === nothing || length(values) in lengths ||
        throw(ArgumentError("pulse.$(name) must have length " * join(unique(lengths), " or ") * "; got $(length(values))"))
    check("frequency_grid", pulse.frequency_grid, (nt, total))
    check("wt_axis", pulse.wt_axis, (nt,))
    check("kx_axis", pulse.kx_axis, (nx,))
    check("ky_axis", pulse.ky_axis, (ny,))
    check("spatial_frequency_grid", pulse.spatial_frequency_grid, _is_tensor_run(pulse) ? (nx * ny, total) : (total,))
    check("potential_grid", pulse.potential_grid, _is_tensor_run(pulse) ? (nx * ny, total) : (total,))
    return pulse
end

function _solver_profile_defaults(profile, propagation_distance::Float64)
    distance = propagation_distance
    if profile == "balanced"
        return (
            starting_step_size = distance / 200.0,
            max_step_size = distance / 25.0,
            min_step_size = distance / 20000.0,
            error_tolerance = 1e-6,
            num_records = 128,
        )
    elseif profile == "fast"
        return (
            starting_step_size = distance / 120.0,
            max_step_size = distance / 12.0,
            min_step_size = distance / 4000.0,
            error_tolerance = 5e-6,
            num_records = 64,
        )
    elseif profile == "accuracy"
        return (
            starting_step_size = distance / 400.0,
            max_step_size = distance / 50.0,
            min_step_size = distance / 80000.0,
            error_tolerance = 1e-7,
            num_records = 192,
        )
    end
    throw(ArgumentError("preset must be one of \"fast\", \"balanced\", \"accuracy\"; got $(repr(profile))"))
end

# A sim config needs propagation parameters even when only its grid matters,
# as for query_runtime_limits.
const _QUERY_PROPAGATION = PropagationParams(1.0 / 200.0, 1.0 / 25.0, 1.0 / 20000.0, 1e-6, 1.0)

_cfg_complex_ptr(values::Vector{ComplexF64}) =
    Base.unsafe_convert(Ptr{NLOComplex}, reinterpret(NLOComplex, values))
_cfg_complex_ptr_or_null(values) = values === nothing ? C_NULL : _cfg_complex_ptr(values)

"""
    _prepare_sim_config(pulse, propagation)

The C `SimulationConfig` for a checked `pulse`, plus the arrays its pointers
refer to, which the caller keeps alive for as long as it uses the config.
"""
function _prepare_sim_config(pulse::PulseSpec, propagation::PropagationParams)
    keepalive = Any[]
    total = length(pulse.samples)
    tensor_mode = _is_tensor_run(pulse)
    nt, nx, ny = tensor_mode ? (pulse.tensor_nt, pulse.tensor_nx, pulse.tensor_ny) : (total, 1, 1)

    keep(values) = (values === nothing || push!(keepalive, values); values)
    frequency = keep(pulse.frequency_grid === nothing ? _default_frequency_grid(nt, pulse.delta_time) : pulse.frequency_grid)
    wt_values = keep(pulse.wt_axis)
    kx_values = keep(pulse.kx_axis)
    ky_values = keep(pulse.ky_axis)
    spatial_values = keep(pulse.spatial_frequency_grid)
    potential_values = pulse.potential_grid
    if tensor_mode && potential_values !== nothing && length(potential_values) == nx * ny
        potential_values = repeat(potential_values, nt)
    end
    keep(potential_values)

    pulse_period = pulse.pulse_period === nothing ? pulse.delta_time * nt : pulse.pulse_period
    tensor = tensor_mode ?
        Tensor3DDesc(Csize_t(nt), Csize_t(nx), Csize_t(ny), TENSOR_LAYOUT_XYT_T_FAST) :
        Tensor3DDesc()
    time = TimeGrid(Csize_t(nt), Cdouble(pulse_period), Cdouble(pulse.delta_time), _cfg_complex_ptr_or_null(wt_values))
    spatial = SpatialGrid(
        tensor_mode ? Csize_t(nx) : Csize_t(0),
        tensor_mode ? Csize_t(ny) : Csize_t(0),
        Cdouble(pulse.delta_x),
        Cdouble(pulse.delta_y),
        _cfg_complex_ptr_or_null(spatial_values),
        _cfg_complex_ptr_or_null(kx_values),
        _cfg_complex_ptr_or_null(ky_values),
        _cfg_complex_ptr_or_null(potential_values),
    )
    sim = SimulationConfig(propagation, tensor, time, FrequencyGrid(_cfg_complex_ptr(frequency)), spatial)
    return sim, keepalive
end

function _shift_constant_indices(expression::AbstractString, offset::Integer)
    offset == 0 && return String(expression)
    return replace(String(expression), r"\bc(\d+)\b" => s -> "c$(parse(Int, s.captures[1]) + offset)")
end

"""
    _constant_bindings(bindings)

`constant_bindings` as a `Dict{String, Float64}`, or `nothing` when unset.
"""
function _constant_bindings(bindings)
    bindings === nothing && return nothing
    bindings isa AbstractDict || bindings isa NamedTuple ||
        throw(ArgumentError("constant_bindings must map names to values, such as Dict(\"beta2\" => -0.02)"))
    out = Dict{String, Float64}()
    for (name, value) in pairs(bindings)
        value isa Real && isfinite(value) ||
            throw(ArgumentError("constant_bindings: '$(name)' must be a finite real number"))
        out[String(name)] = Float64(value)
    end
    return out
end

_operator_error(what, context, fn) = throw(ArgumentError(
    "$(what) must be a function of its symbols ($(join(runtime_context_symbols(context), ", "))), " *
    "such as (A, w) -> im * w^2; got $(repr(fn))"
))

"""
    _lower_operator!(constants, fn, context, bindings, what)

Trace `fn` in `context` into the expression text the C API compiles, appending
any bound constants to `constants` and shifting the text's `cN` indices past
the ones already there. Returns `nothing` when `fn` is `nothing`.
"""
function _lower_operator!(constants::Vector{Float64}, fn, context::AbstractString, bindings, what::AbstractString)
    fn === nothing && return nothing
    fn isa Function || _operator_error(what, context, fn)
    traced, traced_constants = translate_callable(fn, context; constant_bindings = bindings)
    expr = _shift_constant_indices(traced, length(constants))
    append!(constants, traced_constants)
    return expr
end

"""
    _prepare_physics(tensor_mode; linear_operator, nonlinear_operator, ...)

The C `PhysicsConfig` and the strings and arrays it points to. Each operator is
traced once into the C field the core compiles for the run type: a temporal
run compiles the dispersion expressions, a tensor run the linear and potential
ones, and both compile the nonlinear one.
"""
function _prepare_physics(tensor_mode::Bool;
                          linear_operator,
                          nonlinear_operator,
                          linear_step_operator,
                          potential_operator,
                          constant_bindings,
                          nonlinear_model,
                          nonlinear_gamma,
                          raman_fraction,
                          raman_tau1,
                          raman_tau2,
                          shock_omega0,
                          raman_response_time)
    linear_operator isa Function || _operator_error("linear_operator", RUNTIME_CONTEXT_DISPERSION_FACTOR, linear_operator)
    nonlinear_operator isa Function || _operator_error("nonlinear_operator", RUNTIME_CONTEXT_NONLINEAR, nonlinear_operator)
    bindings = _constant_bindings(constant_bindings)
    constants = Float64[]
    lower(fn, context, what) = _lower_operator!(constants, fn, context, bindings, what)
    if tensor_mode
        linear_factor_expr = lower(linear_operator, RUNTIME_CONTEXT_LINEAR_FACTOR, "linear_operator")
        linear_expr = lower(linear_step_operator, RUNTIME_CONTEXT_LINEAR, "linear_step_operator")
        potential_expr = lower(potential_operator, RUNTIME_CONTEXT_POTENTIAL, "potential_operator")
        dispersion_factor_expr = dispersion_expr = nothing
    else
        dispersion_factor_expr = lower(linear_operator, RUNTIME_CONTEXT_DISPERSION_FACTOR, "linear_operator")
        dispersion_expr = lower(linear_step_operator, RUNTIME_CONTEXT_DISPERSION, "linear_step_operator")
        linear_factor_expr = linear_expr = potential_expr = nothing
    end
    nonlinear_expr = lower(nonlinear_operator, RUNTIME_CONTEXT_NONLINEAR, "nonlinear_operator")
    length(constants) <= RUNTIME_OPERATOR_CONSTANTS_MAX || throw(ArgumentError(
        "constant_bindings uses $(length(constants)) constant slots; at most $(RUNTIME_OPERATOR_CONSTANTS_MAX) are available"
    ))

    model = _enum_code(_NONLINEAR_MODELS, nonlinear_model, "nonlinear_model")
    raman_response = _complex_vector_or_nothing(raman_response_time)
    keepalive = Any[linear_factor_expr, linear_expr, potential_expr, dispersion_factor_expr,
                    dispersion_expr, nonlinear_expr, raman_response]
    constants_tuple = ntuple(i -> i <= length(constants) ? constants[i] : 0.0, RUNTIME_OPERATOR_CONSTANTS_MAX)
    physics = PhysicsConfig(
        _string_ptr(linear_factor_expr),
        _string_ptr(linear_expr),
        _string_ptr(potential_expr),
        _string_ptr(dispersion_factor_expr),
        _string_ptr(dispersion_expr),
        _string_ptr(nonlinear_expr),
        model,
        Cdouble(nonlinear_gamma),
        Cdouble(raman_fraction),
        Cdouble(raman_tau1),
        Cdouble(raman_tau2),
        Cdouble(shock_omega0),
        _cfg_complex_ptr_or_null(raman_response),
        Csize_t(raman_response === nothing ? 0 : length(raman_response)),
        Csize_t(length(constants)),
        constants_tuple,
    )
    return physics, keepalive
end

"""
    _c_execution_options(options)

The C execution options for an `ExecutionOptions`, or `nothing` when unset.
"""
function _c_execution_options(options)
    options === nothing && return nothing
    options isa ExecutionOptions ||
        throw(ArgumentError("exec_options must be an ExecutionOptions; got $(typeof(options))"))
    options.device_heap_fraction > 0.0 && options.device_heap_fraction <= 1.0 ||
        throw(ArgumentError("exec_options.device_heap_fraction must be in (0, 1]"))
    options.record_ring_target >= 0 || throw(ArgumentError("exec_options.record_ring_target must be >= 0"))
    options.forced_device_budget_bytes >= 0 ||
        throw(ArgumentError("exec_options.forced_device_budget_bytes must be >= 0"))
    return CExecutionOptions(
        _enum_code(_VECTOR_BACKENDS, options.backend_type, "exec_options.backend_type"),
        _enum_code(_FFT_BACKENDS, options.fft_backend, "exec_options.fft_backend"),
        Cdouble(options.device_heap_fraction),
        Csize_t(options.record_ring_target),
        Csize_t(options.forced_device_budget_bytes),
        VulkanBackendConfig(),
    )
end

"""
    _c_storage_options(options)

The C storage options for a `StorageOptions`, or `nothing` when unset. The C
struct points into `options`' strings, which the caller keeps alive.
"""
function _c_storage_options(options)
    options === nothing && return nothing
    options isa StorageOptions || throw(ArgumentError("storage must be a StorageOptions; got $(typeof(options))"))
    isempty(options.sqlite_path) && throw(ArgumentError("storage.sqlite_path must be non-empty"))
    options.run_id === nothing || ncodeunits(options.run_id) < STORAGE_RUN_ID_MAX ||
        throw(ArgumentError("storage.run_id must be at most $(STORAGE_RUN_ID_MAX - 1) bytes"))
    options.sqlite_max_bytes >= 0 || throw(ArgumentError("storage.sqlite_max_bytes must be >= 0"))
    options.chunk_records >= 0 || throw(ArgumentError("storage.chunk_records must be >= 0"))
    storage_is_available() ||
        throw(NLOLibError("not_implemented", "SQLite storage is not available in this nlolib build"))
    return CStorageOptions(
        _string_ptr(options.sqlite_path),
        _string_ptr(options.run_id),
        Csize_t(options.sqlite_max_bytes),
        Csize_t(options.chunk_records),
        _enum_code(_CAP_POLICIES, options.cap_policy, "storage.cap_policy"),
        Cint(options.log_final_output_field_to_db),
    )
end

_storage_result(raw::CStorageResult) = StorageResult(
    _decode_run_id(raw.run_id),
    Int(raw.records_captured),
    Int(raw.records_spilled),
    Int(raw.chunks_written),
    Int(raw.db_size_bytes),
    raw.truncated != 0,
)

function _step_history(events::Vector{StepEvent}, written::Integer, dropped::Integer)
    n = max(0, Int(written))
    return StepHistory(
        [Int(events[i].step_index) for i in 1:n],
        [Float64(events[i].z_current) for i in 1:n],
        [Float64(events[i].step_size) for i in 1:n],
        [Float64(events[i].next_step_size) for i in 1:n],
        [Float64(events[i].error) for i in 1:n],
        Int(dropped),
        length(events),
    )
end

_progress_info(info::CProgressInfo) = ProgressInfo(
    _enum_name(_PROGRESS_EVENTS, info.event_type),
    Int(info.step_index),
    Int(info.reject_attempt),
    Float64(info.z),
    Float64(info.z_end),
    Float64(info.percent),
    Float64(info.step_size),
    Float64(info.next_step_size),
    Float64(info.error),
    Float64(info.elapsed_seconds),
    Float64(info.eta_seconds),
)

function _validate_record_z(z_values, distance::Float64)
    z_values isa AbstractVector{<:Real} || throw(ArgumentError("record_z must be a vector of positions"))
    values = Float64.(collect(z_values))
    isempty(values) && throw(ArgumentError("record_z must be non-empty"))
    all(z -> 0.0 <= z <= distance, values) ||
        throw(ArgumentError("record_z values must be within [0, propagation_distance]"))
    issorted(values) || throw(ArgumentError("record_z must be nondecreasing"))
    return values
end
