struct NormalizedPropagateRequest
    sim_cfg::SimulationConfig
    phys_cfg::PhysicsConfig
    input_seq::Vector{ComplexF64}
    keepalive::Vector{Any}
    num_records::Int
    output_mode::Cint
    record_z::Union{Nothing, Vector{Float64}}
    z_axis::Vector{Float64}
    exec_options::Union{Nothing, CExecutionOptions}
    storage_options::Union{Nothing, CStorageOptions}
    return_records::Bool
    step_history_capacity::Int
    progress_callback::Union{Nothing, Function}
    meta::Dict{String, Any}
end

function _positive(value, name::AbstractString)
    value isa Real && isfinite(value) && value > 0 ||
        throw(ArgumentError("$(name) must be a finite number > 0; got $(repr(value))"))
    return Float64(value)
end

_positive_or(value, default::Float64, name::AbstractString) =
    value === nothing ? default : _positive(value, name)

function _flag(value, name::AbstractString)
    value isa Bool || throw(ArgumentError("$(name) must be true or false; got $(repr(value))"))
    return value
end

"""
    propagate(pulse, linear_operator, nonlinear_operator; propagation_distance, ...)

Propagate `pulse` (a `PulseSpec`) over `propagation_distance` and return a
`PropagationResult`.

The operators are Julia functions of their context's symbols, traced once at
setup into the expressions the library compiles; captured numbers are
inlined. `linear_operator` takes `(A, w)` for a temporal run and
`(A, wt, kx, ky, x, y, t)` for a tensor run (`w` and `wt` name the same axis),
`nonlinear_operator` takes `(A, I, V)`. A tensor run may add
`potential_operator(x, y, t)`; `linear_step_operator` replaces the per-step
linear operator `(A, D, h, ...)`. `constant_bindings` maps names to values that
each spend one of the 16 runtime constant slots.

Records: with `record_z`, one record at each of those positions; otherwise
`num_records == 1` returns only the final field, and any other `num_records`
that many records spaced evenly from 0 to `propagation_distance`. `preset`
(`"fast"`, `"balanced"` or `"accuracy"`) supplies the step sizes, tolerance and
`num_records` an explicit option does not set.

`exec_options` is an `ExecutionOptions` and `storage` a `StorageOptions`;
`return_records = false` sends the records to `storage` only.
`progress_callback(info::ProgressInfo)` is called during the run, and returning
`false` from it aborts with `PropagationAbortedError`.

A bad pulse, operator or option value throws `ArgumentError`, an unknown option
`MethodError`, and a library failure `NLOLibError`.
"""
function propagate(pulse, linear_operator, nonlinear_operator;
                   propagation_distance,
                   preset = "balanced",
                   starting_step_size = nothing,
                   max_step_size = nothing,
                   min_step_size = nothing,
                   error_tolerance = nothing,
                   num_records = nothing,
                   record_z = nothing,
                   return_records = true,
                   potential_operator = nothing,
                   linear_step_operator = nothing,
                   constant_bindings = nothing,
                   nonlinear_model = "expr",
                   nonlinear_gamma = 0.0,
                   raman_fraction = 0.0,
                   raman_tau1 = 0.0122,
                   raman_tau2 = 0.0320,
                   shock_omega0 = 0.0,
                   raman_response_time = nothing,
                   exec_options = nothing,
                   storage = nothing,
                   capture_step_history = false,
                   step_history_capacity = 200000,
                   progress_callback = nothing)
    spec = _normalize_pulse_spec(pulse)
    tensor_mode = _is_tensor_run(spec)
    distance = _positive(propagation_distance, "propagation_distance")
    profile = _solver_profile_defaults(preset, distance)
    propagation = PropagationParams(
        _positive_or(starting_step_size, profile.starting_step_size, "starting_step_size"),
        _positive_or(max_step_size, profile.max_step_size, "max_step_size"),
        _positive_or(min_step_size, profile.min_step_size, "min_step_size"),
        _positive_or(error_tolerance, profile.error_tolerance, "error_tolerance"),
        distance,
    )

    num_records === nothing || num_records isa Integer && num_records > 0 ||
        throw(ArgumentError("num_records must be an integer > 0; got $(repr(num_records))"))
    explicit_z = record_z === nothing ? nothing : _validate_record_z(record_z, distance)
    if explicit_z !== nothing
        num_records === nothing || num_records == length(explicit_z) ||
            throw(ArgumentError("num_records ($(num_records)) must be unset or equal length(record_z) ($(length(explicit_z)))"))
        count = length(explicit_z)
        mode = PROPAGATE_OUTPUT_DENSE
        # Core 9417e92 returns the field at propagation_distance whenever one
        # record is requested, so a single position reports where that field is.
        z_axis = count == 1 ? [distance] : explicit_z
    else
        count = num_records === nothing ? profile.num_records : Int(num_records)
        mode = count == 1 ? PROPAGATE_OUTPUT_FINAL_ONLY : PROPAGATE_OUTPUT_DENSE
        z_axis = count == 1 ? [distance] : collect(range(0.0, distance, length = count))
    end

    records_returned = _flag(return_records, "return_records")
    records_returned || storage !== nothing ||
        throw(ArgumentError("return_records = false needs storage = StorageOptions(...) to receive the records"))
    capture = _flag(capture_step_history, "capture_step_history")
    capture && !(step_history_capacity isa Integer && step_history_capacity > 0) &&
        throw(ArgumentError("step_history_capacity must be an integer > 0; got $(repr(step_history_capacity))"))
    progress_callback === nothing || progress_callback isa Function ||
        throw(ArgumentError("progress_callback must be a function of a ProgressInfo"))

    potential_operator === nothing || tensor_mode ||
        throw(ArgumentError("potential_operator needs a tensor run (pulse.tensor_nt, tensor_nx and tensor_ny); " *
                            "a temporal run does not compile it"))
    potential_operator === nothing || spec.potential_grid === nothing ||
        throw(ArgumentError("set potential_operator or pulse.potential_grid, not both; the grid takes precedence"))
    for (name, value) in (("nonlinear_gamma", nonlinear_gamma), ("raman_fraction", raman_fraction),
                          ("raman_tau1", raman_tau1), ("raman_tau2", raman_tau2), ("shock_omega0", shock_omega0))
        value isa Real && isfinite(value) || throw(ArgumentError("$(name) must be a finite number; got $(repr(value))"))
    end

    sim_cfg, keepalive = _prepare_sim_config(spec, propagation)
    phys_cfg, physics_keepalive = _prepare_physics(tensor_mode;
        linear_operator = linear_operator,
        nonlinear_operator = nonlinear_operator,
        linear_step_operator = linear_step_operator,
        potential_operator = potential_operator,
        constant_bindings = constant_bindings,
        nonlinear_model = nonlinear_model,
        nonlinear_gamma = nonlinear_gamma,
        raman_fraction = raman_fraction,
        raman_tau1 = raman_tau1,
        raman_tau2 = raman_tau2,
        shock_omega0 = shock_omega0,
        raman_response_time = raman_response_time,
    )
    append!(keepalive, physics_keepalive)
    exec_raw = _c_execution_options(exec_options)
    storage_raw = _c_storage_options(storage)
    push!(keepalive, storage)

    meta = Dict{String, Any}(
        "preset" => String(preset),
        "records_requested" => count,
        "backend_requested" => exec_options === nothing ? "auto" : exec_options.backend_type,
        "coupled" => tensor_mode && (spec.tensor_nx > 1 || spec.tensor_ny > 1),
    )
    request = NormalizedPropagateRequest(
        sim_cfg,
        phys_cfg,
        spec.samples,
        keepalive,
        count,
        mode,
        explicit_z,
        z_axis,
        exec_raw,
        storage_raw,
        records_returned,
        capture ? Int(step_history_capacity) : 0,
        progress_callback,
        meta,
    )
    return _execute(request)
end
