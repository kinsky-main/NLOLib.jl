_string_ptr(value::Union{Nothing, AbstractString}) =
    value === nothing ? C_NULL : Base.unsafe_convert(Cstring, value)

function _decode_run_id(chars::NTuple{STORAGE_RUN_ID_MAX, Cchar})
    bytes = UInt8[UInt8(Int(c) & 0xff) for c in chars]
    stop = findfirst(==(0x00), bytes)
    data = stop === nothing ? bytes : bytes[1:stop - 1]
    return String(data)
end

function _check_status(status::Integer, context::AbstractString)
    status == NLOLIB_STATUS_OK && return
    throw(NLOLibError(_enum_name(_STATUSES, status), context))
end

"""
    storage_is_available()

Whether this nlolib build can write SQLite storage.
"""
function storage_is_available()
    status = ccall(_sym(:nlolib_storage_is_available), Cint, ())
    return status != 0
end

"""
    query_runtime_limits(; pulse = nothing, exec_options = nothing)

The library's resource limits as a `RuntimeLimits`: the largest time-sample
count and record counts a run can use, and the working-set estimate behind
them. With `pulse`, the estimate is for that pulse's grid; `exec_options` is an
`ExecutionOptions`.
"""
function query_runtime_limits(; pulse = nothing, exec_options = nothing)
    exec_raw = _c_execution_options(exec_options)
    prepared = pulse === nothing ? nothing : _prepare_sim_config(_normalize_pulse_spec(pulse), _QUERY_PROPAGATION)
    limits_ref = Ref(CRuntimeLimits())
    sim_ref = prepared === nothing ? nothing : Ref(prepared[1])
    exec_ref = exec_raw === nothing ? nothing : Ref(exec_raw)
    sim_ptr = sim_ref === nothing ? Ptr{SimulationConfig}(C_NULL) : Base.unsafe_convert(Ptr{SimulationConfig}, sim_ref)
    exec_ptr = exec_ref === nothing ? Ptr{CExecutionOptions}(C_NULL) : Base.unsafe_convert(Ptr{CExecutionOptions}, exec_ref)

    status = GC.@preserve prepared limits_ref sim_ref exec_ref begin
        ccall(
            _sym(:nlolib_query_runtime_limits),
            Cint,
            (Ptr{SimulationConfig}, Ptr{PhysicsConfig}, Ptr{CExecutionOptions}, Ref{CRuntimeLimits}),
            sim_ptr,
            Ptr{PhysicsConfig}(C_NULL),
            exec_ptr,
            limits_ref
        )
    end
    _check_status(status, "nlolib_query_runtime_limits")
    raw = limits_ref[]
    return RuntimeLimits(
        Int(raw.max_num_time_samples_runtime),
        Int(raw.max_num_recorded_samples_in_memory),
        Int(raw.max_num_recorded_samples_with_storage),
        Int(raw.estimated_required_working_set_bytes),
        Int(raw.estimated_device_budget_bytes),
        raw.storage_available != 0,
    )
end

"""
    tensor_record_view(record, nt, nx, ny)

One flat tensor record reshaped to `(nt, ny, nx)`, `t` fastest, without a copy.
"""
function tensor_record_view(record, nt, nx, ny)
    record isa AbstractVector || throw(ArgumentError("tensor_record_view: record must be a vector, such as result.records[:, k]"))
    length(record) == nt * nx * ny ||
        throw(ArgumentError("tensor_record_view: record length $(length(record)) does not match nt*nx*ny = $(nt * nx * ny)"))
    return reshape(record, Int(nt), Int(ny), Int(nx))
end

"""
    set_log_file(path; append = false)

Send the library's log to the file at `path`, or stop logging to a file when
`path` is `nothing`.
"""
function set_log_file(path; append = false)
    path === nothing || path isa AbstractString || throw(ArgumentError("set_log_file: path must be a string or nothing"))
    status = ccall(_sym(:nlolib_set_log_file), Cint, (Cstring, Cint), path === nothing ? C_NULL : path, Cint(Bool(append)))
    _check_status(status, "nlolib_set_log_file")
    return nothing
end

"""
    set_log_buffer(capacity_bytes = 262144)

Keep the library's log in an in-memory ring buffer of `capacity_bytes`, read
back with `read_log_buffer`.
"""
function set_log_buffer(capacity_bytes = 262144)
    status = ccall(_sym(:nlolib_set_log_buffer), Cint, (Csize_t,), Csize_t(capacity_bytes))
    _check_status(status, "nlolib_set_log_buffer")
    return nothing
end

"""
    clear_log_buffer()

Discard the contents of the in-memory log buffer.
"""
function clear_log_buffer()
    status = ccall(_sym(:nlolib_clear_log_buffer), Cint, ())
    _check_status(status, "nlolib_clear_log_buffer")
    return nothing
end

"""
    read_log_buffer(; consume = true, max_bytes = 262144)

Up to `max_bytes` of the in-memory log as a string; with `consume`, the text
read is removed from the buffer.
"""
function read_log_buffer(; consume = true, max_bytes = 262144)
    max_bytes >= 2 || throw(ArgumentError("read_log_buffer: max_bytes must be >= 2"))
    buffer = Vector{UInt8}(undef, Int(max_bytes))
    written = Ref{Csize_t}(0)
    status = GC.@preserve buffer written begin
        ccall(
            _sym(:nlolib_read_log_buffer),
            Cint,
            (Ptr{UInt8}, Csize_t, Ptr{Csize_t}, Cint),
            pointer(buffer),
            Csize_t(max_bytes),
            Base.unsafe_convert(Ptr{Csize_t}, written),
            Cint(Bool(consume)),
        )
    end
    _check_status(status, "nlolib_read_log_buffer")
    return String(buffer[1:Int(written[])])
end

"""
    set_log_level(level = "info")

Log messages at `level` and above: `"error"`, `"warn"`, `"info"` or `"debug"`.
"""
function set_log_level(level = "info")
    status = ccall(_sym(:nlolib_set_log_level), Cint, (Cint,), _enum_code(_LOG_LEVELS, level, "set_log_level: level"))
    _check_status(status, "nlolib_set_log_level")
    return nothing
end

"""
    set_progress_options(; enabled = true, milestone_percent = 5, emit_on_step_adjust = false)

Control the library's own progress output: whether it is printed, every how
many percent, and whether step-size changes are reported too.
"""
function set_progress_options(; enabled = true, milestone_percent = 5, emit_on_step_adjust = false)
    status = ccall(
        _sym(:nlolib_set_progress_options),
        Cint,
        (Cint, Cint, Cint),
        Cint(Bool(enabled)),
        Cint(milestone_percent),
        Cint(Bool(emit_on_step_adjust)),
    )
    _check_status(status, "nlolib_set_progress_options")
    return nothing
end

"""
    set_progress_stream(stream_mode = "stderr")

Print the library's progress output to `"stderr"`, `"stdout"` or `"both"`.
"""
function set_progress_stream(stream_mode = "stderr")
    code = _enum_code(_PROGRESS_STREAMS, stream_mode, "set_progress_stream: stream_mode")
    status = ccall(_sym(:nlolib_set_progress_stream), Cint, (Cint,), code)
    _check_status(status, "nlolib_set_progress_stream")
    return nothing
end

"""
    perf_profile_set_enabled(enabled = true)

Turn accumulation of the performance counters on or off.
"""
function perf_profile_set_enabled(enabled = true)
    status = ccall(_sym(:nlolib_perf_profile_set_enabled), Cint, (Cint,), Cint(Bool(enabled)))
    _check_status(status, "nlolib_perf_profile_set_enabled")
    return nothing
end

"""
    perf_profile_is_enabled()

Whether the performance counters are accumulating.
"""
function perf_profile_is_enabled()
    return ccall(_sym(:nlolib_perf_profile_is_enabled), Cint, ()) != 0
end

"""
    perf_profile_reset()

Set every performance counter back to zero.
"""
function perf_profile_reset()
    status = ccall(_sym(:nlolib_perf_profile_reset), Cint, ())
    _check_status(status, "nlolib_perf_profile_reset")
    return nothing
end

"""
    perf_profile_read()

The performance counters accumulated since the last `perf_profile_reset`, as a
`PerfProfileSnapshot`.
"""
function perf_profile_read()
    snapshot = Ref(PerfProfileSnapshot())
    status = ccall(_sym(:nlolib_perf_profile_read), Cint, (Ref{PerfProfileSnapshot},), snapshot)
    _check_status(status, "nlolib_perf_profile_read")
    return snapshot[]
end
