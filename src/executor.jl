const _PROGRESS_CALLBACK_REF = Ref{Any}(nothing)
const _PROGRESS_CALLBACK_ERROR = Ref{Any}(nothing)

function _progress_trampoline(info_ptr::Ptr{CProgressInfo}, ::Ptr{Cvoid})::Cint
    info_ptr == C_NULL && return Cint(1)
    callback = _PROGRESS_CALLBACK_REF[]
    callback === nothing && return Cint(1)
    info = unsafe_load(info_ptr)
    try
        result = callback(_progress_info(info))
        return Cint(result === nothing ? 1 : Bool(result))
    catch err
        _PROGRESS_CALLBACK_ERROR[] = err
        return Cint(0)
    end
end

# Taken at run time: a pointer stored in a const at precompile time is stale
# once the package image is loaded.
_progress_callback_ptr() = @cfunction(_progress_trampoline, Cint, (Ptr{CProgressInfo}, Ptr{Cvoid}))

function _execute(request::NormalizedPropagateRequest)
    sim_ref = Ref(request.sim_cfg)
    phys_ref = Ref(request.phys_cfg)
    exec_ref = request.exec_options === nothing ? nothing : Ref(request.exec_options)
    storage_ref = request.storage_options === nothing ? nothing : Ref(request.storage_options)

    input_field = request.input_seq
    input_storage = reinterpret(NLOComplex, input_field)
    input_ptr = Base.unsafe_convert(Ptr{NLOComplex}, input_storage)

    # ComplexF64 and NLOComplex share a layout, so the library writes straight
    # into the matrix the result returns.
    raw_output = request.return_records ? Matrix{ComplexF64}(undef, length(input_field), request.num_records) : nothing
    output_ptr = raw_output === nothing ? C_NULL : Ptr{NLOComplex}(pointer(raw_output))
    record_z = request.record_z
    step_events = request.step_history_capacity > 0 ? Vector{StepEvent}(undef, request.step_history_capacity) : nothing

    records_written_ref = Ref{Csize_t}(0)
    step_events_written_ref = Ref{Csize_t}(0)
    step_events_dropped_ref = Ref{Csize_t}(0)
    storage_result_ref = Ref(CStorageResult())

    options = PropagateOptions(
        Csize_t(request.num_records),
        request.output_mode,
        Cint(request.return_records),
        exec_ref === nothing ? C_NULL : Base.unsafe_convert(Ptr{CExecutionOptions}, exec_ref),
        storage_ref === nothing ? C_NULL : Base.unsafe_convert(Ptr{CStorageOptions}, storage_ref),
        record_z === nothing ? C_NULL : pointer(record_z),
        Csize_t(record_z === nothing ? 0 : length(record_z)),
        request.progress_callback === nothing ? C_NULL : _progress_callback_ptr(),
        C_NULL,
    )
    output = PropagateOutput(
        output_ptr,
        Csize_t(raw_output === nothing ? 0 : size(raw_output, 2)),
        Base.unsafe_convert(Ptr{Csize_t}, records_written_ref),
        storage_ref === nothing ? C_NULL : Base.unsafe_convert(Ptr{CStorageResult}, storage_result_ref),
        step_events === nothing ? C_NULL : pointer(step_events),
        Csize_t(step_events === nothing ? 0 : length(step_events)),
        Base.unsafe_convert(Ptr{Csize_t}, step_events_written_ref),
        Base.unsafe_convert(Ptr{Csize_t}, step_events_dropped_ref),
    )
    options_ref = Ref(options)
    output_ref = Ref(output)

    _PROGRESS_CALLBACK_REF[] = request.progress_callback
    _PROGRESS_CALLBACK_ERROR[] = nothing
    keepalive = (request.keepalive, input_field, raw_output, record_z, step_events)

    status = GC.@preserve keepalive sim_ref phys_ref exec_ref storage_ref options_ref output_ref records_written_ref step_events_written_ref step_events_dropped_ref storage_result_ref begin
        ccall(
            _sym(:nlolib_propagate),
            Cint,
            (Ptr{SimulationConfig}, Ptr{PhysicsConfig}, Csize_t, Ptr{NLOComplex}, Ref{PropagateOptions}, Ref{PropagateOutput}),
            Base.unsafe_convert(Ptr{SimulationConfig}, sim_ref),
            Base.unsafe_convert(Ptr{PhysicsConfig}, phys_ref),
            Csize_t(length(input_field)),
            input_ptr,
            options_ref,
            output_ref
        )
    end
    callback_error = _PROGRESS_CALLBACK_ERROR[]
    _PROGRESS_CALLBACK_REF[] = nothing
    _PROGRESS_CALLBACK_ERROR[] = nothing

    status in (NLOLIB_STATUS_OK, NLOLIB_STATUS_ABORTED) || _check_status(status, "nlolib_propagate")

    written = Int(records_written_ref[])
    records = raw_output === nothing ? zeros(ComplexF64, length(input_field), 0) : raw_output
    returned = min(written, size(records, 2))
    returned < size(records, 2) && (records = records[:, 1:returned])
    # With return_records = false there are no columns to take the final record
    # from; the field is a vector, so the empty case is an empty vector.
    final = returned > 0 ? view(records, :, returned) : ComplexF64[]
    meta = copy(request.meta)
    meta["records_written"] = written
    meta["status"] = _enum_name(_STATUSES, status)
    result = PropagationResult(
        records,
        request.z_axis[1:returned],
        final,
        step_events === nothing ? nothing :
            _step_history(step_events, step_events_written_ref[], step_events_dropped_ref[]),
        storage_ref === nothing ? nothing : _storage_result(storage_result_ref[]),
        meta,
    )
    callback_error !== nothing && throw(callback_error)
    status == NLOLIB_STATUS_ABORTED && throw(PropagationAbortedError(result))
    return result
end
