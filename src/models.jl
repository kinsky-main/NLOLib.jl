"""
    PulseSpec(; samples, delta_time, ...)

The input field and its grids. `samples` is the flat field; a tensor run sets
`tensor_nt`, `tensor_nx` and `tensor_ny`, with `t` fastest, then `y`, then
`x`. Unset grids are derived: `pulse_period` is `delta_time` times the number
of time samples and `frequency_grid` the matching FFT-ordered angular
frequencies.
"""
Base.@kwdef mutable struct PulseSpec
    samples::Vector{ComplexF64}
    delta_time::Float64
    pulse_period::Union{Nothing, Float64} = nothing
    frequency_grid::Union{Nothing, Vector{ComplexF64}} = nothing
    tensor_nt::Union{Nothing, Int} = nothing
    tensor_nx::Union{Nothing, Int} = nothing
    tensor_ny::Union{Nothing, Int} = nothing
    delta_x::Float64 = 1.0
    delta_y::Float64 = 1.0
    spatial_frequency_grid::Union{Nothing, Vector{ComplexF64}} = nothing
    potential_grid::Union{Nothing, Vector{ComplexF64}} = nothing
    wt_axis::Union{Nothing, Vector{ComplexF64}} = nothing
    kx_axis::Union{Nothing, Vector{ComplexF64}} = nothing
    ky_axis::Union{Nothing, Vector{ComplexF64}} = nothing
end

"""
    ExecutionOptions(; backend_type = "auto", fft_backend = "auto", ...)

Where and how a run executes. `backend_type` is `"cpu"`, `"vulkan"` or
`"auto"`; `fft_backend` is `"auto"`, `"fftw"` or `"vkfft"`. The other fields
are the C `execution_options` resource controls of the same names.
"""
Base.@kwdef struct ExecutionOptions
    backend_type::String = "auto"
    fft_backend::String = "auto"
    device_heap_fraction::Float64 = 0.70
    record_ring_target::Int = 0
    forced_device_budget_bytes::Int = 0
end

"""
    StorageOptions(; sqlite_path, run_id = nothing, ...)

Write the records to the SQLite database at `sqlite_path`. `run_id` names the
run in the database (at most 63 characters). `cap_policy` says what happens
when the database reaches `sqlite_max_bytes`: `"stop_writes"` stops writing
and lets the run continue, `"fail"` makes the run fail. The other fields are
the C `storage_options` fields of the same names.
"""
Base.@kwdef struct StorageOptions
    sqlite_path::String
    run_id::Union{Nothing, String} = nothing
    sqlite_max_bytes::Int = 0
    chunk_records::Int = 0
    cap_policy::String = "stop_writes"
    log_final_output_field_to_db::Bool = false
end

"""
    StepHistory

The accepted steps of a run with `capture_step_history = true`: one entry per
step in each vector. `dropped` counts steps that did not fit in `capacity`.
"""
struct StepHistory
    step_index::Vector{Int}
    z::Vector{Float64}
    step_size::Vector{Float64}
    next_step_size::Vector{Float64}
    error::Vector{Float64}
    dropped::Int
    capacity::Int
end

"""
    StorageResult

What a run with `storage` wrote to its database.
"""
struct StorageResult
    run_id::String
    records_captured::Int
    records_spilled::Int
    chunks_written::Int
    db_size_bytes::Int
    truncated::Bool
end

"""
    ProgressInfo

One progress event, as `progress_callback` receives it. `event_type` is
`"accepted"`, `"rejected"` or `"finish"`.
"""
struct ProgressInfo
    event_type::String
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

"""
    RuntimeLimits

What `query_runtime_limits` returns: the fields of the C `runtime_limits`, in
its order. `storage_available` says whether the build can write SQLite.
"""
struct RuntimeLimits
    max_num_time_samples_runtime::Int
    max_num_recorded_samples_in_memory::Int
    max_num_recorded_samples_with_storage::Int
    estimated_required_working_set_bytes::Int
    estimated_device_budget_bytes::Int
    storage_available::Bool
end

"""
    PropagationResult

The result of `propagate`. `records` is samples x records, `z_axis` the
position of each record and `final` the last returned record,
`records[:, end]` (empty when `return_records = false`). `final` is the field at
`propagation_distance` only when the last record is there: a `record_z` that
ends earlier gives the field at its last position. `step_history` is set only with
`capture_step_history = true` and `storage_result` only with `storage`.
`meta` holds `"preset"`, `"records_requested"`, `"records_written"`,
`"backend_requested"`, `"coupled"` and `"status"`.
"""
struct PropagationResult
    records::Matrix{ComplexF64}
    z_axis::Vector{Float64}
    final::AbstractVector{ComplexF64}
    step_history::Union{Nothing, StepHistory}
    storage_result::Union{Nothing, StorageResult}
    meta::Dict{String, Any}
end

"""
    PropagationAbortedError(result)

Thrown when `progress_callback` returned `false`. `result` holds the records
written before the abort.
"""
struct PropagationAbortedError <: Exception
    result::PropagationResult
end

Base.showerror(io::IO, err::PropagationAbortedError) =
    print(io, "PropagationAbortedError: propagate aborted by progress callback")
