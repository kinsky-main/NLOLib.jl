const RUNTIME_OPERATOR_CONSTANTS_MAX = 16
const STORAGE_RUN_ID_MAX = 64
# Unbounded sentinel for the time-sample count; prefer query_runtime_limits.
const NT_MAX = typemax(UInt64)

const TENSOR_LAYOUT_XYT_T_FAST = Cint(0)

const NONLINEAR_MODEL_EXPR = Cint(0)
const NONLINEAR_MODEL_KERR_RAMAN = Cint(1)

const VECTOR_BACKEND_CPU = Cint(0)
const VECTOR_BACKEND_VULKAN = Cint(1)
const VECTOR_BACKEND_AUTO = Cint(2)

const FFT_BACKEND_AUTO = Cint(0)
const FFT_BACKEND_FFTW = Cint(1)
const FFT_BACKEND_VKFFT = Cint(2)

const STORAGE_DB_CAP_POLICY_STOP_WRITES = Cint(0)
const STORAGE_DB_CAP_POLICY_FAIL = Cint(1)

const PROPAGATE_OUTPUT_DENSE = Cint(0)
const PROPAGATE_OUTPUT_FINAL_ONLY = Cint(1)

const NLOLIB_STATUS_OK = Cint(0)
const NLOLIB_STATUS_INVALID_ARGUMENT = Cint(1)
const NLOLIB_STATUS_ALLOCATION_FAILED = Cint(2)
const NLOLIB_STATUS_NOT_IMPLEMENTED = Cint(3)
const NLOLIB_STATUS_ABORTED = Cint(4)

const NLOLIB_LOG_LEVEL_ERROR = Cint(0)
const NLOLIB_LOG_LEVEL_WARN = Cint(1)
const NLOLIB_LOG_LEVEL_INFO = Cint(2)
const NLOLIB_LOG_LEVEL_DEBUG = Cint(3)

const NLOLIB_PROGRESS_STREAM_STDERR = Cint(0)
const NLOLIB_PROGRESS_STREAM_STDOUT = Cint(1)
const NLOLIB_PROGRESS_STREAM_BOTH = Cint(2)

const PROGRESS_EVENT_ACCEPTED = Cint(0)
const PROGRESS_EVENT_REJECTED = Cint(1)
const PROGRESS_EVENT_FINISH = Cint(2)

struct NLOComplex
    re::Cdouble
    im::Cdouble
end

NLOComplex(z::Complex) = NLOComplex(real(z), imag(z))
Base.convert(::Type{ComplexF64}, z::NLOComplex) = ComplexF64(z.re, z.im)

struct PropagationParams
    starting_step_size::Cdouble
    max_step_size::Cdouble
    min_step_size::Cdouble
    error_tolerance::Cdouble
    propagation_distance::Cdouble
end

struct TimeGrid
    nt::Csize_t
    pulse_period::Cdouble
    delta_time::Cdouble
    wt_axis::Ptr{NLOComplex}
end

struct FrequencyGrid
    frequency_grid::Ptr{NLOComplex}
end

struct SpatialGrid
    nx::Csize_t
    ny::Csize_t
    delta_x::Cdouble
    delta_y::Cdouble
    spatial_frequency_grid::Ptr{NLOComplex}
    kx_axis::Ptr{NLOComplex}
    ky_axis::Ptr{NLOComplex}
    potential_grid::Ptr{NLOComplex}
end

struct Tensor3DDesc
    nt::Csize_t
    nx::Csize_t
    ny::Csize_t
    layout::Cint
end

struct RuntimeOperatorParams
    linear_factor_expr::Cstring
    linear_expr::Cstring
    potential_expr::Cstring
    dispersion_factor_expr::Cstring
    dispersion_expr::Cstring
    nonlinear_expr::Cstring
    nonlinear_model::Cint
    nonlinear_gamma::Cdouble
    raman_fraction::Cdouble
    raman_tau1::Cdouble
    raman_tau2::Cdouble
    shock_omega0::Cdouble
    raman_response_time::Ptr{NLOComplex}
    raman_response_len::Csize_t
    num_constants::Csize_t
    constants::NTuple{RUNTIME_OPERATOR_CONSTANTS_MAX, Cdouble}
end

const PhysicsConfig = RuntimeOperatorParams

struct SimulationConfig
    propagation::PropagationParams
    tensor::Tensor3DDesc
    time::TimeGrid
    frequency::FrequencyGrid
    spatial::SpatialGrid
end

struct VulkanBackendConfig
    physical_device::Ptr{Cvoid}
    device::Ptr{Cvoid}
    queue::Ptr{Cvoid}
    queue_family_index::UInt32
    command_pool::Ptr{Cvoid}
    descriptor_set_budget_bytes::Csize_t
    descriptor_set_count_override::UInt32
end

struct CExecutionOptions
    backend_type::Cint
    fft_backend::Cint
    device_heap_fraction::Cdouble
    record_ring_target::Csize_t
    forced_device_budget_bytes::Csize_t
    vulkan::VulkanBackendConfig
end

"""
    RuntimeLimits

What `query_runtime_limits` returns: the fields of the C `runtime_limits`, in
its order. `storage_available` is nonzero when the build can write SQLite.
"""
struct RuntimeLimits
    max_num_time_samples_runtime::Csize_t
    max_num_recorded_samples_in_memory::Csize_t
    max_num_recorded_samples_with_storage::Csize_t
    estimated_required_working_set_bytes::Csize_t
    estimated_device_budget_bytes::Csize_t
    storage_available::Cint
end

struct CStorageOptions
    sqlite_path::Cstring
    run_id::Cstring
    sqlite_max_bytes::Csize_t
    chunk_records::Csize_t
    cap_policy::Cint
    log_final_output_field_to_db::Cint
end

struct CStorageResult
    run_id::NTuple{STORAGE_RUN_ID_MAX, Cchar}
    records_captured::Csize_t
    records_spilled::Csize_t
    chunks_written::Csize_t
    db_size_bytes::Csize_t
    truncated::Cint
end

struct StepEvent
    step_index::Csize_t
    z_current::Cdouble
    step_size::Cdouble
    next_step_size::Cdouble
    error::Cdouble
end

struct CProgressInfo
    event_type::Cint
    step_index::Csize_t
    reject_attempt::Csize_t
    z::Cdouble
    z_end::Cdouble
    percent::Cdouble
    step_size::Cdouble
    next_step_size::Cdouble
    error::Cdouble
    elapsed_seconds::Cdouble
    eta_seconds::Cdouble
end

struct PropagateOptions
    num_recorded_samples::Csize_t
    output_mode::Cint
    return_records::Cint
    exec_options::Ptr{CExecutionOptions}
    storage_options::Ptr{CStorageOptions}
    explicit_record_z::Ptr{Cdouble}
    explicit_record_z_count::Csize_t
    progress_callback::Ptr{Cvoid}
    progress_user_data::Ptr{Cvoid}
end

struct PropagateOutput
    output_records::Ptr{NLOComplex}
    output_record_capacity::Csize_t
    records_written::Ptr{Csize_t}
    storage_result::Ptr{CStorageResult}
    output_step_events::Ptr{StepEvent}
    output_step_event_capacity::Csize_t
    step_events_written::Ptr{Csize_t}
    step_events_dropped::Ptr{Csize_t}
end

"""
    PerfProfileSnapshot

Accumulated performance counters, as `perf_profile_read` returns them. The
fields are those of the C `nlo_perf_profile_snapshot`, in its order.
"""
struct PerfProfileSnapshot
    dispersion_ms::Cdouble
    nonlinear_ms::Cdouble
    dispersion_calls::UInt64
    nonlinear_calls::UInt64
    gpu_dispatch_count::UInt64
    gpu_copy_count::UInt64
    gpu_device_copy_count::UInt64
    gpu_device_copy_bytes::UInt64
    gpu_host_transfer_copy_count::UInt64
    gpu_host_transfer_copy_bytes::UInt64
    gpu_memory_pass_count::UInt64
    gpu_memory_pass_bytes::UInt64
    gpu_upload_count::UInt64
    gpu_download_count::UInt64
    gpu_upload_bytes::UInt64
    gpu_download_bytes::UInt64
    operator_jit_programs::UInt64
    operator_jit_fallbacks::UInt64
end

"""
    NLOLibError(status, context)

A failure reported by the nlolib library. `status` is the C status as a
lowercase string (`"invalid_argument"`, `"allocation_failed"`,
`"not_implemented"`, ...), or `nothing` when the library could not be loaded
at all. `context` names the call or the missing feature.
"""
struct NLOLibError <: Exception
    status::Union{Nothing, String}
    context::String
end

Base.showerror(io::IO, err::NLOLibError) = print(io, "NLOLibError: ", err.context,
    err.status === nothing ? "" : " (status \"$(err.status)\")")

# -- enum-style values -------------------------------------------------------
#
# The public value of every enum-style option or result is a lowercase string;
# these tables map it to the C integer and back.

const _VECTOR_BACKENDS = Dict("cpu" => VECTOR_BACKEND_CPU, "vulkan" => VECTOR_BACKEND_VULKAN,
                              "auto" => VECTOR_BACKEND_AUTO)
const _FFT_BACKENDS = Dict("auto" => FFT_BACKEND_AUTO, "fftw" => FFT_BACKEND_FFTW,
                           "vkfft" => FFT_BACKEND_VKFFT)
const _NONLINEAR_MODELS = Dict("expr" => NONLINEAR_MODEL_EXPR, "kerr_raman" => NONLINEAR_MODEL_KERR_RAMAN)
const _CAP_POLICIES = Dict("stop_writes" => STORAGE_DB_CAP_POLICY_STOP_WRITES,
                           "fail" => STORAGE_DB_CAP_POLICY_FAIL)
const _LOG_LEVELS = Dict("error" => NLOLIB_LOG_LEVEL_ERROR, "warn" => NLOLIB_LOG_LEVEL_WARN,
                         "info" => NLOLIB_LOG_LEVEL_INFO, "debug" => NLOLIB_LOG_LEVEL_DEBUG)
const _PROGRESS_STREAMS = Dict("stderr" => NLOLIB_PROGRESS_STREAM_STDERR,
                               "stdout" => NLOLIB_PROGRESS_STREAM_STDOUT,
                               "both" => NLOLIB_PROGRESS_STREAM_BOTH)
const _PROGRESS_EVENTS = Dict("accepted" => PROGRESS_EVENT_ACCEPTED, "rejected" => PROGRESS_EVENT_REJECTED,
                              "finish" => PROGRESS_EVENT_FINISH)
const _STATUSES = Dict("ok" => NLOLIB_STATUS_OK, "invalid_argument" => NLOLIB_STATUS_INVALID_ARGUMENT,
                       "allocation_failed" => NLOLIB_STATUS_ALLOCATION_FAILED,
                       "not_implemented" => NLOLIB_STATUS_NOT_IMPLEMENTED,
                       "aborted" => NLOLIB_STATUS_ABORTED)

"""
    _enum_code(table, value, what)

The C integer for the string `value`; an `ArgumentError` naming `what` and the
accepted strings for anything else.
"""
function _enum_code(table::Dict{String, Cint}, value, what::AbstractString)
    code = value isa AbstractString ? get(table, String(value), nothing) : nothing
    code === nothing && throw(ArgumentError(
        "$(what) must be one of " * join(("\"$(k)\"" for k in sort(collect(keys(table)))), ", ") *
        "; got $(repr(value))"
    ))
    return code
end

"""
    _enum_name(table, code)

The string for the C integer `code`, or `"unknown"` for a value the table does
not hold.
"""
function _enum_name(table::Dict{String, Cint}, code::Integer)
    for (name, value) in table
        value == code && return name
    end
    return "unknown"
end

_zero_chars(::Val{N}) where {N} = ntuple(_ -> Cchar(0), N)
_zero_constants() = ntuple(_ -> 0.0, RUNTIME_OPERATOR_CONSTANTS_MAX)

PropagationParams() = PropagationParams(0.0, 0.0, 0.0, 0.0, 0.0)
TimeGrid() = TimeGrid(0, 0.0, 0.0, C_NULL)
FrequencyGrid() = FrequencyGrid(C_NULL)
SpatialGrid() = SpatialGrid(0, 0, 1.0, 1.0, C_NULL, C_NULL, C_NULL, C_NULL)
Tensor3DDesc() = Tensor3DDesc(0, 0, 0, TENSOR_LAYOUT_XYT_T_FAST)
RuntimeOperatorParams() = RuntimeOperatorParams(
    C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
    NONLINEAR_MODEL_EXPR, 0.0, 0.0, 0.0, 0.0, 0.0, C_NULL, 0, 0, _zero_constants()
)
SimulationConfig() = SimulationConfig(PropagationParams(), Tensor3DDesc(), TimeGrid(), FrequencyGrid(), SpatialGrid())
VulkanBackendConfig() = VulkanBackendConfig(C_NULL, C_NULL, C_NULL, 0, C_NULL, 0, 0)
RuntimeLimits() = RuntimeLimits(0, 0, 0, 0, 0, 0)
CStorageResult() = CStorageResult(_zero_chars(Val(STORAGE_RUN_ID_MAX)), 0, 0, 0, 0, 0)
PerfProfileSnapshot() = PerfProfileSnapshot(0.0, 0.0, ntuple(_ -> UInt64(0), 16)...)
