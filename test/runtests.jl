using Test
using NonlinearOptics

const CPU = ExecutionOptions(backend_type = "cpu", fft_backend = "fftw")

function gaussian_pulse(nt = 64, dt = 0.02)
    centered = (collect(0:nt - 1) .- nt / 2) .* dt
    return PulseSpec(samples = ComplexF64.(exp.(-(centered ./ 0.1) .^ 2)), delta_time = dt)
end

zero_linear(A, w) = 0.0
zero_nonlinear(A, I) = 0.0

function runtime_functions_case()
    load()
    @test !isempty(loaded_library_path())
    @test storage_is_available() isa Bool

    limits = query_runtime_limits()
    @test limits isa RuntimeLimits
    @test limits.storage_available isa Bool
    @test limits.storage_available == storage_is_available()
    @test limits.max_num_time_samples_runtime > 0
    pulse_limits = query_runtime_limits(pulse = gaussian_pulse(), exec_options = CPU)
    @test pulse_limits.max_num_recorded_samples_in_memory > 0

    # A tensor pulse is sized as nt*nx*ny samples: doubling nt doubles the
    # working set, where counting the samples twice (N x N) would quadruple it.
    tensor_limits(nt) = query_runtime_limits(
        pulse = PulseSpec(samples = ones(ComplexF64, nt * 8 * 8), delta_time = 0.1,
                          tensor_nt = nt, tensor_nx = 8, tensor_ny = 8),
        exec_options = CPU)
    small, large = tensor_limits(32), tensor_limits(64)
    @test small.estimated_required_working_set_bytes > 0
    @test 1.5 < large.estimated_required_working_set_bytes / small.estimated_required_working_set_bytes < 2.5

    set_log_level("warn")
    @test_throws ArgumentError set_log_level("loud")
    set_log_level()
    set_log_buffer(4096)
    clear_log_buffer()
    @test read_log_buffer(consume = true, max_bytes = 4096) isa String
    set_progress_options(enabled = false)
    set_progress_stream("stdout")
    @test_throws ArgumentError set_progress_stream(1)
    set_progress_stream()
    set_log_file(nothing; append = false)

    perf_profile_set_enabled(true)
    @test perf_profile_is_enabled()
    perf_profile_reset()
    snapshot = perf_profile_read()
    @test snapshot isa PerfProfileSnapshot
    @test snapshot.dispersion_calls == 0
    perf_profile_set_enabled(false)
    @test !perf_profile_is_enabled()

    # An explicit path is the only one tried, and a failure keeps the library
    # that was loaded.
    loaded = loaded_library_path()
    @test_throws NLOLibError load(joinpath(mktempdir(), "missing-libnlolib.so"))
    @test loaded_library_path() == loaded
end

function records_case()
    pulse = gaussian_pulse()
    nt = length(pulse.samples)

    # num_records == 1: the final field only.
    final_only = propagate(pulse, zero_linear, zero_nonlinear;
        propagation_distance = 0.1, num_records = 1, exec_options = CPU)
    @test size(final_only.records) == (nt, 1)
    @test final_only.z_axis == [0.1]
    @test final_only.final ≈ pulse.samples atol = 1e-12 rtol = 1e-12
    @test final_only.step_history === nothing
    @test final_only.storage_result === nothing
    @test sort(collect(keys(final_only.meta))) ==
          sort(["preset", "records_requested", "records_written", "backend_requested", "coupled", "status"])
    @test final_only.meta["status"] == "ok"
    @test final_only.meta["backend_requested"] == "cpu"
    @test final_only.meta["preset"] == "balanced"
    @test final_only.meta["coupled"] === false

    # Otherwise num_records records evenly spaced from 0.
    dense = propagate(pulse, zero_linear, zero_nonlinear;
        propagation_distance = 0.1, num_records = 5, preset = "fast", exec_options = CPU)
    @test size(dense.records) == (nt, 5)
    @test dense.z_axis ≈ collect(range(0.0, 0.1, length = 5))
    @test dense.meta["records_written"] == 5
    @test dense.meta["preset"] == "fast"

    # record_z: exactly those positions, including a single one.
    explicit = propagate(pulse, zero_linear, zero_nonlinear;
        propagation_distance = 0.1, record_z = [0.0, 0.025, 0.1], exec_options = CPU)
    @test explicit.z_axis == [0.0, 0.025, 0.1]
    @test size(explicit.records, 2) == 3
    # A single explicit position returns the field there, the same as the
    # matching record of a multi-record run; final is that last returned record.
    drift = (A, w) -> im * 0.2 * w
    single = propagate(pulse, drift, zero_nonlinear;
        propagation_distance = 0.1, record_z = [0.05], num_records = 1, exec_options = CPU)
    several = propagate(pulse, drift, zero_nonlinear;
        propagation_distance = 0.1, record_z = [0.0, 0.05, 0.1], exec_options = CPU)
    final_drift = propagate(pulse, drift, zero_nonlinear;
        propagation_distance = 0.1, num_records = 1, exec_options = CPU)
    @test single.z_axis == [0.05]
    @test size(single.records, 2) == 1
    @test single.records[:, 1] ≈ several.records[:, 2] rtol = 1e-10
    @test single.final == single.records[:, 1]
    @test !isapprox(single.final, final_drift.final, rtol = 1e-6)
    # final is the last returned record, at the last record_z, not at the distance.
    early = propagate(pulse, drift, zero_nonlinear;
        propagation_distance = 0.1, record_z = [0.0, 0.05], exec_options = CPU)
    @test early.z_axis[end] == 0.05
    @test early.final ≈ several.records[:, 2] rtol = 1e-10
    @test_throws ArgumentError propagate(pulse, zero_linear, zero_nonlinear;
        propagation_distance = 0.1, record_z = [0.0, 0.1], num_records = 3)

    record = dense.records[:, end]
    @test size(tensor_record_view(record, nt, 1, 1)) == (nt, 1, 1)
    @test_throws ArgumentError tensor_record_view(record, nt, 2, 1)
end

function physics_case()
    # A fundamental soliton keeps its peak power.
    nt = 1024
    dt = 0.02
    t = (collect(0:nt - 1) .- nt / 2) .* dt
    pulse = PulseSpec(samples = ComplexF64.(1.0 ./ cosh.(t)), delta_time = dt)
    beta2 = -1.0
    gamma = 1.0
    result = propagate(pulse, (A, w) -> im * (beta2 / 2) * w^2, (A, I) -> im * gamma * A * I;
        propagation_distance = pi / 8, num_records = 1, exec_options = CPU)
    @test maximum(abs2, result.final) ≈ maximum(abs2, pulse.samples) rtol = 1e-3

    # constant_bindings spends a slot on a named value; the result is unchanged.
    bound = propagate(pulse, (A, w) -> im * (beta2 / 2) * w^2, (A, I) -> im * gamma * A * I;
        propagation_distance = pi / 8, num_records = 1, exec_options = CPU,
        constant_bindings = Dict("gamma" => gamma))
    @test bound.final ≈ result.final rtol = 1e-10

    # A linear_step_operator stands in for the default exp(h*D).
    stepped = propagate(pulse, (A, w) -> im * (beta2 / 2) * w^2, (A, I) -> im * gamma * A * I;
        propagation_distance = pi / 8, num_records = 1, exec_options = CPU,
        linear_step_operator = (A, D, h) -> exp(h * D))
    @test stepped.final ≈ result.final rtol = 1e-8
end

function tensor_case()
    nt, nx, ny = 16, 4, 4
    field = zeros(ComplexF64, nt * nx * ny)
    for x in 1:nx, y in 1:ny, k in 1:nt
        field[k + nt * ((y - 1) + ny * (x - 1))] = exp(-((k - nt / 2) / 3)^2 - (x - 2.5)^2 - (y - 2.5)^2)
    end
    pulse = PulseSpec(samples = field, delta_time = 0.1, tensor_nt = nt, tensor_nx = nx, tensor_ny = ny,
                      delta_x = 0.5, delta_y = 0.5)
    # (A, w) works for a tensor run too: w names the wt axis.
    result = propagate(pulse, (A, w, kx, ky) -> im * (0.01 * w^2 - 0.01 * (kx^2 + ky^2)), (A, I) -> 0.0;
        propagation_distance = 0.05, num_records = 2, exec_options = CPU,
        potential_operator = (x, y) -> 0.0)
    @test result.meta["status"] == "ok"
    @test result.meta["coupled"] == true
    @test sum(abs2, result.final) ≈ sum(abs2, field) rtol = 1e-6
    @test size(tensor_record_view(result.final, nt, nx, ny)) == (nt, ny, nx)

    # A static x-y potential grid is broadcast along t: it must give the same
    # run as the full t-fastest volume with each x-y value repeated nt times.
    potential_xy = ComplexF64[0.3 * (x - 2.5)^2 + 0.1 * (y - 1.5) for y in 1:ny, x in 1:nx]
    potential_full = ComplexF64[potential_xy[y, x] for k in 1:nt, y in 1:ny, x in 1:nx]
    run_with(grid) = propagate(
        PulseSpec(samples = field, delta_time = 0.1, tensor_nt = nt, tensor_nx = nx, tensor_ny = ny,
                  delta_x = 0.5, delta_y = 0.5, potential_grid = vec(grid)),
        (A, w, kx, ky) -> im * 0.01 * w^2, (A, I, V) -> im * A * V;
        propagation_distance = 0.5, num_records = 1, exec_options = CPU)
    broadcast_run = run_with(potential_xy)
    @test broadcast_run.final ≈ run_with(potential_full).final rtol = 1e-12
    # The potential only rotates the phase at each (x, y): exp(i V z).
    expected = field .* exp.(im .* vec(potential_full) .* 0.5)
    phase_only = propagate(
        PulseSpec(samples = field, delta_time = 0.1, tensor_nt = nt, tensor_nx = nx, tensor_ny = ny,
                  delta_x = 0.5, delta_y = 0.5, potential_grid = vec(potential_xy)),
        (A, w, kx, ky) -> 0.0, (A, I, V) -> im * A * V;
        propagation_distance = 0.5, num_records = 1, exec_options = CPU)
    @test phase_only.final ≈ expected rtol = 1e-8
end

function step_history_and_progress_case()
    pulse = gaussian_pulse()
    result = propagate(pulse, (A, w) -> im * 0.01 * w^2, zero_nonlinear;
        propagation_distance = 0.1, num_records = 2, exec_options = CPU,
        capture_step_history = true, step_history_capacity = 1000)
    history = result.step_history
    @test history isa StepHistory
    @test !isempty(history.z)
    @test length(history.step_index) == length(history.z) == length(history.error)
    @test history.capacity == 1000
    @test history.dropped == 0

    events = String[]
    seen = propagate(pulse, (A, w) -> im * 0.01 * w^2, zero_nonlinear;
        propagation_distance = 0.1, num_records = 2, exec_options = CPU,
        progress_callback = info -> (push!(events, info.event_type); true))
    @test seen.meta["status"] == "ok"
    @test !isempty(events)
    @test all(in(("accepted", "rejected", "finish")), events)

    err = try
        propagate(pulse, (A, w) -> im * 0.01 * w^2, zero_nonlinear;
            propagation_distance = 0.1, num_records = 2, exec_options = CPU,
            progress_callback = info -> false)
        nothing
    catch exc
        exc
    end
    @test err isa PropagationAbortedError
    @test err.result.meta["status"] == "aborted"
end

function storage_case()
    # return_records = false sends the records only to storage, which the core
    # requires in that case, and returns an empty result.
    pulse = gaussian_pulse()
    nt = length(pulse.samples)
    @test_throws ArgumentError propagate(pulse, zero_linear, zero_nonlinear;
        propagation_distance = 0.1, return_records = false)
    if !storage_is_available()
        @test_throws NLOLibError propagate(pulse, zero_linear, zero_nonlinear;
            propagation_distance = 0.1, storage = StorageOptions(sqlite_path = joinpath(mktempdir(), "run.sqlite")))
        return
    end
    for num_records in (8, 1)
        result = propagate(pulse, zero_linear, zero_nonlinear;
            propagation_distance = 0.1,
            num_records = num_records,
            return_records = false,
            storage = StorageOptions(sqlite_path = joinpath(mktempdir(), "run.sqlite"), run_id = "julia-test"),
            exec_options = CPU,
        )
        @test result.meta["status"] == "ok"
        @test size(result.records) == (nt, 0)
        @test isempty(result.z_axis)
        @test result.final isa AbstractVector{ComplexF64}
        @test isempty(result.final)
        @test result.storage_result isa StorageResult
        @test result.storage_result.run_id == "julia-test"
        @test result.storage_result.truncated isa Bool
        @test result.storage_result.records_captured > 0
    end
    @test_throws ArgumentError propagate(pulse, zero_linear, zero_nonlinear;
        propagation_distance = 0.1, storage = StorageOptions(sqlite_path = "run.sqlite", run_id = repeat("x", 64)))
    @test_throws ArgumentError propagate(pulse, zero_linear, zero_nonlinear;
        propagation_distance = 0.1, storage = StorageOptions(sqlite_path = "run.sqlite", cap_policy = "drop"))
end

function errors_case()
    pulse = gaussian_pulse()
    run(; kwargs...) = propagate(pulse, zero_linear, zero_nonlinear; propagation_distance = 0.1, kwargs...)

    # Both operators are required, and every option is a named keyword.
    @test_throws MethodError propagate(pulse; propagation_distance = 0.1)
    @test_throws MethodError propagate(pulse, zero_linear; propagation_distance = 0.1)
    @test_throws UndefKeywordError propagate(pulse, zero_linear, zero_nonlinear)
    @test_throws MethodError run(t_eval = [0.0, 0.1])
    @test_throws MethodError run(output = "final")
    @test_throws MethodError run(records = 4)
    @test_throws MethodError run(sqlite_path = "run.sqlite")
    @test_throws MethodError run(num_recorded_samples = 4)

    # Bad option values.
    @test_throws ArgumentError run(preset = "slow")
    @test_throws ArgumentError run(num_records = 0)
    @test_throws ArgumentError run(nonlinear_model = 1)
    @test_throws ArgumentError run(nonlinear_model = "kerr")
    @test_throws ArgumentError run(exec_options = ExecutionOptions(backend_type = "gpu"))
    @test_throws ArgumentError run(exec_options = ExecutionOptions(fft_backend = "FFTW"))
    @test_throws ArgumentError run(exec_options = (backend_type = "cpu",))
    @test_throws ArgumentError run(exec_options = Dict("backend_type" => "cpu"))
    @test_throws ArgumentError run(storage = (sqlite_path = "run.sqlite",))
    @test_throws ArgumentError run(storage = Dict("sqlite_path" => "run.sqlite"))
    @test_throws ArgumentError run(constant_bindings = Dict("beta2" => Inf))
    @test_throws ArgumentError run(propagation_distance = -1.0)
    @test_throws ArgumentError run(potential_operator = (x, y) -> 0.0)

    # Bad pulse.
    @test_throws ArgumentError propagate((samples = pulse.samples, delta_time = 0.02), zero_linear, zero_nonlinear;
        propagation_distance = 0.1)
    @test_throws ArgumentError propagate(PulseSpec(samples = pulse.samples, delta_time = 0.02, tensor_nt = 4),
        zero_linear, zero_nonlinear; propagation_distance = 0.1)
    @test_throws ArgumentError propagate(PulseSpec(samples = pulse.samples, delta_time = 0.02,
                                                   frequency_grid = zeros(ComplexF64, 3)),
        zero_linear, zero_nonlinear; propagation_distance = 0.1)

    # A C status failure is an NLOLibError carrying the status string. The
    # binding checks every input it can first, so call the status check the
    # C calls go through directly.
    for (code, name) in ((1, "invalid_argument"), (2, "allocation_failed"), (3, "not_implemented"))
        err = try
            NonlinearOptics._check_status(code, "nlolib_propagate")
            nothing
        catch exc
            exc
        end
        @test err isa NLOLibError
        @test err.status == name
        @test err.context == "nlolib_propagate"
    end
    @test NonlinearOptics._check_status(0, "nlolib_propagate") === nothing
end

@testset "NonlinearOptics Julia wrapper" begin
    @test sizeof(NonlinearOptics.NLOComplex) == 16
    runtime_functions_case()
    records_case()
    physics_case()
    tensor_case()
    step_history_and_progress_case()
    storage_case()
    errors_case()
end

include("test_translate_callable.jl")
