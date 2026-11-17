# Linear dispersive drift.
#
# Nonlinearity is off (gamma = 0) and an initial phase ramp exp(-i chirp t) on
# a Gaussian pulse makes it drift in time as it propagates. The centroid moves
# at the group-velocity slope beta2 * chirp, and the records are checked
# against the exact linear solution.

using NLOLibExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using FFTW
using NLOLib

# nlolib-example-parameters: begin
n = 1024
dt = 0.01
sigma = 0.2
chirp = 12.0
beta2 = 0.05
gamma = 0.0
z_final = 1.0
num_records = 180
# nlolib-example-parameters: end

centroid_curve(t, records) = vec(sum(abs2.(records) .* reshape(t, 1, :), dims = 2) ./ max.(sum(abs2.(records), dims = 2), eps()))

function fitted_slope(x, y)
    xm = sum(x) / length(x)
    ym = sum(y) / length(y)
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
end

function main(argv = ARGS)
    args = parse_example_args("linear_drift", "Linear dispersive drift with DB-backed run/replot.", argv)
    db = ExampleRunDB(args[:db_path])
    example_name = "linear_drift"
    case_key = "default"

    t = centered_time_grid(n, dt)
    field0 = ComplexF64.(exp.(-(t ./ sigma) .^ 2) .* exp.(-im * chirp .* t))
    omega = 2.0 * π .* FFTW.fftfreq(n, 1.0 / dt)

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        loaded = load_case(db; example_name = example_name, run_group = run_group, case_key = case_key)
        z_axis = loaded.z_axis
        records = loaded.records
    else
        run_group = begin_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        storage = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = case_key, chunk_records = 8)
        result = propagate(
            PulseSpec(samples = field0, delta_time = dt, pulse_period = n * dt),
            (A, w) -> im * (0.5 * beta2) * w^2,
            (A, I, V) -> im * A * (gamma * I + V);
            propagation_distance = z_final,
            num_records = num_records,
            preset = "accuracy",
            exec_options = ExecutionOptions(backend_type = "auto"),
            storage = StorageOptions(; storage...),
        )
        save_case_from_result!(db; example_name = example_name, run_group = run_group, case_key = case_key,
            result = result, meta = Dict("n" => n, "dt" => dt, "sigma" => sigma, "beta2" => beta2, "chirp" => chirp))
        z_axis = result.z_axis
        records = permutedims(result.records)
    end

    centroid_shift = centroid_curve(t, records)
    centroid_shift .-= centroid_shift[1]
    measured_slope = fitted_slope(z_axis, centroid_shift)
    predicted_slope = beta2 * chirp
    slope_rel_error = abs(measured_slope - predicted_slope) / max(abs(predicted_slope), 1e-12)
    reference = exact_linear_temporal_records(field0, z_axis, omega, 0.5 * beta2)
    error_curve = 100.0 .* relative_l2_intensity_error_curve(records, reference)

    output_dir = args[:output_dir]
    band = abs.(t) .<= 1.5
    ld = (sigma / 1.665236)^2 / abs(beta2)
    plot_intensity_map(t[band], z_axis ./ ld, abs2.(records[:, band]),
        joinpath(output_dir, "linear_drift_intensity_propagation_map.png");
        xlabel = "Time t", ylabel = "Propagation distance z / L_D")
    plot_curves(t, ["Initial" => abs2.(records[1, :]), "Final" => abs2.(records[end, :])],
        joinpath(output_dir, "linear_drift_final_intensity_comparison.png"); xlabel = "Time t", ylabel = "Intensity")
    plot_curves(z_axis, ["Relative L2 error" => error_curve],
        joinpath(output_dir, "linear_drift_total_error_over_propagation.png");
        xlabel = "Propagation distance z", ylabel = "Relative L2 intensity error (%)")

    println("linear drift example completed (run_group=$(run_group)).")
    println("slope (signed): measured=$(measured_slope), predicted=$(predicted_slope)")
    report_metric("centroid slope relative error", slope_rel_error)
    return slope_rel_error
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
