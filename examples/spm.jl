# Self-phase modulation (SPM) against its closed form.
#
# Pure Kerr nonlinearity with no dispersion or loss: each sample keeps its
# intensity and gains the phase gamma |A0|^2 z, so every record is checked
# against A0 exp(i gamma |A0|^2 z).

using NLOLibExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using FFTW
using NLOLib

# nlolib-example-parameters: begin
n = 1024
dt = 0.01
pulse_width = 0.2
beta2 = 0.0
gamma = 80.0
z_final = 0.5
num_records = 100
# nlolib-example-parameters: end

function main(argv = ARGS)
    args = parse_example_args("spm", "Self-phase modulation validation with DB-backed run/replot.", argv)
    set_log_level("error")
    set_progress_options(enabled = false)
    db = ExampleRunDB(args[:db_path])
    example_name = "spm"
    case_key = "default"

    t = centered_time_grid(n, dt)
    a0 = ComplexF64.(exp.(-(t ./ pulse_width) .^ 2))

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        loaded = load_case(db; example_name = example_name, run_group = run_group, case_key = case_key)
        z_axis = loaded.z_axis
        records = loaded.records
    else
        run_group = begin_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        storage = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = case_key, chunk_records = 8)
        result = propagate(
            # A zero frequency grid: with beta2 = 0 the linear operator is zero anyway.
            PulseSpec(samples = a0, delta_time = dt, pulse_period = n * dt, frequency_grid = zeros(ComplexF64, n)),
            (A, w) -> im * (0.5 * beta2) * w^2,
            (A, I, V) -> im * A * (gamma * I + V);
            propagation_distance = z_final,
            num_records = num_records,
            preset = "accuracy",
            exec_options = ExecutionOptions(backend_type = "auto", fft_backend = "auto"),
            storage = StorageOptions(; storage...),
        )
        save_case_from_result!(db; example_name = example_name, run_group = run_group, case_key = case_key,
            result = result, meta = Dict("n" => n, "dt" => dt, "gamma" => gamma, "z_final" => z_final, "pulse_width" => pulse_width))
        z_axis = result.z_axis
        records = permutedims(result.records)
    end

    error_curve = [relative_l2_intensity_error(records[i, :], a0 .* exp.(im * gamma .* abs2.(a0) .* z))
                   for (i, z) in pairs(z_axis)]
    a_final = records[end, :]
    final_error = error_curve[end]
    phase_num = angle.(a_final .* conj.(a0))
    phase_ref = gamma .* abs2.(a0) .* z_final
    phase_error = abs.(angle.(exp.(im .* (phase_num .- phase_ref))))
    support = abs2.(a0) .>= 1e-4 * maximum(abs2, a0)
    power0 = sum(abs2, a0)
    power_drift = abs(sum(abs2, a_final) - power0) / max(power0, 1e-15)

    output_dir = args[:output_dir]
    plot_curves(t[support], ["Numerical" => phase_num[support], "Analytical" => phase_ref[support]],
        joinpath(output_dir, "spm_final_phase_shift_comparison.png"); xlabel = "Time t", ylabel = "Phase shift (rad)")
    plot_intensity_map(t, z_axis, abs2.(records), joinpath(output_dir, "spm_time_intensity_propagation.png");
        xlabel = "Time t")
    freq_axis = fftshift(FFTW.fftfreq(n, 1.0 / dt))
    plot_intensity_map(freq_axis, z_axis, abs2.(fftshift(fft(records, 2), 2)),
        joinpath(output_dir, "spm_frequency_intensity_propagation.png");
        xlabel = "Frequency detuning (1/time)", colorbar_label = "Normalised spectral intensity")
    plot_curves(z_axis, ["Error" => error_curve], joinpath(output_dir, "spm_error_over_propagation.png");
        xlabel = "Propagation distance z", ylabel = "Relative L2 intensity error")

    println("SPM analytical validation summary (run_group=$(run_group)):")
    report_metric("final mean pointwise relative error", final_error)
    report_metric("power drift", power_drift)
    println("  max wrapped phase error (pulse support) = $(maximum(phase_error[support]))")
    return final_error
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
