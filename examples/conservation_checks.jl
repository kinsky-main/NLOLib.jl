# Energy, momentum and Hamiltonian conservation for a lossless NLSE run.
#
# A carrier-tilted fundamental soliton (the tilt makes the momentum nonzero)
# propagates under dispersion and Kerr nonlinearity; the three invariants are
# computed spectrally from every record and their drifts reported.

using NLOLibExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using FFTW
using NLOLib

# nlolib-example-parameters: begin
n = 1024
tfwhm = 0.1
beta2 = -0.01
gamma = 0.01
carrier = 12.0
num_records = 100
error_tolerance = 1e-06
# nlolib-example-parameters: end
# Derived: dt = 16 t0 / n, z_final = 0.2 t0^2 / |beta2|, step sizes
# z_final / 1000, / 600 and / 15000.

function invariants(a, omega, dt)
    da = ifft(im .* omega .* fft(a))
    energy = sum(abs2, a) * dt
    momentum = imag(sum(conj.(a) .* da)) * dt
    hamiltonian = (0.5 * beta2 * sum(abs2, da) - 0.5 * gamma * sum(abs2.(a) .^ 2)) * dt
    return energy, momentum, hamiltonian
end

relative_drift(values) = (values .- values[1]) ./ max(abs(values[1]), 1e-15)

function main(argv = ARGS)
    args = parse_example_args("conservation_checks", "Conservation checks with DB-backed run/replot.", argv)
    set_log_level("error")
    set_progress_options(enabled = false)
    db = ExampleRunDB(args[:db_path])
    example_name = "conservation_checks"
    case_key = "default"

    t0 = tfwhm / (2.0 * acosh(sqrt(2.0)))
    p0 = abs(beta2) / (gamma * t0^2)
    ld = t0^2 / abs(beta2)
    z_final = 0.2 * ld
    dt = 16.0 * t0 / n
    t = centered_time_grid(n, dt)
    omega = 2.0 * π .* FFTW.fftfreq(n, 1.0 / dt)
    a0 = ComplexF64.((sqrt(p0) ./ cosh.(t ./ t0)) .* exp.(-im * carrier .* t))

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        loaded = load_case(db; example_name = example_name, run_group = run_group, case_key = case_key)
        z_axis = loaded.z_axis
        records = loaded.records
    else
        run_group = begin_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        storage = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = case_key, chunk_records = 8)
        result = propagate(
            PulseSpec(samples = a0, delta_time = dt, pulse_period = n * dt, frequency_grid = ComplexF64.(omega)),
            (A, w) -> im * (0.5 * beta2) * w^2,
            (A, I, V) -> im * A * (gamma * I + V);
            propagation_distance = z_final,
            num_records = num_records,
            starting_step_size = z_final / 1000.0,
            max_step_size = z_final / 600.0,
            min_step_size = z_final / 15000.0,
            error_tolerance = error_tolerance,
            exec_options = ExecutionOptions(backend_type = "auto", fft_backend = "auto"),
            storage = StorageOptions(; storage...),
        )
        save_case_from_result!(db; example_name = example_name, run_group = run_group, case_key = case_key,
            result = result, meta = Dict("n" => n, "dt" => dt, "beta2" => beta2, "gamma" => gamma))
        z_axis = result.z_axis
        records = permutedims(result.records)
    end

    values = [invariants(records[i, :], omega, dt) for i in axes(records, 1)]
    energy_drift = relative_drift(getindex.(values, 1))
    momentum_drift = relative_drift(getindex.(values, 2))
    hamiltonian_drift = relative_drift(getindex.(values, 3))

    plot_curves(z_axis ./ ld,
        ["Energy drift" => energy_drift, "Momentum drift" => momentum_drift, "Hamiltonian drift" => hamiltonian_drift],
        joinpath(args[:output_dir], "energy_conservation_relative_invariant_drift.png");
        xlabel = "Normalised propagation z / L_D", ylabel = "Relative drift")

    println("conservation-check summary (run_group=$(run_group)):")
    report_metric("max energy drift", maximum(abs, energy_drift))
    report_metric("max momentum drift", maximum(abs, momentum_drift))
    report_metric("max Hamiltonian drift", maximum(abs, hamiltonian_drift))
    return maximum(abs, energy_drift), maximum(abs, momentum_drift), maximum(abs, hamiltonian_drift)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
