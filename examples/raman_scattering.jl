using NonlinearOpticsExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using CairoMakie
using FFTW
using NonlinearOptics

# nlolib-example-parameters: begin
n = 4096
dt = 0.002
beta2 = -0.01
gamma = 1.4
pulse_width = 0.08
z_final = 0.4
num_records = 80
f_r = 0.18
tau1 = 0.0522
tau2 = 0.032
shock_omega0 = 0.0
error_tolerance = 2e-08
# nlolib-example-parameters: end
# Derived: step sizes z_final / 500, / 80 and / 20000.

function spectral_centroid(freq_axis, spectral_intensity)
    weights = Float64.(spectral_intensity)
    numer = sum(weights .* reshape(freq_axis, 1, :), dims = 2)
    denom = max.(sum(weights, dims = 2), 1e-15)
    return vec(numer ./ denom)
end

function optics_shifted_spectrum_map(field_records, dt)
    fields = ComplexF64.(field_records)
    n = size(fields, 2)
    omega_axis = fftshift(2.0 * π .* FFTW.fftfreq(n, 1.0 / dt))
    spectra = fftshift(ifft(fields, 2), 2)
    return omega_axis, abs2.(spectra)
end

function default_raman_response(n, dt, tau1, tau2)
    t = collect(0:n - 1) .* Float64(dt)
    coef = (tau1^2 + tau2^2) / (tau1 * tau2^2)
    response = coef .* exp.(-t ./ tau2) .* sin.(t ./ tau1)
    area = sum(response) * dt
    area > 0.0 || error("invalid Raman response normalization area")
    return response ./ area
end

function raman_centroid_rhs_moment(records, dt, gamma, f_r, tau1, tau2)
    fields = ComplexF64.(records)
    n = size(fields, 2)
    omega_unshifted = 2.0 * π .* FFTW.fftfreq(n, 1.0 / dt)
    h_r = default_raman_response(n, dt, tau1, tau2)
    h_r_fft = fft(h_r)
    rhs = zeros(Float64, size(fields, 1))
    for idx in axes(fields, 1)
        intensity = abs2.(view(fields, idx, :))
        delayed = real.(ifft(fft(intensity) .* h_r_fft))
        delayed_dt = real.(ifft((1im .* omega_unshifted) .* fft(delayed)))
        energy = max(sum(intensity) * dt, 1e-15)
        rhs[idx] = (gamma * f_r / energy) * sum(intensity .* delayed_dt) * dt
    end
    return rhs
end

function cumulative_trapezoid(y, x)
    values = Float64.(y)
    axis = Float64.(x)
    out = zeros(Float64, length(values))
    for i in 2:length(values)
        out[i] = out[i - 1] + 0.5 * (values[i] + values[i - 1]) * (axis[i] - axis[i - 1])
    end
    return out
end

function save_raman_plots(output_dir, z_axis_norm, t_axis, raman_records, omega_axis, raman_spec_map, centered_num, centered_pred)
    mkpath(output_dir)

    fig1 = styled_figure()
    ax1 = Axis(fig1[1, 1], xlabel = "time", ylabel = "z / Ld", title = "Temporal intensity")
    hm1 = heatmap!(
        ax1,
        t_axis,
        z_axis_norm,
        permutedims(normalized_plot_data(abs2.(raman_records)), (2, 1)),
        colormap = nlolib_hdr_colormap(),
        colorrange = (0.0, 1.0),
    )
    Colorbar(fig1[1, 2], hm1, label = "Normalized intensity")
    save_example_figure(joinpath(output_dir, "raman_time_intensity.png"), fig1)

    fig2 = styled_figure()
    ax2 = Axis(fig2[1, 1], xlabel = "omega", ylabel = "z / Ld", title = "Spectral intensity")
    hm2 = heatmap!(
        ax2,
        omega_axis,
        z_axis_norm,
        permutedims(normalized_plot_data(raman_spec_map), (2, 1)),
        colormap = nlolib_hdr_colormap(),
        colorrange = (0.0, 1.0),
    )
    Colorbar(fig2[1, 2], hm2, label = "Normalized intensity")
    save_example_figure(joinpath(output_dir, "raman_spectral_intensity.png"), fig2)

    fig3 = styled_figure()
    ax3 = Axis(fig3[1, 1], xlabel = "z / Ld", ylabel = "Delta centroid", title = "Centroid shift")
    lines!(ax3, z_axis_norm, centered_num, label = "Numerical")
    lines!(ax3, z_axis_norm, centered_pred, label = "Moment prediction")
    axislegend(ax3, position = :rt)
    save_example_figure(joinpath(output_dir, "raman_centroid_shift.png"), fig3)
end

function main(argv = ARGS)
    args = parse_example_args("raman_scattering", "Raman self-frequency-shift validation with DB-backed run/replot.", argv)
    activate_example_theme!()
    NonlinearOptics.set_progress_options(enabled = true, milestone_percent = 2, emit_on_step_adjust = true)
    NonlinearOptics.set_log_level("info")
    db = ExampleRunDB(args[:db_path])
    example_name = "raman_scattering"
    kerr_case_key = "kerr_only"
    raman_case_key = "kerr_raman_shock"

    t_axis = centered_time_grid(n, dt)
    omega = 2.0 * π .* FFTW.fftfreq(n, 1.0 / dt)
    p0 = abs(beta2) / (gamma * pulse_width * pulse_width)
    ld = (pulse_width * pulse_width) / abs(beta2)
    field0 = ComplexF64.(sqrt(p0) ./ cosh.(t_axis ./ pulse_width))

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        loaded_kerr = load_case(db; example_name = example_name, run_group = run_group, case_key = kerr_case_key)
        loaded_raman = load_case(db; example_name = example_name, run_group = run_group, case_key = raman_case_key)
        z_axis = loaded_raman.z_axis
        kerr_records = loaded_kerr.records
        raman_records = loaded_raman.records
    else
        run_group = begin_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        pulse = PulseSpec(samples = field0, delta_time = dt, pulse_period = n * dt, frequency_grid = ComplexF64.(omega))
        linear = (A, w) -> im * (0.5 * beta2) * w^2
        exec = ExecutionOptions(backend_type = "auto")

        base_kwargs = (
            propagation_distance = z_final,
            num_records = num_records,
            starting_step_size = z_final / 500.0,
            max_step_size = z_final / 80.0,
            min_step_size = z_final / 20000.0,
            error_tolerance = error_tolerance,
            exec_options = exec,
        )

        storage_kerr = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = kerr_case_key, chunk_records = 8)
        kerr_result = NonlinearOptics.propagate(
            pulse,
            linear,
            (A, I) -> im * gamma * A * I;
            base_kwargs...,
            storage = StorageOptions(; storage_kerr...),
        )
        save_case_from_result!(db;
            example_name = example_name,
            run_group = run_group,
            case_key = kerr_case_key,
            result = kerr_result,
            meta = Dict("n" => n, "dt" => dt, "beta2" => beta2, "gamma" => gamma, "z_final" => z_final, "pulse_width" => pulse_width))

        storage_raman = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = raman_case_key, chunk_records = 8)
        raman_result = NonlinearOptics.propagate(
            pulse,
            linear,
            (A, I) -> 0.0;
            base_kwargs...,
            nonlinear_model = "kerr_raman",
            nonlinear_gamma = gamma,
            raman_fraction = f_r,
            raman_tau1 = tau1,
            raman_tau2 = tau2,
            shock_omega0 = shock_omega0,
            storage = StorageOptions(; storage_raman...),
        )
        save_case_from_result!(db;
            example_name = example_name,
            run_group = run_group,
            case_key = raman_case_key,
            result = raman_result,
            meta = Dict("n" => n, "dt" => dt, "beta2" => beta2, "gamma" => gamma, "z_final" => z_final, "pulse_width" => pulse_width, "f_r" => f_r, "tau1" => tau1, "tau2" => tau2, "shock_omega0" => shock_omega0))

        z_axis = raman_result.z_axis
        kerr_records = permutedims(kerr_result.records, (2, 1))
        raman_records = permutedims(raman_result.records, (2, 1))
    end

    omega_axis, kerr_spec_map = optics_shifted_spectrum_map(kerr_records, dt)
    _, raman_spec_map = optics_shifted_spectrum_map(raman_records, dt)
    kerr_centroid = spectral_centroid(omega_axis, kerr_spec_map)
    raman_centroid = spectral_centroid(omega_axis, raman_spec_map)
    centroid_rhs = -raman_centroid_rhs_moment(raman_records, dt, gamma, f_r, tau1, tau2)
    predicted_centroid = raman_centroid[1] .+ cumulative_trapezoid(centroid_rhs, z_axis)
    centered_num = raman_centroid .- raman_centroid[1]
    centered_pred = predicted_centroid .- predicted_centroid[1]
    centroid_curve_rel_error = relative_l2_intensity_error(centered_num, centered_pred)

    save_raman_plots(args[:output_dir], z_axis ./ ld, t_axis, raman_records, omega_axis, raman_spec_map, centered_num, centered_pred)

    println("raman scattering summary")
    report_metric("centroid-curve relative error vs prediction", centroid_curve_rel_error)
    println("  final centroid shift (raman-kerr) = $(raman_centroid[end] - kerr_centroid[end])")
    println("  final predicted shift = $(predicted_centroid[end] - predicted_centroid[1])")
    return centroid_curve_rel_error
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
