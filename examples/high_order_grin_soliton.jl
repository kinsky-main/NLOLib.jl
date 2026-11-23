# High-order temporal soliton in a GRIN-guided tensor geometry.
#
# A single-lobed guided spatial mode, perturbed by an azimuthal ripple and a
# little seeded noise, carries a higher-order temporal soliton envelope. It
# propagates under anomalous dispersion, diffraction, a parabolic GRIN
# potential and a saturable nonlinearity
#
#     N(A) = i A (gamma I / (1 + I / I_sat) + V).
#
# The radius, overlap with the launch field and power drift are reported.

using NonlinearOpticsExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using FFTW
using LinearAlgebra
using NonlinearOptics
using Random

# nlolib-example-parameters: begin
soliton_order = 4.5
saturation_intensity = 2.0
propagation_periods = 4.0
azimuthal_amplitude = 0.25
azimuthal_order = 4
noise_amplitude = 0.01
noise_seed = 12345
nt = 512
nx = 64
ny = 64
dt = 0.02
dx = 0.06
dy = 0.06
temporal_width = 0.3
mode_width_x = 0.55
mode_width_y = 0.75
spatial_chirp = 1.5
beta2 = -0.08
beta_t = -0.08
grin_strength = 0.0015
gamma_nonlinear = 1.5
num_records = 24
starting_step_size = 0.005
max_step_size = 0.2
min_step_size = 1e-08
error_tolerance = 1e-05
# nlolib-example-parameters: end

dispersion_length() = temporal_width^2 / abs(beta2)
soliton_period() = 0.5 * π * dispersion_length()
fundamental_soliton_power() = abs(beta2) / (gamma_nonlinear * temporal_width^2)

"""
    launch_field(t, x, y)

The launch field as `[t, y, x]`: a sech soliton envelope of the given order
times a chirped, azimuthally rippled Gaussian mode with seeded noise. The
noise comes from Julia's `Xoshiro`, not NumPy's generator, so the draw (and
the metrics, slightly) differ from the Python example.
"""
function launch_field(t, x, y)
    temporal = soliton_order * sqrt(fundamental_soliton_power()) ./ cosh.(t ./ temporal_width)
    xi = randn(Xoshiro(noise_seed), ny, nx)
    function mode(xv, yv, noise)
        envelope = exp(-xv^2 / (2.0 * mode_width_x^2) - yv^2 / (2.0 * mode_width_y^2))
        ripple = 1.0 + azimuthal_amplitude * cos(azimuthal_order * atan(yv, xv)) + noise_amplitude * noise
        return envelope * ripple * exp(im * spatial_chirp * (xv^2 + yv^2))
    end
    transverse = ComplexF64[mode(x[ix], y[iy], xi[iy, ix]) for iy in 1:ny, ix in 1:nx]
    return ComplexF64[temporal[it] * transverse[iy, ix] for it in 1:nt, iy in 1:ny, ix in 1:nx]
end

function overlap_fidelity_curve(records, launch)
    reference = vec(launch)
    return [abs(dot(reference, vec(view(records, :, :, :, k)))) /
            max(norm(reference) * norm(view(records, :, :, :, k)), 1e-30) for k in axes(records, 4)]
end

function rms_radius_curve(records, x, y)
    r2 = [x[ix]^2 + y[iy]^2 for iy in 1:ny, ix in 1:nx]
    return map(axes(records, 4)) do k
        intensity = dropdims(sum(abs2.(view(records, :, :, :, k)), dims = 1), dims = 1)
        sqrt(sum(r2 .* intensity) / max(sum(intensity), 1e-30))
    end
end

function rms_temporal_width_curve(records, t)
    return map(axes(records, 4)) do k
        weights = vec(sum(abs2.(view(records, :, :, :, k)), dims = (2, 3)))
        total = max(sum(weights), 1e-30)
        center = sum(t .* weights) / total
        sqrt(sum((t .- center) .^ 2 .* weights) / total)
    end
end

function main(argv = ARGS)
    args = parse_example_args("high_order_grin_soliton",
        "High-order GRIN-guided nonlinear tensor propagation using the C-backed runtime operators.", argv)
    set_log_level("error")
    set_progress_options(enabled = true, milestone_percent = 2, emit_on_step_adjust = true)
    db = ExampleRunDB(args[:db_path])
    example_name = "high_order_grin_soliton"
    case_key = "nonlinear"

    z_final = propagation_periods * soliton_period()
    t = centered_time_grid(nt, dt)
    x = centered_spatial_grid(nx, dx)
    y = centered_spatial_grid(ny, dy)
    field0 = launch_field(t, x, y)

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        loaded = load_case(db; example_name = example_name, run_group = run_group, case_key = case_key)
        z_axis = loaded.z_axis
        records = reshape(permutedims(loaded.records), nt, ny, nx, length(z_axis))
    else
        run_group = begin_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        storage = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = case_key, chunk_records = 4)
        potential = ComplexF64[grin_strength * (x[ix]^2 + y[iy]^2) for iy in 1:ny, ix in 1:nx]
        half_beta2 = 0.5 * beta2
        result = propagate(
            PulseSpec(samples = vec(field0), delta_time = dt, pulse_period = nt * dt,
                      frequency_grid = ComplexF64.(2.0 * π .* FFTW.fftfreq(nt, 1.0 / dt)),
                      tensor_nt = nt, tensor_nx = nx, tensor_ny = ny, delta_x = dx, delta_y = dy,
                      potential_grid = vec(potential)),
            (A, wt, kx, ky) -> im * (half_beta2 * wt^2 + beta_t * (kx^2 + ky^2)),
            (A, I, V) -> im * A * (gamma_nonlinear * (I / (1.0 + I / saturation_intensity)) + V);
            linear_step_operator = (A, D, h) -> exp(h * D),
            propagation_distance = z_final,
            starting_step_size = starting_step_size,
            max_step_size = max_step_size,
            min_step_size = min_step_size,
            error_tolerance = error_tolerance,
            num_records = num_records,
            exec_options = ExecutionOptions(backend_type = "auto", fft_backend = "auto"),
            storage = StorageOptions(; storage...),
        )
        save_case_from_result!(db; example_name = example_name, run_group = run_group, case_key = case_key,
            result = result, meta = Dict(
                "nt" => nt, "nx" => nx, "ny" => ny, "dt" => dt, "dx" => dx, "dy" => dy,
                "soliton_order" => soliton_order, "saturation_intensity" => saturation_intensity,
                "propagation_periods" => propagation_periods, "noise_seed" => noise_seed))
        z_axis = result.z_axis
        records = reshape(result.records, nt, ny, nx, length(z_axis))
    end

    radius = rms_radius_curve(records, x, y)
    temporal_width_curve = rms_temporal_width_curve(records, t)
    overlap = overlap_fidelity_curve(records, field0)
    power = [sum(abs2, view(records, :, :, :, k)) for k in axes(records, 4)]
    power_drift_curve = abs.(power .- power[1]) ./ max(power[1], 1e-30)
    z_scaled = z_axis ./ dispersion_length()

    output_dir = args[:output_dir]
    centerline = permutedims(dropdims(sum(abs2.(records[:, ny ÷ 2 + 1, :, :]), dims = 1), dims = 1))
    plot_intensity_map(x ./ mode_width_x, z_scaled, centerline,
        joinpath(output_dir, "high_order_grin_soliton_centerline_intensity.png");
        xlabel = "x / w_x", ylabel = "z / L_D")
    temporal = permutedims(dropdims(sum(abs2.(records), dims = (2, 3)), dims = (2, 3)))
    plot_intensity_map(t ./ temporal_width, z_scaled, temporal,
        joinpath(output_dir, "high_order_grin_soliton_temporal_intensity.png");
        xlabel = "t / T0", ylabel = "z / L_D", xlimits = (-5, 5))
    plot_curves(z_scaled, ["RMS radius" => radius], joinpath(output_dir, "high_order_grin_soliton_radius.png");
        xlabel = "z / L_D", ylabel = "RMS radius")
    plot_curves(z_scaled, ["Overlap fidelity" => overlap], joinpath(output_dir, "high_order_grin_soliton_overlap.png");
        xlabel = "z / L_D", ylabel = "Overlap with launch field")
    plot_curves(z_scaled, ["Power drift" => power_drift_curve], joinpath(output_dir, "high_order_grin_soliton_power_drift.png");
        xlabel = "z / L_D", ylabel = "Relative power drift")
    xy = dropdims(sum(abs2.(records), dims = 1), dims = 1)
    plot_3d_intensity_contours_propagation(x ./ mode_width_x, y ./ mode_width_y, z_scaled, permutedims(xy, (3, 1, 2)),
        joinpath(output_dir, "high_order_grin_soliton_3d_intensity_contours.png"); input_is_intensity = true)

    println("high-order GRIN saturable soliton summary")
    println("  soliton order = $(soliton_order), saturation intensity = $(saturation_intensity)")
    report_metric("final radius / launch radius", radius[end] / radius[1])
    println("  radius excursion = $(maximum(radius) / radius[1])")
    println("  final temporal width / launch width = $(temporal_width_curve[end] / temporal_width_curve[1])")
    report_metric("min overlap fidelity", minimum(overlap))
    report_metric("power drift", power_drift_curve[end])
    println("  max power drift = $(maximum(power_drift_curve))")
    return maximum(power_drift_curve)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
