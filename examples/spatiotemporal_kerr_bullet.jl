# Spatiotemporal Gaussian packet in a bulk Kerr medium with anomalous dispersion.
#
#     dA/dz = i (b2 wt^2 + beta_t (kx^2 + ky^2)) A + i gamma |A|^2 A
#
# with b2 = beta2 / 2 (anomalous) and beta_t = -1/(2 k0) (diffraction). The
# waist and duration make the Rayleigh range equal the dispersion length
# (zR = k0 w0^2 / 2 = T0^2 / |beta2| = l0): the packet spreads equally in time
# and space, the regime where a light bullet can form.
#
# Two runs on the same grid: a linear one (gamma = 0), checked against the
# Gaussian laws T0 sqrt(1 + (z/l0)^2) and w0 sqrt(1 + (z/l0)^2), and a Kerr one
# in which self-focusing arrests part of the spread. Lengths in um, times in
# fs; the envelope has unit peak amplitude.

using NonlinearOpticsExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using FFTW
using NonlinearOptics

# nlolib-example-parameters: begin
lambda0 = 1.55
n0 = 1.444
beta2 = -0.028
l0 = 100000.0
nt = 32
nx = 64
ny = 64
dt = 20.0
dx = 32.0
dy = 32.0
num_records = 48
# nlolib-example-parameters: end

function run_case(gamma_kerr, field0, omega, z_final, beta_t, t, x)
    b2 = 0.5 * beta2
    result = propagate(
        PulseSpec(samples = vec(field0), delta_time = dt, pulse_period = nt * dt, frequency_grid = ComplexF64.(omega),
                  tensor_nt = nt, tensor_nx = nx, tensor_ny = ny, delta_x = dx, delta_y = dy),
        (A, wt, kx, ky) -> im * (b2 * wt^2 + beta_t * (kx^2 + ky^2)),
        (A, I) -> im * gamma_kerr * A * I;
        propagation_distance = z_final,
        num_records = num_records,
        preset = "accuracy",
        exec_options = ExecutionOptions(backend_type = "auto"),
    )
    z_axis = result.z_axis
    intensity = abs2.(reshape(result.records, nt, ny, nx, length(z_axis)))
    energy = vec(sum(intensity, dims = (1, 2, 3)))
    sigma_t = sqrt.(vec(sum(intensity .* (t .^ 2), dims = (1, 2, 3))) ./ energy)
    sigma_x = sqrt.(vec(sum(intensity .* reshape(x .^ 2, 1, 1, nx), dims = (1, 2, 3))) ./ energy)
    return (
        z = z_axis,
        power_t = permutedims(dropdims(sum(intensity, dims = (2, 3)), dims = (2, 3))),
        fluence_x = permutedims(dropdims(sum(intensity, dims = (1, 2)), dims = (1, 2))),
        sigma_t = sigma_t,
        sigma_x = sigma_x,
        peak = vec(maximum(intensity, dims = (1, 2, 3))),
        energy_drift = abs(energy[end] - energy[1]) / max(energy[1], 1e-30),
    )
end

function main(argv = ARGS)
    args = parse_example_args("spatiotemporal_kerr_bullet", "Spatiotemporal Gaussian packet in bulk Kerr media, linear vs Kerr.", argv)
    k0 = 2.0 * π * n0 / lambda0
    w0 = sqrt(2.0 * l0 / k0)
    t0 = sqrt(l0 * abs(beta2))
    z_final = 1.5 * l0
    gamma_kerr = 0.6 / l0
    beta_t = -1.0 / (2.0 * k0)
    println("w0 = $(round(w0, digits = 1)) um, T0 = $(round(t0, digits = 1)) fs, zR = LD = $(l0 * 1e-3) mm, z_end = $(z_final * 1e-3) mm")

    t = centered_time_grid(nt, dt)
    x = centered_spatial_grid(nx, dx)
    y = centered_spatial_grid(ny, dy)
    field0 = ComplexF64[exp(-((x[ix]^2 + y[iy]^2) / w0^2)) * exp(-t[it]^2 / (2.0 * t0^2))
                        for it in 1:nt, iy in 1:ny, ix in 1:nx]
    omega = 2.0 * π .* FFTW.fftfreq(nt, 1.0 / dt)

    linear = run_case(0.0, field0, omega, z_final, beta_t, t, x)
    kerr = run_case(gamma_kerr, field0, omega, z_final, beta_t, t, x)
    growth = sqrt.(1.0 .+ (linear.z ./ l0) .^ 2)
    width_error = maximum(abs.(sqrt(2.0) .* linear.sigma_t .- t0 .* growth)) / t0
    radius_error = maximum(abs.(2.0 .* linear.sigma_x .- w0 .* growth)) / w0

    output_dir = args[:output_dir]
    z_mm = kerr.z .* 1e-3
    plot_intensity_map(t, z_mm, kerr.power_t, joinpath(output_dir, "kerr_bullet_power_vs_time.png");
        xlabel = "t (fs)", ylabel = "z (mm)")
    plot_intensity_map(x, z_mm, kerr.fluence_x, joinpath(output_dir, "kerr_bullet_fluence_vs_x.png");
        xlabel = "x (um)", ylabel = "z (mm)")
    plot_curves(z_mm, ["Kerr" => sqrt(2.0) .* kerr.sigma_t, "linear" => sqrt(2.0) .* linear.sigma_t],
        joinpath(output_dir, "kerr_bullet_pulse_width.png"); xlabel = "z (mm)", ylabel = "Pulse width T (fs)")
    plot_curves(z_mm, ["Kerr" => 2.0 .* kerr.sigma_x, "linear" => 2.0 .* linear.sigma_x],
        joinpath(output_dir, "kerr_bullet_beam_radius.png"); xlabel = "z (mm)", ylabel = "Beam radius w (um)")

    println("final peak intensity: linear $(round(linear.peak[end], digits = 3)), Kerr $(round(kerr.peak[end], digits = 3)) (input 1)")
    println("energy drift: linear $(linear.energy_drift), Kerr $(kerr.energy_drift)")
    report_metric("linear-run width error vs analytic Gaussian", width_error)
    report_metric("linear-run radius error vs analytic Gaussian", radius_error)
    report_metric("energy drift", kerr.energy_drift)
    return max(width_error, radius_error)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
