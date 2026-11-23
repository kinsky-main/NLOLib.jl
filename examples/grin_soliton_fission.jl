# Higher-order soliton fission in a graded-index (GRIN) fibre, (t, x, y).
#
# A 4th-order soliton (by launch power) enters the fundamental mode of a GRIN
# multimode fibre. The field A(t, x, y, z), in sqrt(W/um^2), obeys
#
#     dA/dz = i (beta2/2 w^2 + beta3/6 w^3 - (kx^2 + ky^2)/(2 k0)) A
#           + i (gamma |A|^2 + V(x, y)) A
#
# with the parabolic GRIN potential V = -(k0 g^2 / 2)(x^2 + y^2). The fibre
# holds the beam in its fundamental mode while, in time, third-order
# dispersion splits the input into fundamental solitons and sheds a dispersive
# wave on the blue side. Units: um, fs, W.
#
# The operator's wt is the FFT frequency (+2 pi k / T); with the envelope
# convention exp(-i w t) the physical offset is w = -wt, so the odd (beta3)
# term changes sign in the operator.

using NonlinearOpticsExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using FFTW
using NonlinearOptics

# nlolib-example-parameters: begin
lambda0 = 1.55
n0 = 1.45
beta2 = -0.028
beta3 = 0.3
n2 = 2.6e-08
soliton_order = 4
t0 = 30.0
nt = 1024
dt = 2.5
nx = 24
ny = 24
dx = 2.0
t_launch = -400.0
num_records = 48
error_tolerance = 1e-06
# nlolib-example-parameters: end

decibels(values) = max.(10.0 .* log10.(values ./ maximum(values) .+ 1e-12), -40.0) .+ 40.0

function main(argv = ARGS)
    args = parse_example_args("grin_soliton_fission", "Fourth-order soliton fission in a GRIN multimode fibre.", argv)
    k0 = 2.0 * π * n0 / lambda0
    g_grin = sqrt(2.0 * 0.01) / 25.0
    gamma = 2.0 * π * n2 / lambda0
    w0 = sqrt(2.0 / (k0 * g_grin))  # fundamental GRIN mode radius
    ld = t0^2 / abs(beta2)
    gamma_eff = gamma / (π * w0^2)
    p0 = soliton_order^2 / (gamma_eff * ld)
    z_final = 1.2 * ld
    println("mode radius $(round(w0, digits = 1)) um, LD = $(round(ld / 1e3, digits = 1)) mm, " *
            "P0 = $(round(p0 / 1e6, digits = 2)) MW, z = $(round(z_final / 1e3, digits = 1)) mm")

    t = (collect(0:nt - 1) .- nt / 2) .* dt
    x = (collect(0:nx - 1) .- nx / 2) .* dx
    y = (collect(0:ny - 1) .- ny / 2) .* dx
    peak_amp = sqrt(2.0 * p0 / (π * w0^2))
    field0 = ComplexF64[exp(-(x[ix]^2 + y[iy]^2) / w0^2) * peak_amp / cosh((t[it] - t_launch) / t0)
                        for it in 1:nt, iy in 1:ny, ix in 1:nx]
    potential = ComplexF64[-0.5 * k0 * g_grin^2 * (x[ix]^2 + y[iy]^2) for iy in 1:ny, ix in 1:nx]
    b2 = beta2 / 2.0
    b3 = beta3 / 6.0
    bt = -1.0 / (2.0 * k0)

    result = propagate(
        PulseSpec(samples = vec(field0), delta_time = dt, pulse_period = nt * dt,
                  tensor_nt = nt, tensor_nx = nx, tensor_ny = ny, delta_x = dx, delta_y = dx,
                  potential_grid = vec(potential)),
        (A, wt, kx, ky) -> im * (b2 * wt^2 - b3 * wt^3 + bt * (kx^2 + ky^2)),
        (A, I, V) -> im * A * (gamma * I + V);
        propagation_distance = z_final,
        num_records = num_records,
        preset = "balanced",
        error_tolerance = error_tolerance,
        starting_step_size = z_final / 500.0,
        max_step_size = z_final / 50.0,
        min_step_size = z_final / 1e5,
        capture_step_history = true,
        exec_options = ExecutionOptions(backend_type = "auto"),
    )
    steps = result.step_history
    println("$(length(steps.step_size)) accepted steps, step $(round(minimum(steps.step_size), digits = 2)) to " *
            "$(round(maximum(steps.step_size), digits = 2)) um, max local error $(maximum(steps.error))")

    z_mm = result.z_axis ./ 1e3
    fields = reshape(result.records, nt, ny, nx, length(z_mm))
    power_t = permutedims(dropdims(sum(abs2.(fields), dims = (2, 3)), dims = (2, 3))) .* dx^2  # W
    spectrum = permutedims(dropdims(sum(abs2.(fft(fields, 1)), dims = (2, 3)), dims = (2, 3)))
    fluence_x = permutedims(dropdims(sum(abs2.(fields[:, ny ÷ 2 + 1, :, :]), dims = 1), dims = 1))
    w = -2.0 * π .* FFTW.fftfreq(nt, 1.0 / dt)
    order = sortperm(w)
    energy_drift = abs(sum(power_t[end, :]) / sum(power_t[1, :]) - 1.0)
    peak_in = maximum(power_t[1, :]) / 1e6
    peak_out = maximum(power_t[end, :]) / 1e6

    output_dir = args[:output_dir]
    plot_intensity_map(t, z_mm, decibels(power_t), joinpath(output_dir, "grin_soliton_fission_power_db.png");
        xlabel = "t (fs)", ylabel = "z (mm)", colorbar_label = "Power (dB + 40)", normalize = false)
    plot_intensity_map(w[order], z_mm, decibels(spectrum[:, order]), joinpath(output_dir, "grin_soliton_fission_spectrum_db.png");
        xlabel = "omega - omega0 (rad/fs)", ylabel = "z (mm)", colorbar_label = "Spectrum (dB + 40)", normalize = false)
    plot_intensity_map(x, z_mm, fluence_x, joinpath(output_dir, "grin_soliton_fission_fluence_x.png");
        xlabel = "x (um)", ylabel = "z (mm)")

    report_metric("energy drift", energy_drift)
    report_metric("peak power in", peak_in)
    report_metric("peak power out", peak_out)
    report_metric("accepted steps", length(steps.step_size))
    println("(peak powers in MW)")
    return energy_drift
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
