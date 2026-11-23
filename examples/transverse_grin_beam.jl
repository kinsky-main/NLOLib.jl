# Off-axis Gaussian beam in a parabolic-index (GRIN) medium, x-y only.
#
# A purely transverse run (tensor_nt = 1): the field is an x-y sheet propagated
# along z by
#
#     dA/dz = i beta_t (kx^2 + ky^2) A + i V(x, y) A
#
# with beta_t = -1/(2 k0) (diffraction) and V = -(k0 g^2 / 2)(x^2 + y^2), the
# GRIN lens, supplied as a potential grid. A width-matched Gaussian launched at
# rest off-axis oscillates rigidly about the axis with pitch 2 pi / g, so the
# numerical centroid is checked against x_offset cos(g z). Lengths in um.

using NonlinearOpticsExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using NonlinearOptics

# nlolib-example-parameters: begin
lambda0 = 1.03
n0 = 1.45
g_grin = 0.002
nx = 128
ny = 128
dx = 1.5
dy = 1.5
x_offset = 20.0
num_records = 96
# nlolib-example-parameters: end

function main(argv = ARGS)
    args = parse_example_args("transverse_grin_beam", "Off-axis Gaussian beam oscillating in a GRIN medium (x-y only).", argv)
    k0 = 2.0 * π * n0 / lambda0
    z_final = 2.0 * π / g_grin
    w_matched = sqrt(2.0 / (k0 * g_grin))
    beta_t = -1.0 / (2.0 * k0)
    grin_depth = -0.5 * k0 * g_grin^2
    println("matched waist = $(round(w_matched, digits = 2)) um, pitch period = $(round(z_final, digits = 1)) um")

    x = centered_spatial_grid(nx, dx)
    y = centered_spatial_grid(ny, dy)
    # Flat layout: y fastest, then x (tensor_nt = 1).
    xx = [x[ix] for iy in 1:ny, ix in 1:nx]
    yy = [y[iy] for iy in 1:ny, ix in 1:nx]
    field0 = ComplexF64.(exp.(-(((xx .- x_offset) .^ 2 .+ yy .^ 2) ./ w_matched^2)))
    potential = ComplexF64.(grin_depth .* (xx .^ 2 .+ yy .^ 2))

    result = propagate(
        PulseSpec(samples = vec(field0), delta_time = 1.0, pulse_period = 1.0, frequency_grid = [0.0im],
                  tensor_nt = 1, tensor_nx = nx, tensor_ny = ny, delta_x = dx, delta_y = dy,
                  potential_grid = vec(potential)),
        (A, wt, kx, ky) -> im * beta_t * (kx^2 + ky^2),
        (A, I, V) -> im * A * V;
        propagation_distance = z_final,
        num_records = num_records,
        preset = "accuracy",
        exec_options = ExecutionOptions(backend_type = "auto"),
    )
    z_axis = result.z_axis
    intensity = abs2.(reshape(result.records, ny, nx, length(z_axis)))
    power = vec(sum(intensity, dims = (1, 2)))
    power_drift = abs(power[end] - power[1]) / max(power[1], 1e-12)
    centroid_x = vec(sum(intensity .* xx, dims = (1, 2))) ./ max.(power, 1e-30)
    centroid_analytic = x_offset .* cos.(g_grin .* z_axis)
    centroid_error = maximum(abs.(centroid_x .- centroid_analytic))

    output_dir = args[:output_dir]
    plot_intensity_map(x, z_axis, permutedims(intensity[ny ÷ 2 + 1, :, :]),
        joinpath(output_dir, "transverse_grin_beam_xz_intensity.png");
        xlabel = "x (um)", ylabel = "Propagation distance z (um)")
    plot_curves(z_axis, ["nlolib centroid" => centroid_x, "x0 cos(g z)" => centroid_analytic],
        joinpath(output_dir, "transverse_grin_beam_centroid.png");
        xlabel = "Propagation distance z (um)", ylabel = "Beam centroid x (um)")

    println("propagated $(round(z_final, digits = 1)) um on a $(nx)x$(ny) grid ($(length(z_axis)) records)")
    report_metric("power drift", power_drift)
    report_metric("max centroid deviation from x_offset cos(g_grin z)", centroid_error)
    return centroid_error
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
