# Quick start: a chirped Gaussian under pure dispersion, two records.
#
# The shortest complete propagate call: a pulse, a linear and a nonlinear
# operator written as plain functions, and the "balanced" preset. Dispersion
# alone conserves power, so the initial and final powers agree.

using NonlinearOpticsExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using NonlinearOptics

# nlolib-example-parameters: begin
n = 1024
dt = 0.01
sigma = 0.2
chirp = 12.0
beta2 = 0.05
gamma = 0.0
z_final = 1.0
num_records = 2
# nlolib-example-parameters: end

function main(argv = ARGS)
    parse_example_args("quickstart", "Chirped Gaussian under pure dispersion: the shortest propagate call.", argv)
    t = centered_time_grid(n, dt)
    field0 = ComplexF64.(exp.(-(t ./ sigma) .^ 2) .* exp.(-im * chirp .* t))

    result = propagate(
        PulseSpec(samples = field0, delta_time = dt),
        (A, w) -> im * (0.5 * beta2) * w^2,
        (A, I) -> im * gamma * A * I;
        propagation_distance = z_final,
        num_records = num_records,
        preset = "balanced",
        exec_options = ExecutionOptions(backend_type = "auto"),
    )
    initial_power = sum(abs2, result.records[:, 1])
    final_power = sum(abs2, result.records[:, end])

    println("quickstart completed: z records $(result.z_axis)")
    report_metric("initial power", initial_power)
    report_metric("final power", final_power)
    return abs(final_power - initial_power) / initial_power
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
