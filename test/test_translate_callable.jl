# Tests for the function-operator tracer. These exercise translation only, so
# they need no built shared library.

using Test
using Printf
using NLOLib
using NLOLib: translate_callable, runtime_context_symbols

function dispersion_fixture(A, w)
    return im * (0.02 * w^2)
end

@testset "translate_callable" begin

@testset "broadcast and imaginary unit" begin
    # Broadcasting dispatches to the same methods as the scalar operators, and
    # a captured number is inlined rather than given a constant slot.
    beta2 = 0.5
    expr, constants = translate_callable((A, w) -> im * beta2 .* (w .^ 2),
                                         "dispersion_factor")
    @test expr == "0.5*i*w^2"
    @test isempty(constants)
end

@testset "captured numbers are inlined" begin
    beta2 = 0.04
    beta3 = -0.003
    beta4 = 0.0002
    expr, constants = translate_callable(
        (A, w) -> im * (beta2 * w^2 + beta3 * w^3 + beta4 * w^4),
        "dispersion_factor")
    @test occursin("^2", expr)
    @test occursin("^3", expr)
    @test occursin("^4", expr)
    # RUNTIME_OPERATOR_CONSTANTS_MAX does not cap a traced function.
    @test isempty(constants)
end

@testset "ordinary Julia in the body" begin
    lambda = 1.55e-6
    params = (beta2 = 0.03,)
    betas = [0.0, 0.0, 0.07]

    @test translate_callable((A, w) -> pi * w, "dispersion_factor")[1] ==
          string(@sprintf("%.17g", pi), "*w")
    @test translate_callable((A, w) -> (2pi / lambda) * w, "dispersion_factor")[1] ==
          string(@sprintf("%.17g", 2pi / lambda), "*w")
    @test translate_callable((A, w) -> params.beta2 * w, "dispersion_factor")[1] ==
          string(@sprintf("%.17g", 0.03), "*w")
    @test translate_callable((A, w) -> betas[3] * w, "dispersion_factor")[1] ==
          string(@sprintf("%.17g", 0.07), "*w")
    @test translate_callable((A, w) -> 2im * w, "dispersion_factor")[1] == "2*i*w"
    @test translate_callable((A, w) -> (1.0 + 2.0im) * w, "dispersion_factor")[1] ==
          "(1+2*i)*w"
end

@testset "named function binds by position" begin
    expr, constants = translate_callable(dispersion_fixture, "dispersion_factor")
    @test expr == "i*0.02*w^2"
    @test isempty(constants)
end

@testset "binds by name then position" begin
    # A parameter whose name is one of the context's symbols binds to that
    # symbol wherever it sits, rather than binding backwards.
    @test translate_callable((w, A) -> 0.5 * w, "dispersion_factor")[1] == "0.5*w"
    @test translate_callable((A, V, I) -> A * (I + V), "nonlinear")[1] == "A*(I+V)"
    # An unrecognised name takes the next unused symbol in the context's order.
    @test translate_callable((foo, bar) -> 0.5 * bar, "dispersion_factor")[1] == "0.5*w"
end

@testset "scalar result" begin
    # A function that ignores its arguments never touches a placeholder.
    @test translate_callable((A, w) -> 0.0, "dispersion_factor")[1] == "0"
end

@testset "precedence matches the C grammar" begin
    # ^ is right-associative in the compiler, left-associative nowhere that
    # matters here, but (a^b)^c must keep its parentheses either way.
    @test translate_callable((A, w) -> (w^2)^3, "dispersion_factor")[1] == "(w^2)^3"
    # Unary minus binds tighter than ^ in the compiler, which reads -w^2 as
    # (-w)^2.
    @test translate_callable((A, w) -> -w^2, "dispersion_factor")[1] == "-(w^2)"
    # The right operand of / needs atom-or-unary level.
    @test translate_callable((A, w) -> w / (w * w), "dispersion_factor")[1] == "w/(w*w)"
    @test translate_callable((A, w) -> w - (w + 2), "dispersion_factor")[1] == "w-(w+2)"
    # A negative literal is unary minus applied to a magnitude, not an atom.
    @test translate_callable((A, w) -> w^(-2), "dispersion_factor")[1] == "w^(-2)"
end

@testset "all six contexts" begin
    @test translate_callable((A, wt, kx, ky) -> im * (0.02 * wt^2 - 0.01 * (kx^2 + ky^2)),
                             "linear_factor")[1] == "i*(0.02*wt^2-0.01*(kx^2+ky^2))"
    @test translate_callable((A, D, h) -> exp(h * D), "linear")[1] == "exp(h*D)"
    @test translate_callable((x, y) -> 0.0002 * (x^2 + y^2), "potential")[1] ==
          "0.00020000000000000001*(x^2+y^2)"
    @test translate_callable((A, w) -> im * (0.04 * w^2), "dispersion_factor")[1] ==
          "i*0.040000000000000001*w^2"
    @test translate_callable((A, D, h, w) -> exp(h * D) * w, "dispersion")[1] ==
          "exp(h*D)*w"
    @test translate_callable((A, I, V) -> im * A * (0.015 * I + V), "nonlinear")[1] ==
          "i*A*(0.014999999999999999*I+V)"
end

@testset "curated failures" begin
    # Anything the operator VM cannot express fails naming the operation,
    # rather than reaching the C compiler as an unparseable string.
    @test_throws ArgumentError translate_callable((A, w) -> abs(w), "dispersion_factor")
    @test_throws ArgumentError translate_callable((A, w) -> im * (w > 0), "dispersion_factor")
    @test_throws ArgumentError translate_callable((A, w) -> tanh(w), "dispersion_factor")
    @test_throws ArgumentError translate_callable((A, w) -> conj(w), "dispersion_factor")
    @test_throws ArgumentError translate_callable((A, w) -> real(w), "dispersion_factor")
    # A non-scalar operand cannot be traced.
    @test_throws ArgumentError translate_callable((A, w) -> [1, 2] .* w, "dispersion_factor")
    # Inf and NaN have no representation in the expression language.
    @test_throws ArgumentError translate_callable((A, w) -> Inf * w, "dispersion_factor")
    @test_throws ArgumentError translate_callable((A, w) -> w, "not_a_context")
    @test_throws ArgumentError translate_callable((a, b, c, d) -> a, "nonlinear")

    # The message names the operation and what is available.
    err = try
        translate_callable((A, w) -> abs(w), "dispersion_factor")
        nothing
    catch exc
        exc
    end
    @test occursin("abs", err.msg)
    @test occursin("exp log sqrt sin cos", err.msg)
end

@testset "constant_bindings spend slots" begin
    beta2 = 0.04
    expr, constants = translate_callable((A, w) -> im * (beta2 * w^2),
                                         "dispersion_factor";
                                         constant_bindings = Dict("beta2" => 0.04))
    @test expr == "i*c0*w^2"
    @test constants == [0.04]

    # A binding whose value never appears as its own literal does not spend one.
    expr2, constants2 = translate_callable((A, w) -> im * (beta2 * w^2),
                                           "dispersion_factor";
                                           constant_bindings = Dict("nowhere" => 9.75))
    @test isempty(constants2)
    @test occursin(@sprintf("%.17g", beta2), expr2)
end

@testset "context symbols" begin
    @test runtime_context_symbols("nonlinear") == ["A", "I", "V"]
    @test runtime_context_symbols("potential") == ["x", "y", "t", "wt", "kx", "ky"]
    @test_throws ArgumentError runtime_context_symbols("nope")
end

@testset "all six runtime slots are wired" begin
    # Each function traces to the expected text, and RuntimeOperators takes it
    # in the matching slot.
    for (field, fn, context, expected) in (
        (:linear_factor_fn, (A, wt) -> im * (0.02 * wt^2), "linear_factor",
         "i*0.02*wt^2"),
        (:dispersion_factor_fn, (A, w) -> im * (0.02 * w^2), "dispersion_factor",
         "i*0.02*w^2"),
        (:linear_fn, (A, D, h) -> exp(h * D), "linear", "exp(h*D)"),
        (:dispersion_fn, (A, D, h, w) -> exp(h * D), "dispersion", "exp(h*D)"),
        (:potential_fn, (x, y) -> 0.0002 * (x^2 + y^2), "potential",
         "0.00020000000000000001*(x^2+y^2)"),
        (:nonlinear_fn, (A, I, V) -> im * A * I, "nonlinear", "i*A*I"),
    )
        traced, _ = translate_callable(fn, context)
        @test traced == expected
        # The field exists on RuntimeOperators and takes a function.
        operators = RuntimeOperators(; (field => fn,)...)
        @test getfield(operators, field) === fn
    end
end

@testset "aliased slots are rejected" begin
    duplicated = RuntimeOperators(
        linear_factor_fn = (A, wt) -> im * wt,
        dispersion_factor_fn = (A, wt) -> im * wt,
        linear_fn = (A, D, h) -> exp(h * D),
        nonlinear_fn = (A, I) -> 0.0,
    )
    @test_throws ArgumentError NLOLib.prepare_sim_config(4;
        propagation_distance = 0.001,
        starting_step_size = 1e-4,
        max_step_size = 1e-3,
        min_step_size = 1e-6,
        error_tolerance = 1e-7,
        delta_time = 0.02,
        pulse_period = 0.08,
        frequency_grid = zeros(ComplexF64, 4),
        runtime = duplicated,
    )
end

@testset "string operators are rejected" begin
    # Operators are functions only; the string-expression surface is gone.
    for field in (:linear_factor_expr, :linear_expr, :potential_expr,
                  :dispersion_factor_expr, :dispersion_expr, :nonlinear_expr)
        @test !hasfield(RuntimeOperators, field)
        @test_throws MethodError RuntimeOperators(; (field => "0",)...)
    end
    @test !hasfield(OperatorSpec, :expr)
    @test_throws ArgumentError RuntimeOperators(nonlinear_fn = "i*A*I") |>
        runtime -> NLOLib.prepare_sim_config(4;
            propagation_distance = 0.001,
            starting_step_size = 1e-4,
            max_step_size = 1e-3,
            min_step_size = 1e-6,
            error_tolerance = 1e-7,
            delta_time = 0.02,
            pulse_period = 0.08,
            frequency_grid = zeros(ComplexF64, 4),
            runtime = runtime,
        )
    @test_throws MethodError physics_config(nonlinear_expr = "i*A*I")
    @test_throws ArgumentError physics_config(nonlinear_fn = "i*A*I")
    pulse = PulseSpec(samples = ones(ComplexF64, 4), delta_time = 0.02)
    for operator in ("gvd", "i*w*w", OperatorSpec(fn = "i*w*w"), (fn = "i*w*w",))
        @test_throws ArgumentError NLOLib.from_pulse(NLOLib.PropagateRequestBuilder(), pulse,
                                                     operator, (A, I) -> im * A * I;
                                                     propagation_distance = 0.1)
    end
    @test_throws ArgumentError NLOLib.from_pulse(NLOLib.PropagateRequestBuilder(), pulse,
                                                 (A, w) -> im * w^2, "kerr";
                                                 propagation_distance = 0.1)
    err = try
        NLOLib.from_pulse(NLOLib.PropagateRequestBuilder(), pulse, "gvd", "kerr";
                          propagation_distance = 0.1)
        nothing
    catch exc
        exc
    end
    @test err isa ArgumentError
    @test occursin("must be a function", err.msg)
end

end
