# Tests for the function-operator tracer. These exercise translation only, so
# they need no built shared library.

using Test
using Printf
using NonlinearOptics
using NonlinearOptics: translate_callable, runtime_context_symbols

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

@testset "parameter binding rule" begin
    # operator_lowering.parameter_binding in the core's tests/api_surface.json.
    # (a) A parameter named after a context symbol binds to it wherever it is.
    @test translate_callable((w, A) -> 0.5 * w, "dispersion_factor")[1] == "0.5*w"
    @test translate_callable((A, V, I) -> A * (I + V), "nonlinear")[1] == "A*(I+V)"
    # (a) before (b): named parameters bind first, so (B, A) binds A to A and
    # B to the next unused symbol, w.
    @test translate_callable((B, A) -> B * A, "dispersion_factor")[1] == "w*A"
    @test translate_callable((B, I) -> B * I, "nonlinear")[1] == "A*I"
    @test translate_callable((field, D, step) -> field * step * D, "dispersion")[1] == "A*h*D"
    # (b) An unrecognised name takes the next unused symbol in the context's order.
    @test translate_callable((foo, bar) -> 0.5 * bar, "dispersion_factor")[1] == "0.5*w"
    @test translate_callable((A, omega) -> omega, "dispersion_factor")[1] == "w"
    @test translate_callable((A, omega) -> omega, "linear_factor")[1] == "wt"
    # (b) More parameters than symbols is the error.
    @test_throws ArgumentError translate_callable((a, b, c) -> a, "dispersion_factor")
    # (c) A runtime symbol the context does not provide is never rebound.
    @test_throws ArgumentError translate_callable((A, kx) -> kx, "dispersion_factor")
    @test_throws ArgumentError translate_callable((A, I) -> I, "linear_factor")
    # (d) Two parameters resolving to one symbol are the error. Julia rejects a
    # repeated parameter name itself, so only an alias pair can reach the rule.
    @test_throws ArgumentError translate_callable((A, w, wt, kx, ky) -> w, "linear_factor")
    @test_throws ArgumentError translate_callable((A, wt, w) -> w, "dispersion_factor")
    # (e) w and wt are one symbol, emitted in the context's own spelling.
    @test translate_callable((A, w, kx, ky) -> w * kx * ky, "linear_factor")[1] == "wt*kx*ky"
    @test translate_callable((A, wt) -> wt, "dispersion_factor")[1] == "w"
    # (f) A free name in the body is a captured constant, whatever it is called.
    kx = 0.25
    @test translate_callable((A, w) -> kx * w, "dispersion_factor")[1] == "0.25*w"
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
    @test runtime_context_symbols("dispersion_factor") == ["A", "w"]
    @test runtime_context_symbols("dispersion") == ["A", "D", "h", "w", "V"]
    @test runtime_context_symbols("linear_factor") == ["A", "wt", "kx", "ky", "x", "y", "t"]
    @test runtime_context_symbols("linear") == ["A", "D", "h", "wt", "kx", "ky", "x", "y", "t", "V"]
    @test runtime_context_symbols("potential") == ["x", "y", "t"]
    @test runtime_context_symbols("nonlinear") == ["A", "I", "V"]
    @test_throws ArgumentError runtime_context_symbols("nope")
end

@testset "w and wt name the same axis" begin
    @test translate_callable((A, wt) -> im * wt^2, "dispersion_factor")[1] == "i*w^2"
    @test translate_callable((A, D, h, wt) -> exp(h * D) * wt, "dispersion")[1] == "exp(h*D)*w"
    @test translate_callable((A, w) -> im * w^2, "linear_factor")[1] == "i*wt^2"
    @test translate_callable((A, D, h, w) -> exp(h * D) * w, "linear")[1] == "exp(h*D)*wt"
end

@testset "symbols a context does not provide" begin
    # The C parser rejects these, so the tracer does too, before the C call.
    @test_throws ArgumentError translate_callable((A, kx) -> kx, "dispersion_factor")
    @test_throws ArgumentError translate_callable((A, w, t) -> w * t, "dispersion_factor")
    @test_throws ArgumentError translate_callable((A, D, h, x) -> x, "dispersion")
    @test_throws ArgumentError translate_callable((x, wt) -> x * wt, "potential")
    @test_throws ArgumentError translate_callable((x, kx) -> x * kx, "potential")
    @test_throws ArgumentError translate_callable((A, I, x) -> x, "nonlinear")
    @test_throws ArgumentError translate_callable((A, I) -> I, "dispersion_factor")
    err = try
        translate_callable((A, kx) -> kx, "dispersion_factor")
        nothing
    catch exc
        exc
    end
    @test occursin("does not provide 'kx'", err.msg)
end

@testset "operators that are not functions are rejected" begin
    pulse = PulseSpec(samples = ones(ComplexF64, 4), delta_time = 0.02)
    linear = (A, w) -> im * w^2
    nonlinear = (A, I) -> im * A * I
    for operator in ("gvd", "i*w*w", (fn = linear,), nothing, 1.0)
        @test_throws ArgumentError propagate(pulse, operator, nonlinear; propagation_distance = 0.1)
        @test_throws ArgumentError propagate(pulse, linear, operator; propagation_distance = 0.1)
    end
    for name in (:linear_step_operator, :potential_operator)
        @test_throws ArgumentError propagate(pulse, linear, nonlinear;
                                             propagation_distance = 0.1, (name => "exp(h*D)",)...)
    end
    err = try
        propagate(pulse, "gvd", "kerr"; propagation_distance = 0.1)
        nothing
    catch exc
        exc
    end
    @test err isa ArgumentError
    @test occursin("linear_operator must be a function", err.msg)
    # A symbol the run type's context does not provide fails before the C call.
    @test_throws ArgumentError propagate(pulse, (A, kx) -> kx, nonlinear; propagation_distance = 0.1)
end

end
