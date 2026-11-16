"""
Callable-to-runtime-expression translation.

A user function is called once, at setup, with `RuntimeExpr` placeholders bound
to the context's runtime symbols. Julia evaluates the body, so captured values,
field access, indexing and arithmetic on constants all collapse to numbers
before the tracer sees them, and Julia's own precedence builds the tree. The
result is expression text for the compiled operator VM, which applies the
operator without returning to Julia.

The function is never called again. A per-application numeric callback is not
possible for these operators: `src/numerics/rk4_kernel.c` records one command
buffer for the whole propagation and applies the operators 8-10 times per step,
so crossing back into Julia would mean a submit, fence wait and download around
each application.
"""

# Precedence levels of the runtime operator grammar, as
# src/physics/operator_program_compile.c parses it:
# parse_expression -> parse_term -> parse_power -> parse_unary -> parse_primary.
const PREC_ADD = 0    # + -
const PREC_MUL = 1    # * /
const PREC_UNARY = 2  # unary minus
const PREC_POW = 3    # ^
const PREC_ATOM = 4   # symbol, literal, call, parenthesised group

"""
    RuntimeExpr(text, prec = PREC_ATOM)

Placeholder that records a runtime operator expression as text plus the
precedence of its outermost operator.

The operator set is exactly what the compiled VM implements: `+ - * / ^`, unary
minus, and `exp log sqrt sin cos`. Anything else has no method and fails with
an `ArgumentError` naming the operation. Broadcasting over scalars dispatches to
the same methods, so `.*` and `.^` work without extra definitions.

Three places where Julia's grammar disagrees with the C one, and the printer
adds parentheses the source did not have:

  * `^` is right-associative in the compiler (`parse_power` recurses into
    itself) and so is Julia's, but both operands still print at atom level so
    that `(a^b)^c` keeps its parentheses.
  * unary minus binds tighter than `^` in the compiler, because `parse_power`
    takes `parse_unary` as its left operand: C reads `-a^b` as `(-a)^b`, Julia
    as `-(a^b)`. Printing the operand of unary minus at atom level makes that
    `-(a^b)`.
  * the right operand of `/` needs atom-or-unary level, or `a/(b*c)` reprints
    as `a/b*c`.
"""
struct RuntimeExpr
    text::String
    prec::Int
end

RuntimeExpr(text::AbstractString) = RuntimeExpr(String(text), PREC_ATOM)

Base.show(io::IO, expr::RuntimeExpr) = print(io, "RuntimeExpr(", expr.text, ")")
Base.String(expr::RuntimeExpr) = expr.text

"""
    _operand(expr, min_prec)

Render `expr`, parenthesised when it binds looser than `min_prec`.
"""
_operand(expr::RuntimeExpr, min_prec::Integer) =
    expr.prec < min_prec ? string("(", expr.text, ")") : expr.text

"""
    _right_operand(expr, min_prec)

`_operand` for a right-hand operand. Also parenthesises a leading minus sign,
so `a-(-b)` does not print as the valid-but-unreadable `a--b`.
"""
function _right_operand(expr::RuntimeExpr, min_prec::Integer)
    if expr.prec < min_prec || startswith(expr.text, "-")
        return string("(", expr.text, ")")
    end
    return expr.text
end

"""
    _format(value)

Shortest exact decimal form of a `Float64`, matching the `%.17g` the Python and
MATLAB translators use so the three bindings emit the same text.
"""
_format(value::Real) = @sprintf("%.17g", Float64(value))

"""
    _literal(value)

Inline a numeric scalar as runtime expression text. The VM has `TOKEN_NUMBER`,
so a captured number needs no constant slot. `im` becomes `i`, `-im` becomes
`-i`, and a general `a+bi` becomes `a+b*i`.
"""
function _literal(value::Number)
    re = real(value)
    im_part = imag(value)
    if !isfinite(re) || !isfinite(im_part)
        throw(ArgumentError(
            "runtime operator constants must be finite; got $(value). " *
            "Inf and NaN have no representation in the operator expression language."
        ))
    end

    if im_part == 0
        return _real_literal(re)
    end

    if re == 0
        im_part == 1 && return RuntimeExpr("i", PREC_ATOM)
        im_part == -1 && return RuntimeExpr("-i", PREC_UNARY)
        # A product: 0.5im prints as 0.5*i, tagged PREC_MUL so a surrounding ^
        # or / parenthesises it.
        return RuntimeExpr(string(_format(im_part), "*i"), PREC_MUL)
    end

    text = im_part > 0 ?
        string(_format(re), "+", _format(im_part), "*i") :
        string(_format(re), "-", _format(-im_part), "*i")
    return RuntimeExpr(text, PREC_ADD)
end

"""
    _real_literal(value)

Inline a real scalar, tagging a negative one as unary rather than atom: `w^-2`
is not valid in the C grammar, `w^(-2)` is.
"""
_real_literal(value::Real) =
    RuntimeExpr(_format(value), value < 0 ? PREC_UNARY : PREC_ATOM)

"""
    _coerce(value, op)

Return `value` as a `RuntimeExpr`, inlining numeric scalars.
"""
_coerce(expr::RuntimeExpr, op) = expr
_coerce(value::Number, op) = _literal(value)
_coerce(value, op) = throw(ArgumentError(
    "$(op) received a $(typeof(value)). A runtime operator handle combines the " *
    "context's symbols with real or complex numeric scalars only; arrays and " *
    "other values cannot be traced. Reduce the value to a scalar before the " *
    "function, or pass a precomputed grid array instead."
))

# -- arithmetic the operator VM implements ---------------------------------
#
# Each method is defined for the three operand combinations rather than on a
# Union, so a RuntimeExpr never falls back to a generic Number method.

for (op, name) in ((:+, "+"), (:-, "-"), (:*, "*"), (:/, "/"), (:^, "^"))
    @eval begin
        Base.$op(a::RuntimeExpr, b::RuntimeExpr) = _binary($(QuoteNode(op)), a, b)
        Base.$op(a::RuntimeExpr, b::Number) = _binary($(QuoteNode(op)), a, _coerce(b, $name))
        Base.$op(a::Number, b::RuntimeExpr) = _binary($(QuoteNode(op)), _coerce(a, $name), b)
        # Anything that is not a number reaches _coerce only to be rejected, so
        # the failure names the operation instead of surfacing as a MethodError.
        Base.$op(::RuntimeExpr, b::Any) = _coerce(b, $name)
        Base.$op(a::Any, ::RuntimeExpr) = _coerce(a, $name)
    end
end

# Declare RuntimeExpr a scalar for broadcasting, so w .^ 2 and im .* w dispatch
# to the methods above. Without this, broadcast asks the placeholder for its
# length. The result of broadcasting over scalars alone is a zero-dimensional
# array, which _unwrap turns back into the expression it holds.
Base.broadcastable(expr::RuntimeExpr) = Ref(expr)

function _binary(op::Symbol, a::RuntimeExpr, b::RuntimeExpr)
    if op === :+
        return RuntimeExpr(
            string(_operand(a, PREC_ADD), "+", _right_operand(b, PREC_ADD)), PREC_ADD)
    elseif op === :-
        return RuntimeExpr(
            string(_operand(a, PREC_ADD), "-", _right_operand(b, PREC_MUL)), PREC_ADD)
    elseif op === :*
        return RuntimeExpr(
            string(_operand(a, PREC_MUL), "*", _right_operand(b, PREC_MUL)), PREC_MUL)
    elseif op === :/
        return RuntimeExpr(
            string(_operand(a, PREC_MUL), "/", _right_operand(b, PREC_UNARY)), PREC_MUL)
    elseif op === :^
        return RuntimeExpr(
            string(_operand(a, PREC_ATOM), "^", _operand(b, PREC_ATOM)), PREC_POW)
    end
    throw(ArgumentError("unsupported runtime operator $(op)"))
end

Base.:-(a::RuntimeExpr) =
    RuntimeExpr(string("-", _operand(a, PREC_ATOM)), PREC_UNARY)
Base.:+(a::RuntimeExpr) = a

# Integer literal powers go through Base.literal_pow, which would otherwise
# expand w^2 into w*w.
Base.literal_pow(::typeof(^), a::RuntimeExpr, ::Val{N}) where {N} =
    _binary(:^, a, _literal(N))

for (fn, name) in ((:exp, "exp"), (:log, "log"), (:sqrt, "sqrt"),
                   (:sin, "sin"), (:cos, "cos"))
    @eval Base.$fn(a::RuntimeExpr) =
        RuntimeExpr(string($name, "(", a.text, ")"), PREC_ATOM)
end

# -- everything the VM cannot express --------------------------------------
#
# These exist so the failure names the operation and what is available,
# instead of Julia's "MethodError: no method matching abs(::RuntimeExpr)".

const _UNSUPPORTED_UNARY = (:abs, :abs2, :conj, :real, :imag, :angle, :tan,
                            :sinh, :cosh, :tanh, :asin, :acos, :atan, :log2,
                            :log10, :floor, :ceil, :round, :trunc, :sign,
                            :sum, :prod, :cumsum, :inv, :adjoint, :transpose)

for fn in _UNSUPPORTED_UNARY
    @eval Base.$fn(::RuntimeExpr) = _unsupported_op($(string(fn)))
end

for fn in (:mod, :rem, :div, :max, :min, :atan, :hypot)
    @eval begin
        Base.$fn(::RuntimeExpr, ::RuntimeExpr) = _unsupported_op($(string(fn)))
        Base.$fn(::RuntimeExpr, ::Number) = _unsupported_op($(string(fn)))
        Base.$fn(::Number, ::RuntimeExpr) = _unsupported_op($(string(fn)))
    end
end

for fn in (:(==), :(!=), :<, :>, :<=, :>=)
    @eval begin
        Base.$fn(::RuntimeExpr, ::Number) = _unsupported_op($(string(fn)))
        Base.$fn(::Number, ::RuntimeExpr) = _unsupported_op($(string(fn)))
        Base.$fn(::RuntimeExpr, ::RuntimeExpr) = _unsupported_op($(string(fn)))
    end
end

_unsupported_op(op::AbstractString) = throw(ArgumentError(
    "$(op) is not available in a runtime operator function. The compiled " *
    "operator VM implements + - * / ^, unary minus, and exp log sqrt sin cos " *
    "over the context's symbols, numeric literals and i. Fold anything else " *
    "into a constant before the function, or precompute it on the grid and " *
    "pass it as an array."
))

# -- contexts ---------------------------------------------------------------

const RUNTIME_CONTEXT_LINEAR_FACTOR = "linear_factor"
const RUNTIME_CONTEXT_LINEAR = "linear"
const RUNTIME_CONTEXT_POTENTIAL = "potential"
const RUNTIME_CONTEXT_DISPERSION_FACTOR = "dispersion_factor"
const RUNTIME_CONTEXT_DISPERSION = "dispersion"
const RUNTIME_CONTEXT_NONLINEAR = "nonlinear"

# Positional default order per context, which is also the allow-list. Mirrors
# _CONTEXT_ALLOWED_SYMBOLS and _CONTEXT_DEFAULT_PARAM_SYMBOLS in
# python/nlolib/translate/runtime_expr.py and the context_allows_symbol_*
# checks in src/physics/operator_program_compile.c.
const _CONTEXT_SYMBOLS = Dict{String, Vector{String}}(
    RUNTIME_CONTEXT_LINEAR_FACTOR =>
        ["A", "wt", "kx", "ky", "x", "y", "t"],
    RUNTIME_CONTEXT_DISPERSION_FACTOR =>
        ["A", "w", "wt", "kx", "ky", "x", "y", "t"],
    RUNTIME_CONTEXT_LINEAR =>
        ["A", "D", "h", "wt", "kx", "ky", "x", "y", "t", "V"],
    RUNTIME_CONTEXT_DISPERSION =>
        ["A", "D", "h", "w", "wt", "kx", "ky", "x", "y", "t", "V"],
    RUNTIME_CONTEXT_POTENTIAL =>
        ["x", "y", "t", "wt", "kx", "ky"],
    RUNTIME_CONTEXT_NONLINEAR =>
        ["A", "I", "V"],
)

"""
    runtime_context_symbols(context)

The runtime symbols `context` accepts, in the order unnamed parameters bind.
"""
function runtime_context_symbols(context::AbstractString)
    symbols = get(_CONTEXT_SYMBOLS, String(context), nothing)
    symbols === nothing && throw(ArgumentError(
        "context must be one of 'linear_factor', 'linear', 'potential', " *
        "'dispersion_factor', 'dispersion' or 'nonlinear'; got '$(context)'"
    ))
    return symbols
end

"""
    _argument_names(fn, arity)

Parameter names of `fn`, or `nothing` when they cannot be read.

`Base.method_argnames` reports the names a method was written with, which is
what lets a parameter bind by name. It is not public API and a closure's
generated method may carry placeholder names, so anything unusable falls back
to positional binding.
"""
function _argument_names(fn, arity::Integer)
    try
        candidates = String[]
        for method in methods(fn)
            names = Base.method_argnames(method)
            # names[1] is the function itself.
            length(names) == arity + 1 || continue
            parsed = String[String(n) for n in names[2:end]]
            all(name -> occursin(r"^[A-Za-z_]\w*$", name), parsed) || continue
            # A closure whose parameters were erased shows up as #unused# or _.
            any(name -> startswith(name, "#") || name == "_", parsed) && continue
            isempty(candidates) || candidates == parsed || return nothing
            candidates = parsed
        end
        return isempty(candidates) ? nothing : candidates
    catch
        return nothing
    end
end

"""
    _bind_symbols(names, arity, context, symbols)

Map each parameter to the runtime symbol it stands for.

Name first, then position, so `(w, A) -> ...` binds `w` to `w` and `A` to `A`
rather than binding backwards the way a purely positional rule would. This is
the rule `translate_callable` in python/nlolib/translate/runtime_expr.py uses.
"""
function _bind_symbols(names, arity::Integer, context::AbstractString, symbols::Vector{String})
    bound = Vector{String}(undef, arity)
    used = Set{String}()
    default_idx = 1
    for idx in 1:arity
        name = names === nothing ? nothing : names[idx]
        if name !== nothing && name in symbols && !(name in used)
            symbol = name
        else
            while default_idx <= length(symbols) && symbols[default_idx] in used
                default_idx += 1
            end
            default_idx > length(symbols) && throw(ArgumentError(
                "argument $(idx) of this $(context) function has no symbol left " *
                "to bind; the context provides $(join(symbols, ", "))"
            ))
            symbol = symbols[default_idx]
            default_idx += 1
        end
        bound[idx] = symbol
        push!(used, symbol)
    end
    return bound
end

"""
    _arity(fn)

Number of positional parameters `fn` takes, for the single applicable method.
"""
function _arity(fn)
    arities = Set{Int}()
    for method in methods(fn)
        method.isva && throw(ArgumentError(
            "a runtime operator function cannot take varargs, so its symbols " *
            "cannot be determined; wrap it in a function with explicit " *
            "arguments, such as (A, w) -> f(A, w)"
        ))
        push!(arities, Int(method.nargs) - 1)
    end
    isempty(arities) && throw(ArgumentError(
        "a runtime operator function must have at least one method"
    ))
    length(arities) == 1 || throw(ArgumentError(
        "a runtime operator function must have a single arity; this one has " *
        "methods taking $(join(sort(collect(arities)), ", ")) arguments"
    ))
    return first(arities)
end

"""
    translate_callable(fn, context; constant_bindings = nothing)

Compile `fn` into runtime operator expression text, returning
`(expression, constants)`.

`fn` is traced once with `RuntimeExpr` placeholders bound to `context`'s
symbols, so ordinary Julia in the body -- `pi`, `p.beta2`, `beta[3]`,
`2pi/lambda` -- works. The contexts and their symbols are

    "linear_factor"      A, wt, kx, ky, x, y, t
    "dispersion_factor"  A, w, wt, kx, ky, x, y, t            (alias, adds w)
    "linear"             A, D, h, wt, kx, ky, x, y, t, V
    "dispersion"         A, D, h, w, wt, kx, ky, x, y, t, V    (alias, adds w)
    "potential"          x, y, t, wt, kx, ky
    "nonlinear"          A, I, V

`constants` is empty for a traced function: captured numbers are inlined as
literals, so `RUNTIME_OPERATOR_CONSTANTS_MAX` (16) does not limit how many one
may use. Pass `constant_bindings` as a `name => value` mapping to spend a slot
on a named value instead; each bound name whose value appears in the traced
expression as its own literal is rewritten to `cN` and returned in `constants`.
The substitution is by value, so the expression computes the same result either
way.

# Example
```julia
beta2 = 0.05
expr, _ = translate_callable((A, w) -> im * (beta2 / 2) * w^2, "dispersion_factor")
# expr == "0.025000000000000001*i*w^2"
```

Note that `w` is a symbol only in the `dispersion_factor` and `dispersion`
contexts; in `linear_factor` a parameter named `w` takes that context's next
symbol, `wt`.
"""
function translate_callable(fn, context::AbstractString; constant_bindings = nothing)
    symbols = runtime_context_symbols(context)
    arity = _arity(fn)
    if arity < 1 || arity > length(symbols)
        throw(ArgumentError(
            "a $(context) function must take between 1 and $(length(symbols)) " *
            "arguments; this one takes $(arity). The symbols it may bind are: " *
            "$(join(symbols, ", "))"
        ))
    end

    names = _argument_names(fn, arity)
    bound = _bind_symbols(names, arity, context, symbols)
    traced = [RuntimeExpr(symbol) for symbol in bound]

    result = _unwrap(fn(traced...))

    expression = if result isa RuntimeExpr
        result.text
    elseif result isa Number
        # A function that ignores its arguments, such as (A, w) -> 0.0, never
        # touches a placeholder and so returns a plain number.
        _literal(result).text
    else
        throw(ArgumentError(
            "a $(context) function must return a scalar expression built from " *
            "its arguments; this one returned a $(typeof(result)). Operators " *
            "are traced once at setup, so the body cannot branch on or index " *
            "the field."
        ))
    end

    return _apply_constant_bindings(expression, constant_bindings)
end

"""
    _apply_constant_bindings(expression, bindings)

Spend a constant slot on each bound named value.

Tracing inlines captured numbers, so there is no identifier left to bind by
name. A binding is applied by matching the exact text of its value, which is
the same `Float64` either way, so the expression's result does not change. A
name whose value was folded into a larger constant, as `beta2` is in
`beta2/2`, has no literal of its own to rewrite and is skipped.
"""
function _apply_constant_bindings(expression::AbstractString, bindings)
    constants = Float64[]
    bindings === nothing && return (String(expression), constants)

    text = String(expression)
    for (name, value) in bindings
        isfinite(value) || throw(ArgumentError(
            "constant binding '$(name)' must be a finite real scalar"
        ))
        literal = _format(Float64(value))
        pattern = Regex("(?<![\\w.])" * _escape_regex(literal) * "(?![\\w.])")
        occursin(pattern, text) || continue
        text = replace(text, pattern => "c$(length(constants))")
        push!(constants, Float64(value))
    end
    return (text, constants)
end

"""
    _unwrap(value)

Unwrap the zero-dimensional array that broadcasting over scalars alone
produces, so `(A, w) -> im .* w` traces the same as `(A, w) -> im * w`.
"""
_unwrap(value::AbstractArray{<:Any, 0}) = value[]
_unwrap(value) = value

# A %.17g literal holds only digits, '.', '-', '+' and 'e', so these two are
# the only characters a regex would read as syntax.
_escape_regex(text::AbstractString) =
    replace(text, "." => "\\.", "+" => "\\+")
