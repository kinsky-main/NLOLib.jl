const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _LIB_PATH = Ref{String}("")

const _PREFERENCE_HINT =
    "To use a library of your own, set the nlolib_jll preference and restart Julia:\n" *
    "    using Preferences, NonlinearOptics\n" *
    "    set_preferences!(NonlinearOptics.nlolib_jll, \"libnlolib_path\" => \"/path/to/libnlolib.$(Libdl.dlext)\")\n" *
    "or pass the file to load(path)."

_missing_library_error() = NLOLibError(nothing,
    "nlolib_jll has no nlolib build for $(Sys.MACHINE). " * _PREFERENCE_HINT)

# The library `load()` opens: `nlolib_jll.libnlolib_path`, which JLLWrappers
# takes from the `libnlolib_path` preference when it is set. A platform without
# a build never reads the preference, so it is read here instead.
function _jll_library_path()
    nlolib_jll.is_available() && return nlolib_jll.libnlolib_path
    return Preferences.load_preference(nlolib_jll, "libnlolib_path", nothing)
end

"""
    load(path = nothing)

Load the nlolib shared library: the file at `path` when given, otherwise the
library of `nlolib_jll`. Every other function loads the library on first use,
so calling `load` is only needed to pick a specific file.

To use a local build everywhere, including on first use, set the
`libnlolib_path` preference of `nlolib_jll` (read when `nlolib_jll`
precompiles, so restart Julia after changing it):

```julia
using Preferences, NonlinearOptics
set_preferences!(NonlinearOptics.nlolib_jll, "libnlolib_path" => "/path/to/libnlolib.so")
```

Throws `NLOLibError` when the library cannot be loaded; a failed `load(path)`
keeps the library already loaded.
"""
function load(path = nothing)
    path === nothing || path isa AbstractString ||
        throw(ArgumentError("load: path must be a string or nothing"))
    current = _LIB_HANDLE[]
    if current != C_NULL && (path === nothing || normpath(String(path)) == _LIB_PATH[])
        return nothing
    end

    if path === nothing
        candidate = _jll_library_path()
        candidate === nothing && throw(_missing_library_error())
    else
        candidate = normpath(String(path))
        isfile(candidate) || throw(NLOLibError(nothing, "no nlolib shared library at $(candidate)"))
    end

    handle = Libdl.dlopen_e(candidate)
    handle == C_NULL && throw(NLOLibError(nothing,
        "failed to load the nlolib shared library from $(candidate)" *
        (path === nothing ? ". " * _PREFERENCE_HINT : "")))

    current == C_NULL || Libdl.dlclose(current)
    _LIB_HANDLE[] = handle
    _LIB_PATH[] = candidate
    return nothing
end

"""
    loaded_library_path()

Path of the loaded nlolib shared library, or `""` before it is loaded.
"""
loaded_library_path() = _LIB_PATH[]

function _handle()
    _LIB_HANDLE[] == C_NULL && load()
    return _LIB_HANDLE[]
end
_sym(name::Symbol) = Libdl.dlsym(_handle(), name)
