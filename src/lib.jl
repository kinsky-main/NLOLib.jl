const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _LIB_PATH = Ref{String}("")

function _candidate_library_paths(path::Union{Nothing, AbstractString})
    # An explicit path is the only candidate, so a wrong one fails rather than
    # quietly loading a different library.
    path === nothing || return filter(isfile, [normpath(String(path))])

    candidates = String[]
    env_path = get(ENV, "NLOLIB_LIBRARY", "")
    isempty(env_path) || push!(candidates, normpath(env_path))
    nlolib_jll.is_available() && push!(candidates, nlolib_jll.libnlolib_path)
    return unique(filter(isfile, candidates))
end

"""
    load(path = nothing)

Load the nlolib shared library: from `path` when given, and only from there;
otherwise from `NLOLIB_LIBRARY` when set, then from `nlolib_jll`.
Every other function loads the library on first use, so calling `load` is only
needed to pick a specific file. Throws `NLOLibError` when no library loads.
"""
function load(path = nothing)
    path === nothing || path isa AbstractString ||
        throw(ArgumentError("load: path must be a string or nothing"))
    current = _LIB_HANDLE[]
    if current != C_NULL && (path === nothing || normpath(String(path)) == _LIB_PATH[])
        return nothing
    end

    candidates = _candidate_library_paths(path)
    isempty(candidates) && throw(NLOLibError(nothing, path !== nothing ?
        "no nlolib shared library at $(path)" :
        "cannot locate the nlolib shared library: nlolib_jll has no build for " *
        "$(Sys.MACHINE); set NLOLIB_LIBRARY or pass a path to load()"))

    if current != C_NULL
        Libdl.dlclose(current)
        _LIB_HANDLE[] = C_NULL
        _LIB_PATH[] = ""
    end

    for candidate in candidates
        handle = Libdl.dlopen_e(candidate)
        if handle != C_NULL
            _LIB_HANDLE[] = handle
            _LIB_PATH[] = candidate
            return nothing
        end
    end

    throw(NLOLibError(nothing, "failed to load the nlolib shared library from: " * join(candidates, ", ")))
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
