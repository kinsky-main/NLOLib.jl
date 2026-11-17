const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _LIB_PATH = Ref{String}("")

function _library_filename()
    if Sys.iswindows()
        return "nlolib.dll"
    elseif Sys.isapple()
        return "libnlolib.dylib"
    end
    return "libnlolib.so"
end

function _package_root()
    return normpath(joinpath(@__DIR__, ".."))
end

"""
    _artifact_library_paths()

Locate the shared library inside the installed `nlolib` artifact, if the
package has one bound for this platform and it is present in the depot.

Returns an empty vector for a plain source checkout, for a platform the
`Artifacts.toml` does not cover, and for an artifact that has not been
downloaded, so the development paths below stay reachable.
"""
function _artifact_library_paths()
    artifacts_toml = joinpath(_package_root(), "Artifacts.toml")
    isfile(artifacts_toml) || return String[]

    hash = try
        Artifacts.artifact_hash("nlolib", artifacts_toml)
    catch
        nothing
    end
    (hash === nothing || !Artifacts.artifact_exists(hash)) && return String[]

    root = Artifacts.artifact_path(hash)
    filename = _library_filename()
    return [joinpath(root, "lib", filename), joinpath(root, "bin", filename)]
end

function _candidate_library_paths(path::Union{Nothing, AbstractString})
    filename = _library_filename()
    package_root = _package_root()
    repo_root = normpath(joinpath(package_root, ".."))
    candidates = String[]

    # An explicit path is the only candidate, so a wrong one fails rather than
    # quietly loading a different library.
    path === nothing || return filter(isfile, [normpath(String(path))])

    env_path = get(ENV, "NLOLIB_LIBRARY", "")
    if !isempty(env_path)
        push!(candidates, normpath(env_path))
    end

    # The artifact is the installed distribution, so it wins over the build
    # tree; the paths after it keep a built repo checkout working unchanged.
    append!(candidates, _artifact_library_paths())

    append!(candidates, [
        joinpath(package_root, "lib", filename),
        joinpath(repo_root, "python", filename),
        joinpath(repo_root, "python", "Debug", filename),
        joinpath(repo_root, "python", "Release", filename),
        joinpath(repo_root, "build", "julia_package", "lib", filename)
    ])

    return unique(filter(isfile, candidates))
end

"""
    load(path = nothing)

Load the nlolib shared library: from `path` when given, and only from there;
otherwise from `NLOLIB_LIBRARY`, the package artifact or a built checkout, in
that order.
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
        "cannot locate the nlolib shared library; install NLOLib so its artifact is available, " *
        "set NLOLIB_LIBRARY, or build the julia_stage target"))

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
