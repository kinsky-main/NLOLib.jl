# Developing the Julia Package

This repository holds the Julia package, its tests, its examples and its
release pipeline. The native library it wraps is built from the core
repository, [nlo-rk4ip-code](https://github.com/kinsky-main/nlo-rk4ip-code).

## Layout

| Path | Contents |
| --- | --- |
| `src/`, `Project.toml` | the package; `Project.toml` is the source of truth for the version |
| `test/` | `runtests.jl`, plus the CTest registration in `test/CMakeLists.txt` |
| `examples/` | a Julia project of example scripts and their `backend/` helpers |
| `tools/` | version check and core checkout helpers |
| `CORE_REF` | the core commit releases and CI are built against |
| `CHANGELOG.md` | one entry per release |

## Two ways to work

**As a submodule of the core repository.** Clone the core repository with
`--recurse-submodules`; this repository is its `julia/` directory.

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release --target julia_stage
ctest --test-dir build -C Release -R "^test_julia_"
```

**Standalone.** Clone this repository alone and fetch the pinned core revision
into `./core`:

```bash
bash tools/fetch_core.sh
cmake -S core -B build -DBUILD_TESTING=ON -DNLOLIB_JULIA_DIR="$PWD"
cmake --build build --config Release --target julia_stage
ctest --test-dir build -C Release -R "^test_julia_"
```

`NLOLIB_JULIA_DIR` points the core build at this checkout. `julia_stage` writes
a self-contained package, with the library, to `build/julia_package`:

```julia
using Pkg
Pkg.activate("build/julia_package")
Pkg.instantiate()
using NonlinearOptics
NonlinearOptics.load("build/julia_package/lib/libnlolib.so")
```

`load()` opens `nlolib_jll`'s library and `load(path)` the given file. To run
this checkout, or anything else, against a built library by default, set the
`libnlolib_path` preference of `nlolib_jll` and restart Julia:

```julia
using Preferences, NonlinearOptics
set_preferences!(NonlinearOptics.nlolib_jll, "libnlolib_path" => "/path/to/libnlolib.so")
```

The CTest cases pass the library they just built to `load(path)`, so they
ignore both `nlolib_jll`'s library and the preference.

## The core pin

`CORE_REF` holds one full commit SHA of the core repository. CI checks out exactly
that commit, so every build of a given revision here uses the same native
library. To move to a newer core, replace the SHA in a pull request.
Inside the core repository the pin is not used — the submodule builds against
the tree it is mounted in.

## Versioning and release

The package version in `Project.toml` is independent of the core version. See
[release.md](release.md): releases go out through Registrator, and TagBot creates
the tag and GitHub release afterwards. Do not push version tags by hand.
