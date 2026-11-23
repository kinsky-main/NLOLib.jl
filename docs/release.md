# Julia Release Guide

`NonlinearOptics` is distributed as a registered Julia package whose native library comes
from a [Julia artifact](https://pkgdocs.julialang.org/v1/artifacts/). The
package itself is pure Julia; the per-platform tarballs holding `nlolib` and its
runtime dependencies are built by
[`.github/workflows/release-julia.yml`](../.github/workflows/release-julia.yml)
and attached to this repository's GitHub release for the tag. The library is
built from the core repository at the commit `CORE_REF` pins.

The Vulkan + VkFFT + statically linked SQLite toolchain is why the binary is
built here rather than through Yggdrasil/BinaryBuilder: the release matrix
already builds all of it on the images we support.

## Installing

Once the package is registered:

```julia
using Pkg
Pkg.add("NonlinearOptics")
```

`Pkg` downloads the artifact for the host platform, so nothing needs to be
compiled and no Vulkan SDK is required. On Linux the Vulkan **loader**
(`libvulkan.so.1`, from `libvulkan1` or a GPU driver package) must be present,
because it is a normal shared-library dependency and deliberately not bundled —
it belongs to the system. Windows delay-loads `vulkan-1.dll`, so it is optional
there. Without a Vulkan driver the CPU backend runs unchanged.

### From a build tree

A built checkout keeps working with no artifact at all; see
[development.md](development.md) for the build itself.

```julia
using Pkg
Pkg.activate("build/julia_package")
Pkg.instantiate()
using NonlinearOptics
NonlinearOptics.load("build/julia_package/lib/libnlolib.so")
```

`src/lib.jl` opens one library: the path passed to `NonlinearOptics.load(path)`,
otherwise `nlolib_jll.libnlolib_path`. To make a local build the default, set
the `libnlolib_path` preference of `nlolib_jll`; JLLWrappers reads it when
`nlolib_jll` precompiles, so restart Julia after changing it:

```julia
using Preferences, NonlinearOptics
set_preferences!(NonlinearOptics.nlolib_jll, "libnlolib_path" => "/path/to/libnlolib.so")
```

## Releasing

A release is a registration. Registrator records the version in the General
registry, and TagBot then creates the tag and the GitHub release on the commit
that was registered. **Nobody pushes version tags by hand**: a hand-pushed tag
can point at a different commit than the registered one.

### 1. Prepare the version

The package version lives in `Project.toml` and is independent of the core
library version; `CORE_REF` names the core commit the release is built
against. Nothing advances the version automatically.

In a pull request, set the new version in `Project.toml`, add its entry to
[`CHANGELOG.md`](../CHANGELOG.md), and, if the release needs a newer core, move
`CORE_REF` (and the `nlolib_jll` compat bound when the JLL changes). Merge it.

### 2. Register

1. Install the [JuliaRegistrator](https://github.com/JuliaRegistries/Registrator)
   GitHub App on `kinsky-main/NonlinearOptics.jl`.
2. Comment on the merge commit:

   ```
   @JuliaRegistrator register

   Release notes:

   <the CHANGELOG.md entry for this version>
   ```

   No `subdir` is needed: the package is at the repository root. The release
   notes become the GitHub release text.
3. Registrator opens a pull request against `JuliaRegistries/General`. Watch its
   AutoMerge checks and respond there.

### 3. Tag (automatic)

[`.github/workflows/TagBot.yml`](../.github/workflows/TagBot.yml) runs when
JuliaTagBot comments after the registry pull request merges. It tags the
registered commit `v<version>` (the first release is `v0.1.0`) and creates the
GitHub release from the Registrator release notes, a link to `CHANGELOG.md`,
and the merged pull requests and closed issues since the previous release. If it
did not run, start it from the Actions tab with `workflow_dispatch`.

TagBot pushes the tag with the `DOCUMENTER_KEY` deploy key rather than
`GITHUB_TOKEN`, so the tag push triggers the documentation deploy. That key is
also what lets it push a tag on a commit that changes a workflow file, which
`GITHUB_TOKEN` is not allowed to do.

The bare `1.20.1` and `2.0.0` tags belong to the unregistered `NLOLib` line and
do not clash with TagBot's `v`-prefixed tags.

#### What AutoMerge will say

- AutoMerge requires a compat entry for every dependency and an upper bound on
  each non-stdlib one; `nlolib_jll = "~0.1.0"` is bounded.
- A new package waits a mandatory three days before it is merged, however
  clean its checks are.
- A later version must follow the previous registered one: only the next
  patch, minor or major auto-merges. Skipping a version leaves the pull
  request for a General maintainer to merge by hand.
- Any comment on the registry pull request blocks the auto-merge unless it
  contains `[noblock]`.
- The repository is public and GPL-2.0 licensed with a `LICENSE` file at the
  root, which satisfies the license check.
- AutoMerge installs and imports the package on its own runners. `NonlinearOptics` does
  not open the library at precompile time — `load()` is explicit — so the import
  succeeds on platforms `nlolib_jll` does not cover.

#### Version numbering

The version is frozen between releases and advances only when someone edits
`Project.toml`. A registry version can never be reused and must increase
monotonically.

## Files

- `.github/workflows/TagBot.yml` — tags and releases registered versions
- `CHANGELOG.md` — one entry per release
- `CORE_REF` — the core commit the release is built against
