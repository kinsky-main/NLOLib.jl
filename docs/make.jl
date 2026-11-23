# Build the site locally into docs/build:
#
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs docs/make.jl
#
# docs/Project.toml takes NonlinearOptics from this checkout ([sources] needs
# Julia 1.11; on 1.10 run Pkg.develop(path = ".") in the docs project first).
# deploydocs only pushes from CI.

using Documenter
using NonlinearOptics

# The doctests call the native library from nlolib_jll. They pick the CPU
# backend, so the build needs no GPU.
DocMeta.setdocmeta!(NonlinearOptics, :DocTestSetup, :(using NonlinearOptics); recursive = true)

makedocs(
    sitename = "NonlinearOptics.jl",
    modules = [NonlinearOptics],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://kinsky-main.github.io/NonlinearOptics.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "API reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/kinsky-main/NonlinearOptics.jl.git",
    devbranch = "main",
    push_preview = true,
)
