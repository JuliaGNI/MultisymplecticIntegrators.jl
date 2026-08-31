using MultiSymplectic
using Documenter

DocMeta.setdocmeta!(MultiSymplectic, :DocTestSetup, :(using MultiSymplectic); recursive = true)

makedocs(;
    modules = [MultiSymplectic],
    authors = "Michael Kraus",
    repo = "https://github.com/ZeyuanLee/MultiSymplectic.jl/blob/{commit}{path}#{line}",
    sitename = "MultiSymplectic.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ZeyuanLee.github.io/MultiSymplectic.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Library" => "library.md"
    ]
)

deploydocs(;
    repo = "github.com/ZeyuanLee/MultiSymplectic.jl",
    devbranch = "main"
)
