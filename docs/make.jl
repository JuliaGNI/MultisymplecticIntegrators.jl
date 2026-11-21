using MultisymplecticIntegrators
using Documenter

DocMeta.setdocmeta!(MultisymplecticIntegrators, :DocTestSetup,
    :(using MultisymplecticIntegrators); recursive = true)

makedocs(;
    modules = [MultisymplecticIntegrators],
    authors = "Michael Kraus",
    repo = "https://github.com/JuliaGNI/MultisymplecticIntegrators.jl/blob/{commit}{path}#{line}",
    sitename = "MultisymplecticIntegrators.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaGNI.github.io/MultisymplecticIntegrators.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "The families" => [
            "The B-spline family" => "bspline.md",
            "The multiplier network" => "network.md",
            "The ELM family" => "elm.md",
            "The TrialNN family" => "trialnn.md",
            "The symbolic family" => "symbolic.md",
            "The fixed-boundary network" => "fixed_boundary.md",
            "The box scheme" => "box_scheme.md"
        ],
        "The slab seam" => "seam.md",
        "Library" => "library.md",
        "The ansatz families" => "ansatz.md",
        "Couplings and initial guesses" => "couplings.md",
        "Example problems and diagnostics" => "problems.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaGNI/MultisymplecticIntegrators.jl",
    devbranch = "main"
)
