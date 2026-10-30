# The symbolic family's driver.
#
# The registry holds no symbolic entry, and this driver says so rather than inventing one: the
# package has no symbolic ansatz, and the registry entries come with it. The option parsing is the
# shared one, so a mistyped flag is refused here as everywhere else.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))

function main(args)
    parse_arguments(args)
    entries = family_entries(:symbolic)
    if isempty(entries)
        println("run_symbolic.jl: the registry holds no symbolic entry; the entries come with " *
                "the symbolic ansatz.")
        return 0
    end
    error("the symbolic family has entries but no driver body")
end

exit(main(ARGS))
