# The symbolic family's driver.
#
# The registry holds no symbolic entry yet, and this driver says so rather than inventing one: the
# symbolic ansatz arrives with part G, and G adds its registry entries with it. The option parsing is
# the shared one, so a mistyped flag is refused here as everywhere else.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))

function main(args)
    parse_arguments(args)
    entries = family_entries(:symbolic)
    if isempty(entries)
        println("run_symbolic.jl: the registry holds no symbolic entry; part G adds the symbolic " *
                "ansatz and its registry entries.")
        return 0
    end
    error("the symbolic family has entries but no driver body; part G adds both")
end

exit(main(ARGS))
