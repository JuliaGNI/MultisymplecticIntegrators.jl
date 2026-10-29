# The network family's driver.
#
# The registry holds no network entry yet, and this driver says so rather than inventing one: the
# network ansatz and its initial guess arrive with part F, and F adds its registry entries with them.
# The option parsing is the shared one, so a mistyped flag is refused here as everywhere else.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))

function main(args)
    parse_arguments(args)
    entries = family_entries(:network)
    if isempty(entries)
        println("run_network.jl: the registry holds no network entry; part F adds the network " *
                "ansatz and its registry entries.")
        return 0
    end
    error("the network family has entries but no driver body; part F adds both")
end

exit(main(ARGS))
