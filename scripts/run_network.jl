# The network family's driver.
#
# The registry holds no network entry, and this driver says so rather than inventing one: the
# package has no network ansatz and initial guess, and the registry entries come with them. The
# option parsing is the shared one, so a mistyped flag is refused here as everywhere else.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))

function main(args)
    parse_arguments(args)
    entries = family_entries(:network)
    if isempty(entries)
        println("run_network.jl: the registry holds no network entry; the entries come with " *
                "the network ansatz.")
        return 0
    end
    error("the network family has entries but no driver body")
end

exit(main(ARGS))
