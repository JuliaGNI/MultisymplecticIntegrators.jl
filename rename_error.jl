using JLD2
using Logging
using Printf

const ERR_PATTERN = r"_err[^_]+_"

function format_err(err)
    return @sprintf("%.3e", err)
end

function target_name(path)
    data = with_logger(NullLogger()) do
        load(path)
    end

    ham_ls = data["ham_ls"]
    analytic_ham = data["analytic_ham"]
    ham_err = abs.((ham_ls .- analytic_ham) ./ analytic_ham)

    err = maximum(abs.(ham_err))

    new_base = replace(basename(path), ERR_PATTERN => "_err$(format_err(err))_")
    return joinpath(dirname(path), new_base), err
end

function main()
    apply = "--apply" in ARGS
    dirs = filter(arg -> arg != "--apply", ARGS)
    dir = isempty(dirs) ? joinpath(@__DIR__, "sindyint_results") : only(dirs)

    files = sort(filter(f -> endswith(f, ".jld2"), readdir(dir; join = true)))
    plans = Tuple{String,String,Float64}[]

    for file in files
        new_file, err = target_name(file)
        if file != new_file
            push!(plans, (file, new_file, err))
        end
    end

    targets = [new_file for (_, new_file, _) in plans]
    if length(unique(targets)) != length(targets)
        error("Renaming would create duplicate target filenames.")
    end

    sources = first.(plans)
    existing_targets = filter(new_file -> isfile(new_file) && !(new_file in sources), targets)
    if !isempty(existing_targets)
        error("Renaming would overwrite existing files: $(join(basename.(existing_targets), ", "))")
    end

    println(apply ? "Applying renames:" : "Dry run:")
    for (old_file, new_file, err) in plans
        @printf("%s -> %s  (max abs relative_ham_err = %.16g)\n",
            basename(old_file), basename(new_file), err)
    end
    println("Files to rename: $(length(plans)) / $(length(files))")

    if apply
        temp_plans = [(old_file, tempname(dirname(old_file)), new_file) for (old_file, new_file, _) in plans]
        for (old_file, temp_file, _) in temp_plans
            mv(old_file, temp_file)
        end
        for (_, temp_file, new_file) in temp_plans
            mv(temp_file, new_file)
        end
    else
        println("Run with --apply to rename files.")
    end
end

main()
