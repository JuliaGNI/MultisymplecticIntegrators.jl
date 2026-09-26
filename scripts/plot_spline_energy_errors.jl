using CairoMakie
using JLD2
using Printf

input_path = get(ENV, "SPLINE_SCAN_INPUT", "debug_results/spline_energy_scan.jld2")
output_path = get(ENV, "SPLINE_ENERGY_PLOT", "debug_results/spline_reg0_energy_errors.png")
t_step = parse(Float64, get(ENV, "SPLINE_T_STEP", "0.1"))

data = load(input_path)
results = data["results"]
isempty(results) && error("No scan results found in $(input_path)")

result = results[1]
signed_relerr = result.signed_relerr
abs_relerr = abs.(signed_relerr)
t = collect(0:(length(signed_relerr) - 1)) .* t_step

fig = Figure(size = (980, 620), fontsize = 18)
ax1 = Axis(fig[1, 1],
    xlabel = "t",
    ylabel = "signed relative Hamiltonian error",
    title = @sprintf("reg = %.1e, tspan = [0, %.1f]", result.regularization_factor,
        t[end] + t_step)
)
lines!(ax1, t, signed_relerr, color = :dodgerblue3, linewidth = 2, label = "signed")
hlines!(ax1, [0.0], color = :gray40, linestyle = :dash, linewidth = 1)
axislegend(ax1, position = :rt)

ax2 = Axis(fig[2, 1],
    xlabel = "t",
    ylabel = "absolute relative Hamiltonian error"
)
lines!(ax2, t, abs_relerr, color = :firebrick3, linewidth = 2, label = "absolute")
axislegend(ax2, position = :rt)

save(output_path, fig)

@printf("saved plot: %s\n", output_path)
@printf("points=%d  max_abs=%9.3e  final_signed=%+9.3e  sign_changes=%d\n",
    length(signed_relerr),
    maximum(abs_relerr),
    signed_relerr[end],
    count(diff(sign.(signed_relerr)) .!= 0))
