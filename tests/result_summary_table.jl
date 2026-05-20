using JLD2
using CairoMakie
using Statistics
# using Colors

# R_list = [8,16,4]
# S_list = [4,6,8]

h_list = [0.05,0.1,0.2,0.4]#2.0,5.0
k_list = [3,4,5]
# λ_list = [0.0,1e-3,1e-5,1e-7]
t_knot_intervals = [0.5,0.25]
x_knot_intervals = [0.5,0.2]

second_order_reference = h_list .^ 2
third_order_reference = h_list .^ 3

num_lines = length(k_list) * length(t_knot_intervals) * length(x_knot_intervals)
# col = distinguishable_colors(num_lines,[RGB(1,1,1), RGB(0,0,0)], dropseed=true)
# line_colors = map(col -> (red(col), green(col), blue(col)), col)

begin
    spline_err_tensor = zeros(length(h_list),length(k_list),length(t_knot_intervals),length(x_knot_intervals))
    # q_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(λ_list))
    for (hi,int_step) in enumerate(h_list)
        for (ki,k) in enumerate(k_list)
            for (t_knot_interval_i,t_knot_interval) in enumerate(t_knot_intervals)
                for (x_knot_interval_i,x_knot_interval) in enumerate(x_knot_intervals)
                    data_file="full_restriction_spline_1005/SplineInt_Hamiltonian_Error_h=$(int_step)_k=$(k)_t_knot_interval=$(t_knot_interval)_x_knot_interval=$(x_knot_interval)_regulizer=0.0_078.jld2"
                    # isfile(data_file) ? nothing : continue
                    try
                        result_data = load(data_file)
                        err = result_data["maximum_relative_ham_err"]
                        # err = minimum(result_data["relative_hamiltonian_error"])

                        if err >10.0
                            println("Warning: Unusually high error in $(data_file): $err")
                            continue
                        end
                        spline_err_tensor[hi,ki,t_knot_interval_i,x_knot_interval_i] = err
                    catch e
                        println("Failed to load data from $(data_file): $(e)")
                        continue
                    end
                end
            end
        end
    end

    tick_size = 22
    label_size = 22
    fig = Figure(size = (1000, 1200))
    ax = Axis(fig[1, 1], xlabel = "Time Step h", ylabel = "Maximum Hamiltonian Error", 
        yscale = log10,xticks=([0.05,0.1,0.2,0.4], ["0.05", "0.1", "0.2", "0.4"]),limits = (nothing, (1e-8, 10)),
        yticklabelsize=tick_size, xticklabelsize=tick_size,xlabelsize=label_size, ylabelsize=label_size)
    global line_idx = 1
    line_styles = [:solid, :dash, :dot]
    for (ki,k) in enumerate(k_list)
        for (t_knot_interval_i,t_knot_interval) in enumerate(t_knot_intervals)
            for (x_knot_interval_i,x_knot_interval) in enumerate(x_knot_intervals)
                global line_idx
                scatterlines!(ax, h_list, spline_err_tensor[:,ki,t_knot_interval_i,x_knot_interval_i], label="k$(k)t_knot_interval$(t_knot_interval)x_knot_interval$(x_knot_interval)",
                            markersize=6, linewidth=3,linestyle=line_styles[ki])                            
                line_idx += 1
            end
        end
    end
    # axislegend(ax, position=:lt, labelsize=22)
    # axislegend(ax2, position=:rb,labelsize=22)

    # ref2 = lines!(ax2, h_list, 1e-4 .* second_order_reference, color=:black, linestyle=:dash)
    # ref3 = lines!(ax2, h_list, 1e-5 .* third_order_reference, color=:black, linestyle=:dot)

    Legend(
        fig[1,2],
        ax,
        orientation = :vertical,fontsize=label_size
    )

    save("full_restriction_spline_1005/Wave_hamiltonian_error.pdf", fig)
end

### Harmonic Oscillator with Time-Reversible Neural Variational Integrators
# begin
#     TR_HO_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(f_abs_list),length(x_abs_list))
#     TR_q_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(f_abs_list),length(x_abs_list))

#     for (hi,int_step) in enumerate(h_list)
#         for (Si,S) in enumerate(S_list)
#             for (ki,k_relu) in enumerate(k_list)
#                 for (Ri,R) in enumerate(R_list)
#                     Q = 2 * R
#                     for (fi,f_abs) in enumerate(f_abs_list)
#                         for (xi,x_abs) in enumerate(x_abs_list)
#                             # println("Loading TR data for h=$(int_step), S=$(S), k=$(k_relu), R=$(R), f_abs=$(f_abs), x_abs=$(x_abs)")
#                             TR_data_file ="time_reversible/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xabs$(x_abs).jld2"
#                             isfile(TR_data_file) ? nothing : continue
#                             try
#                                 result_data = load(TR_data_file)
#                                 TR_HO_err_tensor[hi,Si,ki,Ri,fi,xi] = result_data["HO_max_hams_err"]
#                                 TR_q_err_tensor[hi,Si,ki,Ri,fi,xi] = result_data["HO_qerror"] 
#                             catch e
#                                 println("Failed to load data from $(TR_data_file): $(e)")
#                                 continue
#                             end

#                         end
#                     end
#                 end

#             end
#         end
#     end

#     fig2 = Figure(size = (700, 800))
#     ax2 = Axis(fig2[1, 1], xlabel = "Time Step h", ylabel = "Maximum Hamiltonian Error", 
#         xscale = log10, yscale = log10, title = "Time-Reversible Neural Variational Integrators",
#         limits = (nothing, (1e-10, 1e2)))
#     global line_idx = 1
#     # fix S,k and compute the mean,max,min over R,f_abs,x_abs
#     for (Si, S) in enumerate(S_list)
#         for (ki, k_relu) in enumerate(k_list)
#             global line_idx
#             TR_err_mean = zeros(length(h_list))
#             TR_err_max = zeros(length(h_list))
#             TR_err_min = zeros(length(h_list))
            
#             for (hi, h) in enumerate(h_list)
#                 errors = TR_HO_err_tensor[hi, Si, ki, :, :, :]
#                 valid_err = filter(!isnan, errors)
#                 valid_err = filter(!iszero,valid_err)

#                 if isempty(valid_err) 
#                     println("Skipping h=$(h), S=$(S), k=$(k_relu) due to all NaNs, zeros")
#                     continue
#                 end

#                 TR_err_mean[hi] = mean(valid_err)
#                 TR_err_max[hi] = maximum(valid_err)
#                 TR_err_min[hi] = minimum(valid_err)
#             end
            
#             TR_errlow = TR_err_mean .- TR_err_min
#             TR_errhigh = TR_err_max .- TR_err_mean
#             scatterlines!(ax2, h_list, TR_err_mean, label="S$(S)k$(k_relu)", color=line_colors[line_idx],
#                         markersize=6, linewidth=2)
#             errorbars!(ax2, h_list, TR_err_mean, TR_errlow, TR_errhigh, linewidth=2, color=line_colors[line_idx],whiskerwidth = 10)

#             if S == 4 && k_relu == 3
#                 @show TR_err_mean
#                 @show TR_err_max
#                 @show TR_err_min

#                 @show TR_errlow
#                 @show TR_errhigh
#             end

#             line_idx += 1
#         end
#     end

#     axislegend(ax2, position=:rb)
#     fig2
#     save("time_reversible/HO_hamiltonian_error_TR.pdf", fig2)
# end

### Double Pendulum with Plain Neural Variational Integrators
# begin
#     DP_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(f_abs_list),length(x_abs_list))
#     DP_q_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(f_abs_list),length(x_abs_list))

#     for (hi,int_step) in enumerate(h_list)
#         for (Si,S) in enumerate(S_list)
#             for (ki,k_relu) in enumerate(k_list)
#                 for (Ri,R) in enumerate(R_list)
#                     Q = 2 * R
#                     for (fi,f_abs) in enumerate(f_abs_list)
#                         for (xi,x_abs) in enumerate(x_abs_list)
#                             data_file="time_reversible/NVI_DP_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xabs$(x_abs).jld2"
#                             isfile(data_file) ? nothing : continue
#                             result_data = load(data_file)
#                             DP_err_tensor[hi,Si,ki,Ri,fi,xi] = result_data["DP_max_hams_err"]
#                             DP_q_err_tensor[hi,Si,ki,Ri,fi,xi] = result_data["DP_qerror"] 
#                         end
#                     end
#                 end

#             end
#         end
#     end


#     fig3 = Figure(size = (700, 800))
#     ax3 = Axis(fig3[1, 1], xlabel = "Time Step h", ylabel = "Maximum Hamiltonian Error", 
#         xscale = log10, yscale = log10, title = "Neural Variational Integrators" ,limits = (nothing, (1e-10, 1e2)))

#     global line_idx = 1
#     for (Si, S) in enumerate(S_list)
#         for (ki, k_relu) in enumerate(k_list)
#             global line_idx
#             err_mean = zeros(length(h_list))
#             err_max = zeros(length(h_list))
#             err_min = zeros(length(h_list))
            
#             for (hi, h) in enumerate(h_list)
#                 errors = DP_err_tensor[hi, Si, ki, :, :, :]
#                 if S == 6 && k_relu == 3 && h == 1.0
#                     @show errors
#                 end
#                 valid_err = filter(!isnan, errors)
#                 valid_err = filter(!iszero,valid_err)

#                 if isempty(valid_err) 
#                     println("Skipping h=$(h), S=$(S), k=$(k_relu) due to all NaNs, zeros")
#                     continue
#                 end

#                 err_mean[hi] = mean(valid_err)
#                 err_max[hi] = maximum(valid_err)
#                 err_min[hi] = minimum(valid_err)
#             end

#             errlow = err_mean .- err_min
#             errhigh = err_max .- err_mean
#             scatterlines!(ax3, h_list, err_mean, label="S$(S)k$(k_relu)", color=line_colors[line_idx],
#                         markersize=6, linewidth=2)
#             errorbars!(ax3, h_list, err_mean, errlow, errhigh, linewidth=2, color=line_colors[line_idx],whiskerwidth = 10)
#             line_idx += 1
#         end
#     end
#     axislegend(ax3, position=:rb)
#     save("time_reversible/DP_hamiltonian_error.pdf", fig3)
# end

### Double Pendulum with Time-Reversible Neural Variational Integrators
# begin
#     TR_DP_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(f_abs_list),length(x_abs_list))
#     TR_DP_q_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(f_abs_list),length(x_abs_list))

#     for (hi,int_step) in enumerate(h_list)
#         for (Si,S) in enumerate(S_list)
#             for (ki,k_relu) in enumerate(k_list)
#                 for (Ri,R) in enumerate(R_list)
#                     Q = 2 * R
#                     for (fi,f_abs) in enumerate(f_abs_list)
#                         for (xi,x_abs) in enumerate(x_abs_list)
#                             # println("Loading TR data for h=$(int_step), S=$(S), k=$(k_relu), R=$(R), f_abs=$(f_abs), x_abs=$(x_abs)")
#                             TR_data_file ="time_reversible/NVI_DP_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xabs$(x_abs).jld2"
#                             isfile(TR_data_file) ? nothing : continue
#                             try
#                                 result_data = load(TR_data_file)
#                                 TR_DP_err_tensor[hi,Si,ki,Ri,fi,xi] = result_data["DP_max_hams_err"]
#                                 TR_DP_q_err_tensor[hi,Si,ki,Ri,fi,xi] = result_data["DP_qerror"] 
#                             catch e
#                                 println("Failed to load data from $(TR_data_file): $(e)")
#                                 continue
#                             end

#                         end
#                     end
#                 end

#             end
#         end
#     end


#     fig4 = Figure(size = (700, 800))
#     ax4 = Axis(fig4[1, 1], xlabel = "Time Step h", ylabel = "Maximum Hamiltonian Error", 
#         xscale = log10, yscale = log10, title = "Time-Reversible Neural Variational Integrators",
#         limits = (nothing, (1e-10, 1e2)))
#     global line_idx = 1
#     # fix S,k and compute the mean,max,min over R,f_abs,x_abs
#     for (Si, S) in enumerate(S_list)
#         for (ki, k_relu) in enumerate(k_list)
#             global line_idx
#             TR_err_mean = zeros(length(h_list))
#             TR_err_max = zeros(length(h_list))
#             TR_err_min = zeros(length(h_list))
            
#             for (hi, h) in enumerate(h_list)
#                 errors = TR_DP_err_tensor[hi, Si, ki, :, :, :]
#                 valid_err = filter(!isnan, errors)
#                 valid_err = filter(!iszero,valid_err)

#                 if isempty(valid_err) 
#                     println("Skipping h=$(h), S=$(S), k=$(k_relu) due to all NaNs, zeros")
#                     continue
#                 end

#                 TR_err_mean[hi] = mean(valid_err)
#                 TR_err_max[hi] = maximum(valid_err)
#                 TR_err_min[hi] = minimum(valid_err)
#             end
            
#             TR_errlow = TR_err_mean .- TR_err_min
#             TR_errhigh = TR_err_max .- TR_err_mean
#             scatterlines!(ax4, h_list, TR_err_mean, label="S$(S)k$(k_relu)", color=line_colors[line_idx],
#                         markersize=6, linewidth=2)
#             errorbars!(ax4, h_list, TR_err_mean, TR_errlow, TR_errhigh, linewidth=2, color=line_colors[line_idx],whiskerwidth = 10)

#             if S == 4 && k_relu == 3
#                 @show TR_err_mean
#                 @show TR_err_max
#                 @show TR_err_min

#                 @show TR_errlow
#                 @show TR_errhigh
#             end

#             line_idx += 1
#         end
#     end

#     axislegend(ax4, position=:rb)
#     fig4
#     save("time_reversible/DP_hamiltonian_error_TR.pdf", fig4)
# end

# # Harmonic Oscillator with tanh activation function 
# begin
#     HO_err_tensor_tanh = zeros(length(h_list),length(S_list),length(R_list),length(f_abs_list),length(x_abs_list))
#     q_err_tensor_tanh = zeros(length(h_list),length(S_list),length(R_list),length(f_abs_list),length(x_abs_list))

#     for (hi,int_step) in enumerate(h_list)
#         for (Si,S) in enumerate(S_list)
#             for (Ri,R) in enumerate(R_list)
#                 Q = 2 * R
#                 for (fi,f_abs) in enumerate(f_abs_list)
#                     for (xi,x_abs) in enumerate(x_abs_list)
#                         data_file="time_reversible/NVI_HO_h$(int_step)S$(S)R$(R)fabs$(f_abs)xabs$(x_abs)tanh.jld2"
#                         isfile(data_file) ? nothing : continue
#                         result_data = load(data_file)
#                         HO_err_tensor_tanh[hi,Si,Ri,fi,xi] = result_data["HO_max_hams_err"]
#                         q_err_tensor_tanh[hi,Si,Ri,fi,xi] = result_data["HO_qerror"] 
#                     end
#                 end
#             end
#         end
#     end


#     fig5 = Figure(size = (700, 800))
#     ax5 = Axis(fig5[1, 1], xlabel = "Time Step h", ylabel = "Maximum Hamiltonian Error", 
#         xscale = log10, yscale = log10, title = "Neural Variational Integrators" ,limits = (nothing, (1e-10, 1e2)))

#     global line_idx = 1

#     for (Si, S) in enumerate(S_list)
#             global line_idx
#             err_mean = zeros(length(h_list))
#             err_max = zeros(length(h_list))
#             err_min = zeros(length(h_list))
            
#             for (hi, h) in enumerate(h_list)
#                 errors = HO_err_tensor_tanh[hi, Si, :, :, :]
#                 valid_err = filter(!isnan, errors)
#                 valid_err = filter(!iszero,valid_err)

#                 if isempty(valid_err) 
#                     println("Skipping h=$(h), S=$(S) due to all NaNs, zeros")
#                     continue
#                 end

#                 err_mean[hi] = mean(valid_err)
#                 err_max[hi] = maximum(valid_err)
#                 err_min[hi] = minimum(valid_err)
#             end

#             errlow = err_mean .- err_min
#             errhigh = err_max .- err_mean
#             scatterlines!(ax5, h_list, err_mean, label="S$(S)", color=line_colors[line_idx],
#                         markersize=6, linewidth=2)
#             errorbars!(ax5, h_list, err_mean, errlow, errhigh, linewidth=2, color=line_colors[line_idx],whiskerwidth = 10)
#             line_idx += 1
#     end
#     axislegend(ax5, position=:rb)
#     fig5
#     save("time_reversible/HO_hamiltonian_error_tanh.pdf", fig5)
# end

# # Double Pendulum with tanh activation function 
# begin
#     DP_err_tensor_tanh = zeros(length(h_list),length(S_list),length(R_list),length(f_abs_list),length(x_abs_list))
#     DP_q_err_tensor_tanh = zeros(length(h_list),length(S_list),length(R_list),length(f_abs_list),length(x_abs_list))

#     for (hi,int_step) in enumerate(h_list)
#         for (Si,S) in enumerate(S_list)
#             for (Ri,R) in enumerate(R_list)
#                 Q = 2 * R
#                 for (fi,f_abs) in enumerate(f_abs_list)
#                     for (xi,x_abs) in enumerate(x_abs_list)
#                         data_file="time_reversible/NVI_DP_h$(int_step)S$(S)R$(R)fabs$(f_abs)xabs$(x_abs)tanh.jld2"
#                         isfile(data_file) ? nothing : continue
#                         result_data = load(data_file)
#                         DP_err_tensor_tanh[hi,Si,Ri,fi,xi] = result_data["DP_max_hams_err"]
#                         DP_q_err_tensor_tanh[hi,Si,Ri,fi,xi] = result_data["DP_qerror"] 
#                     end
#                 end
#             end
#         end
#     end


#     fig6 = Figure(size = (700, 800))
#     ax6 = Axis(fig6[1, 1], xlabel = "Time Step h", ylabel = "Maximum Hamiltonian Error", 
#         xscale = log10, yscale = log10, title = "Neural Variational Integrators" ,limits = (nothing, (1e-10, 1e2)))

#     global line_idx = 1

#     for (Si, S) in enumerate(S_list)
#             global line_idx
#             err_mean = zeros(length(h_list))
#             err_max = zeros(length(h_list))
#             err_min = zeros(length(h_list))
            
#             for (hi, h) in enumerate(h_list)
#                 errors = DP_err_tensor_tanh[hi, Si, :, :, :]
#                 valid_err = filter(!isnan, errors)
#                 valid_err = filter(!iszero,valid_err)

#                 if isempty(valid_err) 
#                     println("Skipping h=$(h), S=$(S) due to all NaNs, zeros")
#                     continue
#                 end

#                 err_mean[hi] = mean(valid_err)
#                 err_max[hi] = maximum(valid_err)
#                 err_min[hi] = minimum(valid_err)
#             end

#             errlow = err_mean .- err_min
#             errhigh = err_max .- err_mean
#             scatterlines!(ax6, h_list, err_mean, label="S$(S)", color=line_colors[line_idx],
#                         markersize=6, linewidth=2)
#             errorbars!(ax6, h_list, err_mean, errlow, errhigh, linewidth=2, color=line_colors[line_idx],whiskerwidth = 10)
#             line_idx += 1
#     end
#     axislegend(ax6, position=:rb)
#     fig6
#     save("DP_hamiltonian_error_tanh.pdf", fig6)
# end