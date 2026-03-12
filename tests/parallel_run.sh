#!/bin/bash

# export JULIA_NUM_THREADS=20
# for i in {1..10}
# do
#     julia IntegratorNN/test_sh.jl "$i"|| true
# done

# seq 12 | xargs -I{} -P 6 bash -c 'julia IntegratorNN/test_sh2.jl {}|| true'


# Function to run the Julia script with the specified activation function
run_configuration() {
    local internal_k=$1
    local t_knot_interval=$2
    local x_knot_interval=$3
    local tstep=$4

    # Print the activation for debugging
    echo "Running Julia script with internal_k=$internal_k, t_knot_interval=$t_knot_interval, x_knot_interval=$x_knot_interval, tstep=$tstep" | tee -a parallel_run.log

    # Run the Julia script in the background
    julia --project=. tests/test_constraint_spline.jl $internal_k $t_knot_interval $x_knot_interval $tstep >> parallel_run.log 2>&1 &
}

# Loop through the activations
for internal_k in {4,5,6}; do # ,
    for t_knot_interval in {0.5,}; do #  
        for x_knot_interval in {0.1,0.05}; do # 
            for tstep in {0.1,0.2,0.4}; do
                # for reg_factor in {0.0,1e-3}; do
                run_configuration $internal_k $t_knot_interval $x_knot_interval $tstep 
                # done
            done
        done
    done
done

