#!/bin/bash

# export JULIA_NUM_THREADS=20
# for i in {1..10}
# do
#     julia IntegratorNN/test_sh.jl "$i"|| true
# done

# seq 12 | xargs -I{} -P 6 bash -c 'julia IntegratorNN/test_sh2.jl {}|| true'


# Function to run the Julia script with the specified activation function
run_configuration() {
    local k=$1
    local t_step=$2
    local t_knot_interval=$3
    local x_knot_interval=$4

    # Print the activation for debugging
    echo "Running Julia script with k=$k, t_step=$t_step, t_knot_interval=$t_knot_interval,x_knot_interval=$x_knot_interval" 

    # Run the Julia script in the background
    julia --project=. tests/test_Spline_int.jl $k $t_step $t_knot_interval $x_knot_interval &
}

# Loop through the activations
for k in {3,4}; do # ,
    for t_step in {0.05,0.1,0.2}; do #  ,
        for t_knot_interval in {0.5,0.25}; do # 
            for x_knot_interval in {0.05,0.1}; do # 
                run_configuration $k $t_step $t_knot_interval $x_knot_interval
            done
        done
    done
done

