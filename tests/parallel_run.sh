#!/bin/bash

# export JULIA_NUM_THREADS=20
# for i in {1..10}
# do
#     julia IntegratorNN/test_sh.jl "$i"|| true
# done

# seq 12 | xargs -I{} -P 6 bash -c 'julia IntegratorNN/test_sh2.jl {}|| true'


# Function to run the Julia script with the specified activation function
run_configuration() {
    local tstep=$1
    local reg_factor=$2
    local S=$3

    # Print the activation for debugging
    echo "Running Julia script with h=$tstep, reg_factor=$reg_factor, S=$S" 

    # Run the Julia script in the background
    julia --project=. tests/test_Trial_NN_PDE_int.jl $tstep $reg_factor $S &
}

# Loop through the activations
for tstep in {0.2,0.5}; do
    for reg_factor in {1e-5,1e-7}; do #  ,1e-7
        for S in {70,60}; do # 
            run_configuration $tstep $reg_factor $S 
        done
    done
done

