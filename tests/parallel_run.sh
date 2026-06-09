#!/bin/bash

# export JULIA_NUM_THREADS=20
# for i in {1..10}
# do
#     julia IntegratorNN/test_sh.jl "$i"|| true
# done

# seq 12 | xargs -I{} -P 6 bash -c 'julia IntegratorNN/test_sh2.jl {}|| true'


# Function to run the Julia script with the specified activation function
max_jobs=${MAX_JOBS:-$(nproc)}

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
        sleep 1
    done
}

run_configuration() {
    local t_step=$1
    local reg=$2
    local S=$3

    # Print the activation for debugging
    echo "Running Julia script with t_step=$t_step, S=$S,reg = $reg" 

    # Run the Julia script in the background
    julia --project=. tests/test_NN_PDE_int.jl $t_step $reg $S &
}

# Loop through the activations
for t_step in {0.5,0.2,0.1}; do #  ,
    for reg in {1e-7,1e-9};do
        for S in {70,60,80,50};do     
            wait_for_slot
            run_configuration $t_step $reg $S
        done
    done

done

