#!/bin/bash

# export JULIA_NUM_THREADS=20
# for i in {1..10}
# do
#     julia IntegratorNN/test_sh.jl "$i"|| true
# done

# seq 12 | xargs -I{} -P 6 bash -c 'julia IntegratorNN/test_sh2.jl {}|| true'


# Function to run the Julia script with the specified activation function
run_configuration() {
    local h=$1
    local rt=$2
    local Nw=$3
    local Nb=$4

    # Print the activation for debugging
    echo "Running Julia script with Step Size: $h, RT: $rt, Nw: $Nw, Nb: $Nb"

    # Run the Julia script in the background
    julia --project=. tests/test_Trial_NN_PDE_int.jl $h $rt $Nw $Nb &
}

# Loop through the activations
for h in {0.1,0.3}; do # ,
    for rt in {12,24}; do #  
        for Nw in {500,600}; do # 
            for Nb in {500,600}; do # 
                run_configuration $h $rt $Nw $Nb
            done
        done
    done
done

