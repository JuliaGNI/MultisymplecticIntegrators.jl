#!/bin/bash

max_jobs=${MAX_JOBS:-$(nproc)}

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
        sleep 1
    done
}

# Function to run the Julia script with the specified SINDy configuration
run_configuration() {
    local t_num_interval=$1
    local x_num_interval=$2
    local RT_per_interval=$3
    local RX_per_interval=$4
    local k_mu=$5
    local k_lambda=$6
    local reg=$7
    local mu_basis=$8
    local lambda_basis=$9
    local t_step=${10}
    echo "Running Julia script with t_num_interval=$t_num_interval, x_num_interval=$x_num_interval, RT_per_interval=$RT_per_interval, RX_per_interval=$RX_per_interval, k_mu=$k_mu, k_lambda=$k_lambda, reg=$reg, mu=$mu_basis, lambda=$lambda_basis, t_step=$t_step"

    SINDY_OUTPUT_DIR="sindyint_results" julia --project=. scripts/test_sindy_PDE_int.jl \
        $t_num_interval $x_num_interval $RT_per_interval $RX_per_interval $k_mu $k_lambda $reg $mu_basis $lambda_basis $t_step &
}
for t_num_interval in 6; do
    for x_num_interval in 5 10; do
        for RT_per_interval in 6; do
            for RX_per_interval in 6; do
                for k_mu in 3; do
                    for k_lambda in 3; do
                        for reg in 1e-3 1e-5 1e-7 1e-9 1e-11 1e-13; do
                            for mu_basis in BSplineDirichlet; do
                                for lambda_basis in BSplineDirichlet; do
                                    for t_step in 0.01 0.02 0.05; do
                                        wait_for_slot
                                        run_configuration $t_num_interval $x_num_interval $RT_per_interval $RX_per_interval $k_mu $k_lambda $reg $mu_basis $lambda_basis $t_step
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

wait
