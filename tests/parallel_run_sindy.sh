#!/bin/bash

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

    echo "Running Julia script with t_num_interval=$t_num_interval, x_num_interval=$x_num_interval, RT_per_interval=$RT_per_interval, RX_per_interval=$RX_per_interval, k_mu=$k_mu, k_lambda=$k_lambda, reg=$reg, mu=$mu_basis, lambda=$lambda_basis"

    SINDY_OUTPUT_DIR="sindyint_results" julia --project=. tests/test_sindy_PDE_int.jl \
        $t_num_interval $x_num_interval $RT_per_interval $RX_per_interval $k_mu $k_lambda $reg $mu_basis $lambda_basis &
}

for t_num_interval in 2 4; do
    for x_num_interval in 5; do
        for RT_per_interval in 4; do
            for RX_per_interval in 2 4; do
                for k_mu in 3; do
                    for k_lambda in 3; do
                        for reg in 1e-5 1e-7 1e-3; do
                            for mu_basis in BSplineDirichlet; do
                                for lambda_basis in BSplineDirichlet; do
                                    run_configuration $t_num_interval $x_num_interval $RT_per_interval $RX_per_interval $k_mu $k_lambda $reg $mu_basis $lambda_basis
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
