#!/bin/bash

# Load the necessary modules
module load intel/mpi/64

# Define an array of MPI processes to use "2" "4" "8" "16" "32" "64" "96" "128" "160"
declare -a mpi_processes=("1" "2" "4" "8" "16" "32" "64" "128" "192")

# Define the dataset ranges
declare -a datasets=("1 15000" "1 30000" "1 100000")

# Calculate nodes and tasks for each MPI process setup
for mpi_procs in "${mpi_processes[@]}"; do
    # Calculate the required number of nodes, ensuring at least one node is used
    required_nodes=$((mpi_procs / 64 + (mpi_procs % 64 != 0)))

    echo "Using $required_nodes nodes . . ."

    # Limit the number of nodes to not exceed the total number available
    if [ $required_nodes -gt 3 ]; then
        required_nodes=3
    fi

    for ds in "${datasets[@]}"; do
        IFS=' ' read -ra ADDR <<< "$ds"  # Convert the string to an array
        lower=${ADDR[0]}
        upper=${ADDR[1]}

        # Construct the filename based on the dataset and mpi process count
        filename="benchmark_${upper}.csv"

        # Execute the program using srun
        for iteration in {1..3}; do
            echo "Running: MPI Processes = $mpi_procs, Range = $lower to $upper, Iteration = $iteration"
            srun --partition=amd-longq --ntasks="$mpi_procs" --nodes=$required_nodes --ntasks-per-node=64 --mpi=pmi2 SumTotient "$lower" "$upper" --filename "$filename"
        done
    done
done

echo "All benchmarks completed."

