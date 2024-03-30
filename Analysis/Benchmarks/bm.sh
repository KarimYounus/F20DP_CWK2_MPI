#!/bin/bash
#SBATCH --partition=amd-longq
#SBATCH --job-name=sum_totients
#SBATCH --output=sum_totients_%j.out
#SBATCH --error=sum_totients_error_%j.out

# Load the necessary modules
module load mpich

# Define an array of MPI processes to use "2" "4" "8" "16" "32" "64" "96" "128" "160"
declare -a mpi_processes=("2" "4" "8" "16" "32" "64" "96" "128" "160" "192")

# Define the dataset ranges
declare -a datasets=("1 15000" "1 30000" "1 100000")

# Define whether to use the --dynamic flag
declare -a dynamic_modes=("" "--dynamic")

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

        for dynamic in "${dynamic_modes[@]}"; do
          # Adjust filename based on whether --dynamic is used
          if [ -z "$dynamic" ]; then
              filename="benchmark_${upper}.csv"
          else
              filename="benchmark_${upper}_dynamic.csv"
          fi

          for iteration in {1..3}; do
            sbatch <<EOF
#!/bin/bash
#SBATCH --partition=amd-longq
#SBATCH --nodes=$required_nodes
#SBATCH --ntasks=$mpi_procs
#SBATCH --job-name=sum_totients_$mpi_procs
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load mpich
mpirun ./SumTotient $lower $upper $dynamic --filename $filename
EOF
            done
        done
    done
done

echo "All benchmarks completed."

