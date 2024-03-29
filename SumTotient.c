//
// Created by Karim Younus on 20/03/2024.
//
#include <stdbool.h>
#include <printf.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

// Define a struct to represent a chunk
typedef struct {
    unsigned long lower;
    unsigned long upper;
} Chunk;

// Define a struct to represent the workload
typedef struct {
    Chunk* chunks;
    int num_chunks;
} Workload;

long gcdEuclid(unsigned long a, unsigned long b) {
    /*
    - Function to calculate the greatest common divisor (GCD) of two numbers.
    The Euclidean algorithm as implemented in the sequential version of the algorithm provided in the F20DP Gitlab:
    [source]: https://gitlab-student.macs.hw.ac.uk/f20dp/f20dp-totient-range/-/blob/master/TotientRange.c
     */

    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int relPrime(unsigned long a, unsigned long b) {
    // Function to determine if 2 numbers are relatively prime
    return gcdEuclid(a, b) == 1;
}

long euler(unsigned long n) {
    // Function to count the numbers up to n that are relatively prime to n

    unsigned long count = 0;
    unsigned long i;

    for (i = 1; i < n; i++)
        if (relPrime(i, n))
            count++;

    return count;
}

long sumTotients(unsigned long lower, unsigned long upper) {
    // Function to sum the totients across a range of numbers from n=lower to n=upper

    unsigned sum = 0;
    unsigned n;

    for (n = lower; n <= upper; n++)
        sum = sum + euler(n);

    return sum;
}

void output_metrics(const char* filename, long upper, double exec_t, int n_cores) {
    FILE *fp;

    // Open the file. If the file does not exist, it will be created.
    fp = fopen(filename, "a");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    // Check if file is empty to write header
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    if (size == 0) {
        fprintf(fp, "\"Upper\",\"Execution Time\", \"Core Count\"\n");
    }

    // Move back to the end of the file to append data
    fseek(fp, 0, SEEK_END);

    // Write the metrics to the file in CSV format
    fprintf(fp, "%ld, %f, %d\n", upper, exec_t, n_cores);

    // Close the file
    fclose(fp);
}

// Function to calculate the workload and return a Workload struct
Workload calculateWorkload(int rank, int size, unsigned long lower, unsigned long upper, double chunk_multiplier) {
    // Calculate the total range of numbers
    unsigned long range = upper - lower + 1;

    // Calculate the total number of chunks by multiplying the number of processes by the chunk multiplier and rounding to the nearest integer
    int num_chunks = (int)round(size * chunk_multiplier);

    // Allocate memory for the array of chunks
    Chunk* chunks = (Chunk*)malloc(num_chunks * sizeof(Chunk));

    // Calculate the chunk size by dividing the range by the total number of chunks
    unsigned long chunk_size = range / num_chunks;

    // Calculate the remainder (numbers left after dividing the range by the number of chunks)
    unsigned long remainder = range % num_chunks;

    // Calculate the start index of the chunk for the current process
    int chunk_start = (int)round(rank * chunk_multiplier);

    // Calculate the end index of the chunk for the current process
    int chunk_end = (int)round((rank + 1) * chunk_multiplier);

    // Iterate over the chunks assigned to the current process
    for (int i = chunk_start; i < chunk_end && i < num_chunks; i++) {
        // Calculate the lower bound for the current chunk
        chunks[i].lower = lower + i * chunk_size;

        // Adjust the lower bound based on the remainder
        if (i < remainder) {
            chunks[i].lower += i;
        } else {
            chunks[i].lower += remainder;
        }

        // Calculate the upper bound for the current chunk
        if (i < remainder) {
            chunks[i].upper = chunks[i].lower + chunk_size;
        } else {
            chunks[i].upper = chunks[i].lower + chunk_size - 1;
        }

        // Adjust the upper bound for the last chunk to ensure it ends at the upper bound of the range
        if (i == num_chunks - 1) {
            chunks[i].upper = upper;
        }
    }

    // Create a Workload struct to hold the chunks and the number of chunks
    Workload workload;
    workload.chunks = chunks;
    workload.num_chunks = num_chunks;

    return workload;
}

unsigned long calculateTotalWeight(unsigned long lower, unsigned long upper) {
//    Assuming the computational weight of each number n is proportional to n itself,
//    the total weight of a range can be approximated by the sum of all numbers in the range.
    unsigned long total_weight;

    // The formula for the sum of the first N natural numbers is N*(N+1)/2.
    // To find the sum of a range, we use the formula for the upper bound and subtract the sum for (lower-1).
    unsigned long long upper_sum = upper * (upper + 1) / 2;
    unsigned long long lower_sum = (lower > 0) ? (lower - 1) * (lower) / 2 : 0;

    total_weight = (upper_sum - lower_sum);

    return total_weight;
}

// Find next upper bound based on given computational weight
unsigned long countToWeight(unsigned long lower, double weight) {
    double current_weight = 0;
    unsigned long upper = lower;

    while(current_weight < weight) {
        current_weight += upper;
        upper++;
    }
    // Return the previous upper before we surpassed the weight limit
    return upper-1;
}

// Assign each process an equal amount of computational work
Chunk calculateWorkloads(int rank, int size, unsigned long lower, unsigned long upper) {
    // Get the total number of integers to cover
    int total_range = upper - lower;
    // Get the total weight of the range
    unsigned long total_weight = calculateTotalWeight(lower, upper);
    if (rank == 0) printf("Total Weight = %lu, ", total_weight);
    // Get the average weight per process for the current world size, rounding down to leave a remainder for the last process
    int weight_per_process = total_weight / size;
    if (rank == 0) printf("Process Weight = %i\n", weight_per_process);

    // At the start, we have the entire range to cover
    unsigned long remaining_range = total_range;
//    if (rank==0) printf("Range = %lu\n", remaining_range);
    // At the start, the lower for the current chunk is just the lower
    unsigned long current_lower = lower;
    unsigned long current_upper;

    // For each process in the world
    for (int i=0; i<size; i++) {
        // Calculate the next upper by summing the integers from the current lower until we accumulate the average weight
        current_upper = countToWeight(current_lower, weight_per_process);

        // Assign chunks to each process . . .
        if (rank == i) {
            // If last process, assign upper as remaining range to cleanup
            if (rank == size-1) {
                Chunk chunk;
                chunk.lower = current_lower;
                chunk.upper = current_lower+remaining_range;
//                printf("Remaining Range = %lu\n", remaining_range);
                printf("Rank = [%i], L/U = %lu > %lu\n", rank, chunk.lower, chunk.upper);
                return chunk;
            }
            // Normal Assignment
            Chunk chunk;
            chunk.lower = current_lower;
            chunk.upper = current_upper;
            printf("Rank = [%i], L/U = %lu > %lu\n", rank, chunk.lower, chunk.upper);
            return chunk;
        }
        // Remove this chunk from the remaining range
        remaining_range = remaining_range - (current_upper - current_lower) - 1;
        // Set the lower for the next chunk as the upper of this chunk + 1
        current_lower = current_upper+1;
    }
}

int main(int argc, char *argv[])
{
    // Algorithm Bounds
    long lower, upper;
    // Time tracking
    double start_time, end_time, runtime;
    // Optional Args
    bool seq, dynamic;
    char *filename = NULL;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq") == 0) {    // Sequential?
            seq = true;
            continue;
        }
        if (strcmp(argv[i], "--dynamic") == 0) {    // Dynamic chunking?
            dynamic = true;
            continue;
        }
        if (strcmp(argv[i], "--filename") == 0 && i + 1 < argc) {   // Filename? - check to ensure there is at least one more arg after the flag
            filename = argv[++i]; // Use the next arg as the filename
            continue;
        }
        if (i == 1) {    // Lower bound
            lower = strtoul(argv[i], NULL, 10);
        } else if (i == 2) {    // Upper bound
            upper = strtoul(argv[i], NULL, 10);
        }
    }

    if (!seq) {
        //Init MPI
        MPI_Init(&argc, &argv);
        int world_size, world_rank;
        unsigned long local_sum, global_sum, local_lower, local_upper;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Assign a unique ID to each process in the scope
        MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of unique processes in the scope
        if (world_rank == 0) {
            start_time = MPI_Wtime();
            if (dynamic) {
                printf("Using dynamic chunking . . .\n");
            }
            printf("MPI Comm Size=[%i]\n", world_size);
            printf("Lower=[%ld], Upper=[%ld]\n", lower, upper);
        }

        // Calculate the workload distribution for each process
        if (dynamic) {
            // If dynamic chunking, calculate chunks across the given range
            Chunk chunk;
            chunk = calculateWorkloads(world_rank, world_size, lower, upper);
            local_lower = chunk.lower;
            local_upper = chunk.upper;
        }
        else {
            // Else just split the range equally
            unsigned long range = upper - lower + 1;
            unsigned long chunk_size = range / world_size;
            unsigned long remainder = range % world_size;
            local_lower = lower + chunk_size * world_rank;
            local_upper = local_lower + chunk_size - 1;
            // Cleanup remainder
            if (world_rank == world_size - 1) {
                local_upper += remainder;
            }
        }

        // Barrier for tidy printing
        MPI_Barrier(MPI_COMM_WORLD);
//        if (world_rank == 0) printf("Summing . . . \n");
        // Calculate local sum for the assigned range
        local_sum = sumTotients(local_lower, local_upper);
//        printf("%lu\n", local_sum);

        // Get all local sums and add to obtain final result (in the root process)
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == 0) {
            end_time = MPI_Wtime();
            runtime = end_time - start_time;
            printf("Total sum of Totients from [%ld..%ld] is: %ld\nRuntime=[%f]\n", lower, upper,
                   global_sum, runtime);
        }
        if (filename != NULL && world_rank==0) {
            output_metrics(filename, upper, runtime, world_size);
        }
        MPI_Finalize();
    } else {
        // Else seq is true, run sequentially
        clock_t start, end;

        start = clock(); // Start the timer

        // Execute the sequential sum of totients calculation
        long totientSum = sumTotients(lower, upper);

        end = clock(); // End the timer

        // Calculate the elapsed time
        runtime = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Total sum of Totients from [%ld..%ld] is: %ld\nRuntime=[%f]\n",
               lower, upper, totientSum, runtime);
        if (filename != NULL) {
            output_metrics(filename, upper, runtime, 1);
        }
        return 0;
    }
    return 0;
}