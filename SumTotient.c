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

int main(int argc, char *argv[])
{
    // Algorithm Bounds
    long lower, upper;
    double start_time, end_time, runtime, multiplier;
    bool seq, chunk;
    char *filename = NULL;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq") == 0) {    // Sequential?
            seq = true;
            continue;
        }
        if (strcmp(argv[i], "--dynamic") == 0 && i + 1 < argc) {    // Dynamic chunking? - check to ensure there is at least one more arg after the flag
            chunk = true;
            multiplier = atof(argv[++i]);
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
            if (chunk) {
                printf("Using dynamic chunking . . .\n");
            }
            printf("MPI Comm Size=[%i]\n", world_size);
            printf("Lower=[%ld], Upper=[%ld]\n", lower, upper);
        }

        // Workload object for potential dynamic chunking
        Workload workload;

        // Calculate the workload distribution
        if (chunk) {
            // If dynamic chunking, calculate chunks across the given range
            workload = calculateWorkload(world_rank, world_size, lower, upper, multiplier);
            if (world_rank == 0) {
                printf("Workload: [%i] processes processing approximately [%d] chunks each\n", world_size,
                       (workload.num_chunks / world_size));
            }
        }
        else {
            // Else just split the range equally
            unsigned long range = upper - lower + 1;
            unsigned long chunk_size = range / world_size;
            unsigned long remainder = range % world_size;

            local_lower = lower + chunk_size * world_rank;
            local_upper = local_lower + chunk_size - 1;
            if (world_rank == world_size - 1) {
                local_upper += remainder;
            }
        }

        // Perform local summation of totients for the current rank
        local_sum = 0;
        if (chunk) {
            // Use Dynamic Chunking, processing each chunk in the workload struct
            for (int i = 0; i < workload.num_chunks; i++) {
                unsigned long chunk_lower = workload.chunks[i].lower;
                unsigned long chunk_upper = workload.chunks[i].upper;
                // Only print the chunk range if it belongs to the current process
                if (chunk_lower != 0 || chunk_upper != 0) {
                    printf("Process %d handling range [%lu, %lu]\n", world_rank, chunk_lower, chunk_upper);
                }
                local_sum += sumTotients(chunk_lower, chunk_upper);
            }
            // Free the memory allocated for the chunks
            free(workload.chunks);
        } else {
            // Use Equal Chunking
            printf("Process %d handling range [%lu, %lu]\n", world_rank, local_lower, local_upper);
            local_sum = sumTotients(local_lower, local_upper);
        }

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
        // If seq is true, run sequentially
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