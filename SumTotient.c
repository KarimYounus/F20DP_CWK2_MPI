//
// Created by Karim Younus on 20/03/2024.
//
#include <stdbool.h>
#include <printf.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

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

long sumTotientsParalllel(unsigned long lower, unsigned long upper) {
    // Function to sum the totients across a range of numbers from n=lower to n=upper

    unsigned sum = 0;
    unsigned n;

    for (n = lower; n <= upper; n++)
        sum = sum + euler(n);

    return sum;
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
    fprintf(fp, "%ld, %f, %d", upper, exec_t, n_cores);

    // Close the file
    fclose(fp);
}

int main(int argc, char *argv[])
{
    // Algorithm Bounds
    long lower, upper;
    double start_time, end_time, runtime;
    bool seq;
    char *filename = NULL;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq") == 0) {    // Sequential?
            seq = true;
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

    // Run sequentially if seq argument is given
    if (seq) {
        printf("C: Sum of Totients  between [%ld..%ld] is %ld\n",
               lower, upper, sumTotients(lower, upper));
    }

    //Init MPI
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    unsigned long local_sum, global_sum;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Assign a unique ID to each process in the scope
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of unique processes in the scope

    if (world_rank == 0) {
        start_time = MPI_Wtime();
        printf("MPI Comm Size=[%i]\n", world_size);
        printf("Lower=[%ld], Upper=[%ld]\n", lower, upper);
    }

    // Calculate the workload distribution
    unsigned long local_lower = lower + (upper - lower) / world_size * world_rank;
    unsigned long local_upper = lower + (upper - lower) / world_size * (world_rank + 1);
    if (world_rank == world_size - 1) local_upper = upper; // Ensure the last process goes up to 'upper'

    // Perform local summation of totients for the current rank
    local_sum = sumTotients(local_lower, local_upper);

    // Get all local sums and add to obtain final result (in the root process)
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        end_time = MPI_Wtime();
        runtime = end_time - start_time;
        printf("Total sum of Euler's Totient function from %ld to %ld is: %ld\nRuntime=[%f]\n", lower, upper, global_sum, runtime);
    }
    MPI_Finalize();

    return 0;
}