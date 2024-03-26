//
// Created by Karim Younus on 25/03/2024.
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

long sumTotientsSequential(unsigned long lower, unsigned long upper) {
    // Function to sum the totients across a range of numbers from n=lower to n=upper

    unsigned sum = 0;
    unsigned n;

    for (n = lower; n <= upper; n++)
        sum = sum + euler(n);

    return sum;
}

int main(int argc, char *argv[])
{
    // Algorithm Bounds
    long lower, upper;
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
               lower, upper, sumTotientsSequential(lower, upper));
    }

    //Init MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    MPI_Finalize();



    return 0;
}