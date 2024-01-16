/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>

#define M 1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N 8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


//function declaration
void initialize();
void routine1_vec(float alpha, float beta);
void routine2_vec(float alpha, float beta);
void routine1(float alpha, float beta);
void routine2(float alpha, float beta);

__declspec(align(64)) float  y[M], z[M];
__declspec(align(64)) float A[N][N], x[N], w[N];

int main() {

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    initialize();

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));

    printf("\nRoutine1_vec:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1_vec(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2_vec:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine2_vec(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));




    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }

    //initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
    }


}

void routine1(float alpha, float beta) {

    unsigned int i;


    for (i = 0; i < M; i++)
        y[i] = alpha * y[i] + beta * z[i];

}

void routine2(float alpha, float beta) {

    unsigned int i, j;


    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            w[i] = w[i] - beta + alpha * A[i][j] * x[j];


}

void routine1_vec(float alpha, float beta) {
    
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    unsigned int i;

    for (i = 0; i < M; i += 8) {
        //y[i] = alpha * y[i] + beta * z[i];
        __m256 y_vec = _mm256_load_ps(&y[i]);
        __m256 z_vec = _mm256_load_ps(&z[i]);

        //alpha* y[i] + beta * z[i]
        __m256 result = _mm256_add_ps(_mm256_mul_ps(alpha_vec, y_vec), _mm256_mul_ps(beta_vec, z_vec));

        //store result to y[i]
        _mm256_store_ps(&y[i], result);
    }
}

void routine2_vec(float alpha, float beta) {

    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    unsigned int i, j;

    for (i = 0; i < N; i++) {
        __m256 w_vec = _mm256_loadu_ps(&w[i]);
        for (j = 0; j < N; j += 8) {
            //Load A[i][j:j+7] and x[j:j+7]
            __m256 A_vec0 = _mm256_loadu_ps(&A[i][j]);
            __m256 A_vec1 = _mm256_loadu_ps(&A[i][j + 4]);
            __m256 x_vec0 = _mm256_loadu_ps(&x[j]);
            __m256 x_vec1 = _mm256_loadu_ps(&x[j + 4]);

            //alpha * A[i][j:j+7] * x[j:j+7]
            __m256 alpha_Ax0 = _mm256_mul_ps(alpha_vec, _mm256_mul_ps(A_vec0, x_vec0));
            __m256 alpha_Ax1 = _mm256_mul_ps(alpha_vec, _mm256_mul_ps(A_vec1, x_vec1));

            // Combine results
            __m256 alpha_Ax_combined = _mm256_add_ps(alpha_Ax0, alpha_Ax1);

            //w[i] - beta + alpha * A[i][j:j+7] * x[j:j+7]
            __m256 result = _mm256_add_ps(_mm256_sub_ps(w_vec, beta_vec), alpha_Ax_combined);

            // Store the result back to w[i]
            _mm256_storeu_ps(&w[i], result);

            // Move to the next set
            w_vec = _mm256_loadu_ps(&w[i + 8]);
        }
    }
}
