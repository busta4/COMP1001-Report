
int vec1Check = 0;
int vec2Check = 0;
float epsilon = 0.01;

// Validation for routine1_vec
for (int t = 0; t < TIMES1; t++) {
    routine1_vec(alpha, beta);

    for (int i = 0; i < M; i++) {
        if (fabs(y[i] - r1Checker[i]) > epsilon) {
            printf("Validation failed for Routine1_vec at index %d\n", i);
            return -1;
        }
    }
}
printf("Routine1_vec passed.\n");


// Validation for routine2_vec
for (int t = 0; t < TIMES2; t++) {
    routine2_vec(alpha, beta);

    for (int i = 0; i < N; i++) {
        if (fabs(w[i] - r2Checker[i]) < epsilon) {
            vec2Check += 1;
        }
    }
}
if (vec2Check == TIMES2 * N) {
    printf("Routine2_vec passed.\n");
}
else {
    printf("Validation failed for Routine2_vec\n");
}
