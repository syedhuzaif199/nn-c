#ifndef MATRIX_H_
#define MATRIX_H_

#include <stddef.h>
#include <stdio.h>

#ifndef MATRIX_MALLOC
#include <stdlib.h>
#define MATRIX_MALLOC malloc
#endif // MATRIX_MALLOC

#ifndef MATRIX_ASSERT
#include <assert.h> 
#define MATRIX_ASSERT assert
#endif //MATRIX_ASSERT

typedef struct {
    size_t rows;
    size_t cols;
    float *elements;
} Matrix;

#define MAT_AT(m, i, j) (m).elements[(i) * (m).cols + (j)]

float rand_float(void);

Matrix mat_alloc(size_t rows, size_t cols);
void mat_rand(Matrix m, float start, float end);
void mat_mul(Matrix dst, Matrix a, Matrix b); 
void mat_sum(Matrix dst, Matrix m);
void mat_print(Matrix m);
void mat_fill(Matrix m, float val);

#endif // MATRIX_H_

#ifdef MATRIX_IMPLEMENTATION

Matrix mat_alloc(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.elements = MATRIX_MALLOC(sizeof(*m.elements) * rows * cols);
    MATRIX_ASSERT(m.elements != NULL);
    return m;
}

float rand_float(void) {
    return rand() / (float) RAND_MAX;
}

void mat_rand(Matrix m, float start, float end) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = start + (end - start) * rand_float();
        }
    }
}

void mat_mul(Matrix dst, Matrix a, Matrix b) {
    MATRIX_ASSERT(a.cols == b.rows);
    MATRIX_ASSERT(dst.rows == a.rows);
    MATRIX_ASSERT(dst.cols == b.cols);
    for(size_t i = 0; i < dst.rows; i++) {
        for(size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < a.cols; k++) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}
void mat_sum(Matrix dst, Matrix a) {
    MATRIX_ASSERT(dst.rows == a.rows);
    MATRIX_ASSERT(dst.cols == a.cols);
    for(size_t i = 0; i < dst.rows; i++) {
        for(size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}
void mat_print(Matrix m) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
}

void mat_fill(Matrix m, float val) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = val;
        }
    }
}

#endif // MATRIX_IMPLEMENTATION 