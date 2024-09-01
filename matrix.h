#ifndef MATRIX_H_
#define MATRIX_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef MATRIX_MALLOC
#include <stdlib.h>
#define MATRIX_MALLOC malloc
#endif // MATRIX_MALLOC

#ifndef MATRIX_ASSERT
#include <assert.h>
#define MATRIX_ASSERT assert
#endif // MATRIX_ASSERT

typedef struct
{
    size_t rows;
    size_t cols;
    float *elements;
} Matrix;

#define MAT_AT(m, i, j) (m).elements[(i) * (m).cols + (j)]

float randf(void);
float sigmoidf(float x);

Matrix mat_alloc(size_t rows, size_t cols);
void mat_rand(Matrix m, float start, float end);
void mat_mul(Matrix dst, Matrix a, Matrix b);
Matrix mat_row(Matrix m, size_t row);
void mat_copy(Matrix dst, Matrix src);
void mat_sum(Matrix dst, Matrix m);
void mat_sigmoid(Matrix m);
void mat_print(Matrix m, char *name);
void mat_fill(Matrix m, float val);
#define MAT_PRINT(m) mat_print(m, #m)

#endif // MATRIX_H_

#ifdef MATRIX_IMPLEMENTATION

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

Matrix mat_alloc(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.elements = MATRIX_MALLOC(sizeof(*m.elements) * rows * cols);
    MATRIX_ASSERT(m.elements != NULL);
    return m;
}

float randf(void)
{
    return rand() / (float)RAND_MAX;
}

void mat_rand(Matrix m, float start, float end)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = start + (end - start) * randf();
        }
    }
}

void mat_mul(Matrix dst, Matrix a, Matrix b)
{
    MATRIX_ASSERT(a.cols == b.rows);
    MATRIX_ASSERT(dst.rows == a.rows);
    MATRIX_ASSERT(dst.cols == b.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Matrix mat_row(Matrix m, size_t row)
{
    return (Matrix){.rows = 1, .cols = m.cols, .elements = &MAT_AT(m, row, 0)};
}
void mat_copy(Matrix dst, Matrix src)
{
    MATRIX_ASSERT(dst.rows == src.rows);
    MATRIX_ASSERT(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_sum(Matrix dst, Matrix a)
{
    MATRIX_ASSERT(dst.rows == a.rows);
    MATRIX_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_sigmoid(Matrix m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}
void mat_print(Matrix m, char *name)
{
    printf("%s = [\n", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("    [");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("  %f", MAT_AT(m, i, j));
        }
        printf(" ]\n");
    }
    printf("]\n");
}

void mat_fill(Matrix m, float val)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = val;
        }
    }
}

#endif // MATRIX_IMPLEMENTATION