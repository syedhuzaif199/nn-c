#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *elements;
} Matrix;

#define MAT_AT(m, i, j) (m).elements[(i) * (m).stride + (j)]

#define ARRAY_LEN(x) sizeof((x)) / sizeof((x)[0])

typedef struct
{
    size_t count; // The number of layers
    Matrix *weights;
    Matrix *biases;
    Matrix *activations; // The number of activation layers is count + 1
} NN;

float randf(void);
float sigmoidf(float x);

Matrix mat_alloc(size_t rows, size_t cols);
Matrix mat_from_data(size_t rows, size_t cols, float *data);
void mat_rand(Matrix m, float start, float end);
void mat_mul(Matrix dst, Matrix a, Matrix b);
Matrix mat_row(Matrix m, size_t row);
void mat_copy(Matrix dst, Matrix src);
void mat_sum(Matrix dst, Matrix m);
void mat_sigmoid(Matrix m);
void mat_print(Matrix m, char *name);
void mat_fill(Matrix m, float val);
Matrix mat_sub(Matrix m, size_t i, size_t j, size_t width, size_t height);
NN nn_alloc(size_t *layers, size_t count);
#define MAT_PRINT(m) mat_print(m, #m)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

NN nn_alloc(size_t *layers, size_t count)
{
    NN_ASSERT(count > 0);
    NN nn;
    nn.count = count - 1;
    nn.weights = (Matrix *)NN_MALLOC(sizeof(*nn.weights) * (nn.count));
    NN_ASSERT(nn.weights != NULL);

    nn.biases = (Matrix *)NN_MALLOC(sizeof(*nn.biases) * nn.count);
    NN_ASSERT(nn.biases != NULL);

    nn.activations = (Matrix *)NN_MALLOC(sizeof(*nn.activations) * count);
    NN_ASSERT(nn.activations != NULL);

    nn.activations[0] = mat_alloc(1, layers[0]);

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.weights[i] = mat_alloc(nn.activations[i].cols, layers[i + 1]);
        nn.biases[i] = mat_alloc(1, layers[i + 1]);
        nn.activations[i + 1] = mat_alloc(1, layers[i + 1]);
    }

    return nn;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

Matrix mat_alloc(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.elements = NN_MALLOC(sizeof(*m.elements) * rows * cols);
    NN_ASSERT(m.elements != NULL);
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
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);
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
    return (Matrix){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .elements = &MAT_AT(m, row, 0),
    };
}
void mat_copy(Matrix dst, Matrix src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);
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
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
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
Matrix mat_sub(Matrix m, size_t i, size_t j, size_t rows, size_t cols)
{
    Matrix sub = {
        .rows = rows,
        .cols = cols,
        .stride = m.stride,
        .elements = &MAT_AT(m, i, j),
    };

    return sub;
}
Matrix mat_from_data(size_t rows, size_t cols, float *data)
{
    return (Matrix){
        .rows = rows,
        .cols = cols,
        .stride = cols,
        .elements = data,
    };
}
#endif // NN_IMPLEMENTATION