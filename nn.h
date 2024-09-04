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
void mat_fill(Matrix m, float val);
Matrix mat_sub(Matrix m, size_t i, size_t j, size_t width, size_t height);

void mat_print(Matrix m, char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct
{
    size_t count; // The number of layers
    Matrix *weights;
    Matrix *biases;
    Matrix *activations; // The number of activation layers is count + 1
} NN;

#define NN_INPUT(nn) (nn).activations[0]
#define NN_OUTPUT(nn) (nn).activations[(nn).count]

NN nn_alloc(size_t *layers, size_t count);
void nn_zero(NN nn);
void nn_rand(NN nn, float start, float end);
void nn_forward(NN nn);
float nn_cost(NN nn, Matrix x, Matrix y);
void nn_finite_diff(NN nn, NN grads, float eps, Matrix x, Matrix y);
void nn_backprop(NN nn, NN grads, Matrix x, Matrix y);
void nn_learn(NN nn, NN grads, float rate);

void nn_print(NN nn, char *name);
#define NN_PRINT(nn) nn_print((nn), #nn);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

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
void mat_print(Matrix m, char *name, size_t padding)
{
    printf("%*s", (int)padding, "");
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s    [", (int)padding, "");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%*s  %f", (int)padding, "", MAT_AT(m, i, j));
        }
        printf("%*s ]\n", (int)padding, "");
    }
    printf("%*s]\n", (int)padding, "");
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

void nn_zero(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_fill(nn.weights[i], 0);
        mat_fill(nn.biases[i], 0);
        mat_fill(nn.activations[i], 0);
    }
    mat_fill(NN_OUTPUT(nn), 0);
}

void nn_rand(NN nn, float start, float end)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.weights[i], start, end);
        mat_rand(nn.biases[i], start, end);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_mul(nn.activations[i + 1], nn.activations[i], nn.weights[i]);
        mat_sum(nn.activations[i + 1], nn.biases[i]);
        mat_sigmoid(nn.activations[i + 1]);
    }
}

float nn_cost(NN nn, Matrix x, Matrix y)
{
    NN_ASSERT(x.rows == y.rows);
    NN_ASSERT(NN_INPUT(nn).cols == x.cols);
    NN_ASSERT(y.cols == NN_OUTPUT(nn).cols);
    const size_t n = x.rows;
    float cost = 0.f;
    for (size_t i = 0; i < n; i++)
    {
        const Matrix xi = mat_row(x, i);
        const Matrix yi = mat_row(y, i);
        mat_copy(NN_INPUT(nn), xi);
        nn_forward(nn);
        size_t q = y.cols;
        for (size_t j = 0; j < q; j++)
        {
            const float diff = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(yi, 0, j);
            cost += diff * diff;
        }
    }

    return cost / n;
}

void nn_finite_diff(NN nn, NN grads, float eps, Matrix x, Matrix y)
{
    float saved;
    float cost = nn_cost(nn, x, y);
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.weights[i].rows; j++)
        {
            for (size_t k = 0; k < nn.weights[i].cols; k++)
            {
                saved = MAT_AT(nn.weights[i], j, k);
                MAT_AT(nn.weights[i], j, k) += eps;
                MAT_AT(grads.weights[i], j, k) = (nn_cost(nn, x, y) - cost) / eps;
                MAT_AT(nn.weights[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.biases[i].rows; j++)
        {
            for (size_t k = 0; k < nn.biases[i].cols; k++)
            {
                saved = MAT_AT(nn.biases[i], j, k);
                MAT_AT(nn.biases[i], j, k) += eps;
                MAT_AT(grads.biases[i], j, k) = (nn_cost(nn, x, y) - cost) / eps;
                MAT_AT(nn.biases[i], j, k) = saved;
            }
        }
    }
}

void nn_backprop(NN nn, NN grads, Matrix x, Matrix y)
{
    NN_ASSERT(x.rows == y.rows);
    NN_ASSERT(NN_INPUT(nn).cols == x.cols);
    NN_ASSERT(y.cols == NN_OUTPUT(nn).cols);

    nn_zero(grads);

    size_t n = y.rows;

    for (size_t i = 0; i < n; i++)
    {
        // i: sample index
        const Matrix xi = mat_row(x, i);
        mat_copy(NN_INPUT(nn), xi);
        nn_forward(nn);

        for (size_t j = 0; j <= grads.count; j++)
        {
            mat_fill(grads.activations[j], 0);
        }

        for (size_t j = 0; j < y.cols; j++)
        {
            // j: output index
            MAT_AT(NN_OUTPUT(grads), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, i, j);
        }

        for (size_t l = nn.count; l > 0; l--)
        {
            // l: layer index
            for (size_t j = 0; j < nn.activations[l].cols; j++)
            {
                // j: neuron index of layer l
                const float a = MAT_AT(nn.activations[l], 0, j);
                const float da = MAT_AT(grads.activations[l], 0, j);
                MAT_AT(grads.biases[l - 1], 0, j) += 2 * da * a * (1 - a);
                for (size_t k = 0; k < nn.activations[l - 1].cols; k++)
                {
                    // k: neuron index of layer l - 1
                    const float prev_a = MAT_AT(nn.activations[l - 1], 0, k);
                    MAT_AT(grads.weights[l - 1], k, j) += 2 * da * a * (1 - a) * prev_a;
                    const float w = MAT_AT(nn.weights[l - 1], k, j);
                    MAT_AT(grads.activations[l - 1], 0, k) += 2 * da * a * (1 - a) * w;
                }
            }
        }
    }

    for (size_t i = 0; i < grads.count; i++)
    {
        for (size_t j = 0; j < grads.weights[i].rows; j++)
        {
            for (size_t k = 0; k < grads.weights[i].cols; k++)
            {
                MAT_AT(grads.weights[i], j, k) /= n;
            }
        }
        for (size_t j = 0; j < grads.biases[i].rows; j++)
        {
            for (size_t k = 0; k < grads.biases[i].cols; k++)
            {
                MAT_AT(grads.biases[i], j, k) /= n;
            }
        }
    }
}

void nn_learn(NN nn, NN grads, float rate);
void nn_learn(NN nn, NN grads, float rate)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.weights[i].rows; j++)
        {
            for (size_t k = 0; k < nn.weights[i].cols; k++)
            {
                MAT_AT(nn.weights[i], j, k) -= rate * MAT_AT(grads.weights[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.biases[i].rows; j++)
        {
            for (size_t k = 0; k < nn.biases[i].cols; k++)
            {
                MAT_AT(nn.biases[i], j, k) -= rate * MAT_AT(grads.biases[i], j, k);
            }
        }
    }
}

void nn_print(NN nn, char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buf, sizeof(buf), "weights[%zu]", i);
        mat_print(nn.weights[i], buf, 2);
        snprintf(buf, sizeof(buf), "biases[%zu]", i);
        mat_print(nn.biases[i], buf, 2);
    }
    printf("]\n");
}

#endif // NN_IMPLEMENTATION