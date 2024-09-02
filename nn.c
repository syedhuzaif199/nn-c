#include <time.h>

#define NN_IMPLEMENTATION

#include "nn.h"

float train_data[] = {
    0, 0, 0, //
    0, 1, 1, //
    1, 0, 1, //
    1, 1, 0, //
};

int main()
{
    srand(time(0));
    const Matrix train_matrix = mat_from_data(4, 3, train_data);
    const Matrix train_x = mat_sub(train_matrix, 0, 0, 4, 2);
    const Matrix train_y = mat_sub(train_matrix, 0, 2, 4, 1);

    size_t layers[] = {2, 2, 1};

    NN nn = nn_alloc(layers, ARRAY_LEN(layers));
    NN grads = nn_alloc(layers, ARRAY_LEN(layers));
    nn_rand(nn, 1, 2);

    printf("Cost = %f\n", nn_cost(nn,
                                  train_x,
                                  train_y));

    const float eps = 1e-1;
    const float rate = 1e-1;

    for (size_t i = 0; i < 100000; i++)
    {
        nn_finite_diff(nn, grads, eps, train_x, train_y);
        nn_learn(nn, grads, rate);
        printf("%zu: Cost = %f\n", i, nn_cost(nn, train_x, train_y));
    }

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;

            nn_forward(nn);
            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    return 0;
}