#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "nn.h"

#define LINES 10000
#define VALUES_PER_LINE 785
#define TOTAL_VALUES (LINES * VALUES_PER_LINE)

float *load_mnist_data_raw()
{
    // Open the CSV file for reading
    FILE *file = fopen("mnist.csv", "r");
    if (file == NULL)
    {
        perror("Error opening file");
        return NULL;
    }

    // Allocate memory for the array to hold all the values
    float *all_values = (float *)malloc(TOTAL_VALUES * sizeof(int));
    if (all_values == NULL)
    {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }

    int value, index = 0;
    char line[16000]; // Adjust size if needed, based on expected line length

    // Read each line from the file
    while (fgets(line, sizeof(line), file) != NULL && index < TOTAL_VALUES)
    {
        char *ptr = line;
        while (*ptr)
        {
            if (sscanf(ptr, "%d", &value) == 1)
            {
                all_values[index++] = (float)value;
            }

            // Move to the next integer value in the line
            while (*ptr && *ptr != ',')
                ptr++; // Skip to the next comma
            if (*ptr)
                ptr++; // Skip past the comma
        }
    }

    // Close the file
    fclose(file);

    return all_values;
}

int load_mnist_data_preprocessed(Matrix *x, Matrix *y)
{
    float *img_data = load_mnist_data_raw();
    for (size_t i = 0; i < LINES; i++)
    {
        for (size_t j = 1; j < VALUES_PER_LINE; j++)
        {
            img_data[i * VALUES_PER_LINE + j] /= 255.0f;
        }
    }
    if (img_data == NULL)
    {
        return 1;
    }
    Matrix img_matrix = mat_from_data(LINES, VALUES_PER_LINE, img_data);
    *x = mat_sub(img_matrix, 0, 1, LINES, VALUES_PER_LINE - 1);
    float img_y_data[LINES * 10] = {0};
    for (size_t i = 0; i < LINES; i++)
    {
        img_y_data[i * 10 + (size_t)img_data[i * VALUES_PER_LINE]] = 1;
    }
    *y = mat_from_data(LINES, 10, img_y_data);
    return 0;
}

void print_image(float *data)
{
    for (size_t i = 0; i < 28; i++)
    {
        for (size_t j = 0; j < 28; j++)
        {
            printf("%c", data[i * 28 + j] > 0.5 ? '#' : '.');
        }
        printf("\n");
    }
}
