#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int LEFT_ROWS = 0;
int LEFT_COLUMNS = 0;
int RIGHT_ROWS = 0;
int RIGHT_COLUMNS = 0;

void transpose_matrix(double* matrix, int rows, int cols) {
    double* temp = (double*) malloc(rows * cols * sizeof(double));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp[j * rows + i] = matrix[i * cols + j];
        }
    }
    
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            matrix[i * rows + j] = temp[i * rows + j];
        }
    }
    
    free(temp);
}

void print_matrix(double* matrix, int rows, int cols) {
    printf("Print matrix:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%lf\t", matrix[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
}

void matrixMultiply(double* A, double* B, double* C) {
    for (int i = 0; i < LEFT_ROWS; i++) {
        for (int j = 0; j < RIGHT_COLUMNS; j++) {
            C[i * RIGHT_COLUMNS+j] = 0.0;
            for (int k = 0; k < LEFT_COLUMNS; k++) {
                C[i * RIGHT_COLUMNS + j] += A[i * LEFT_COLUMNS + k] * B[k * RIGHT_COLUMNS + j];
            }
        }
    }
}

void parallelByRows(double* A, double* B, double* C, int rank, int size) {
    // Переменные для вычисления времени работы алгоритма
    double end_time;
    double start_time = MPI_Wtime();

    // Следующий ранг, куда передаем группу строк матрицы B
    int next_rank = (rank + 1) % size;
    // Предыдущий ранг, из которого получаем группу строк матрицы B 
    int prev_rank = (rank + size - 1) % size;

    int rowsA_per_task = LEFT_ROWS / size; // определяем число строк на каждый процесс
    int rowsB_per_task = RIGHT_ROWS / size; // определяем число строк на каждый процесс
    int extra_Arows = LEFT_ROWS % size; // определяем число оставшихся строк
    int extra_Brows = RIGHT_ROWS % size; // определяем число оставшихся строк

    /*
        countsA, countsB, countsC содержат количество элементов в локальных матрицах
        displsA, displsB, displsC содержат количество элементов в локальных матрицах перед текущей локальной матрицей
    */
    
    int* countsA = (int*)malloc(size * sizeof(int)); // массив для хранения строк матрицы A, передаваемых каждому процессу
    int* displsA = (int*)malloc(size * sizeof(int)); // массив для хранения смещений для каждого процесса

    int* countsB = (int*)malloc(size * sizeof(int)); // массив для хранения строк матрицы B, передаваемых каждому процессу
    int* displsB = (int*)malloc(size * sizeof(int)); // массив для хранения смещений для каждого процесса

    int* countsC = NULL; // массив для хранения строк результирующей матрицы, передаваемых каждому процессу
    int* displsC = NULL; // массив для хранения смещений для каждого процесса
    // получаем информацию о числе строк на каждый процесс (MPI_ALLgather записывает в counts[rank] значение row_per_task)
    MPI_Allgather(&rowsA_per_task, 1, MPI_INT, countsA, 1, MPI_INT, MPI_COMM_WORLD);
    // получаем информацию о числе столбцов на каждый процесс (MPI_ALLgather записывает в counts[rank] значение column_per_task)
    MPI_Allgather(&rowsB_per_task, 1, MPI_INT, countsB, 1, MPI_INT, MPI_COMM_WORLD);

    // определяем смещения и количество строк матрицы A, передаваемых каждому процессу
    for (int i = 0, disp = 0; i < size; ++i) {
        displsA[i] = disp;
        if (extra_Arows > 0) { // если есть оставшиеся строки, добавляем их к процессу
            countsA[i]++;
            extra_Arows--;
        }
        countsA[i] *= LEFT_COLUMNS;
        disp += countsA[i];
    }

    // определяем смещения и количество строк матрицы B, передаваемых каждому процессу
    for (int i = 0, disp = 0; i < size; ++i) {
        displsB[i] = disp;
        if (extra_Brows > 0) {
            countsB[i]++;
            extra_Brows--;
        }
        countsB[i] *= RIGHT_COLUMNS;
        disp += countsB[i];
    }

    if (rank == 0) {
        countsC = (int*)malloc(size * sizeof(int));
        displsC = (int*)malloc(size * sizeof(int));
        for(int i = 0; i < size; i++) {
            countsC[i] = countsA[i] / LEFT_COLUMNS * RIGHT_COLUMNS;
        }
        
        displsC[0] = 0;
        for(int i = 1; i < size; i++) {
            displsC[i] = displsC[i - 1] + countsC[i - 1];
        }   
    }

    int maxRowsInTask = 0;

    for(int i = 0; i < size; i++) {
        if (countsB[i] > maxRowsInTask) {
            maxRowsInTask = countsB[i];
        }
    }

    double* local_A = (double*)malloc(countsA[rank] * sizeof(double)); // локальная матрица A, которую получает текущий процесс
    double* local_B = (double*)malloc(maxRowsInTask * sizeof(double)); // Задаем максимальный размер, чтобы MPI корректно перезаписывал значения
    double* local_C = (double*)malloc(countsA[rank] / LEFT_COLUMNS * RIGHT_COLUMNS * sizeof(double)); // кусок результирующей матрицы
    for(int i = 0; i < countsA[rank] / LEFT_COLUMNS * RIGHT_COLUMNS; i++) {
        local_C[i] = 0;
    }
    // передаем каждому процессу его часть матрицы A
    MPI_Scatterv(A, countsA, displsA, MPI_DOUBLE, local_A, countsA[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // передаем каждому процессу его часть матрицы B (rank 0 => 0 строка, 1 => 1, etc..)
    MPI_Scatterv(B, countsB, displsB, MPI_DOUBLE, local_B, countsB[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int iteration = 0; iteration < size; ++iteration) {
        int dataFromRank = (rank + size - iteration) % size;
        int startFromRow = displsB[dataFromRank] / RIGHT_COLUMNS;
        int currentRowCount = countsB[dataFromRank] / RIGHT_COLUMNS;
        
        for (int i = 0; i < countsA[rank] / LEFT_COLUMNS; ++i) { // Проходимся по строке левой матрицы
            for (int j = 0; j < currentRowCount; ++j) { // Проходимся по строке правой матрицы
                for (int k = 0; k < RIGHT_COLUMNS; ++k) {
                    local_C[i * RIGHT_COLUMNS + k] += local_A[i * LEFT_COLUMNS + (startFromRow + j) % LEFT_COLUMNS] * local_B[j * RIGHT_COLUMNS + k];
                }
            }
        }
        MPI_Sendrecv_replace(local_B, maxRowsInTask, MPI_DOUBLE, next_rank, 0, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Gatherv(local_C, countsA[rank] / LEFT_COLUMNS * RIGHT_COLUMNS, MPI_DOUBLE, C, countsC, displsC, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Runtime = %lf\n", end_time - start_time);
    }
    free(local_A);
    free(local_B);
    free(local_C);
    free(countsA);
    free(displsA);
    free(countsB);
    free(displsB);
    if (rank == 0) {
        free(countsC);
        free(displsC);
    }
}

void parallelByColumns(double* A, double* B, double* C, int rank, int size) {
    // Переменные для вычисления времени работы алгоритма
    double end_time;
    double start_time = MPI_Wtime();

    // Следующий ранг, куда передаем группу столбцов матрицы B
    int next_rank = (rank + 1) % size;
    // Предыдущий ранг, из которого получаем группу столбцов матрицы B 
    int prev_rank = (rank + size - 1) % size;

    int rows_per_task = LEFT_ROWS / size; // определяем число строк на каждый процесс
    int columns_per_task = RIGHT_COLUMNS / size; // определяем число столбцов на каждый процесс
    int extra_rows = LEFT_ROWS % size; // определяем число оставшихся строк
    int extra_columns = RIGHT_COLUMNS % size; // определяем число оставшихся столбцов

    if (rank == 0) { // Транспонируем матрицу для удобства передачи столбцов
        transpose_matrix(B, RIGHT_ROWS, RIGHT_COLUMNS);
    }

    /*
        countsA, countsB, countsC содержат количество элементов в локальных матрицах
        displsA, displsB, displsC содержат количество элементов в локальных матрицах перед текущей локальной матрицей
    */
    
    int* countsA = (int*)malloc(size * sizeof(int)); // массив для хранения строк, передаваемых каждому процессу
    int* displsA = (int*)malloc(size * sizeof(int)); // массив для хранения смещений для каждого процесса

    int* countsB = (int*)malloc(size * sizeof(int)); // массив для хранения строк транспонированной матрицы (столбцов оригинальной), передаваемых каждому процессу
    int* displsB = (int*)malloc(size * sizeof(int)); // массив для хранения смещений для каждого процесса

    int* countsC = NULL; // массив для хранения строк результирующей матрицы, передаваемых каждому процессу
    int* displsC = NULL; // массив для хранения смещений для каждого процесса
    // получаем информацию о числе строк на каждый процесс (MPI_ALLgather записывает в counts[rank] значение row_per_task)
    MPI_Allgather(&rows_per_task, 1, MPI_INT, countsA, 1, MPI_INT, MPI_COMM_WORLD);
    // получаем информацию о числе столбцов на каждый процесс (MPI_ALLgather записывает в counts[rank] значение column_per_task)
    MPI_Allgather(&columns_per_task, 1, MPI_INT, countsB, 1, MPI_INT, MPI_COMM_WORLD);

    // определяем смещения и количество строк матрицы A, передаваемых каждому процессу
    for (int i = 0, disp = 0; i < size; ++i) {
        displsA[i] = disp;
        if (extra_rows > 0) { // если есть оставшиеся строки, добавляем их к процессу
            countsA[i]++;
            extra_rows--;
        }
        countsA[i] *= LEFT_COLUMNS;
        disp += countsA[i];
    }

    // определяем смещения и количество столбцов матрицы B, передаваемых каждому процессу
    for (int i = 0, disp = 0; i < size; ++i) {
        displsB[i] = disp;
        if (extra_columns > 0) {
            countsB[i]++;
            extra_columns--;
        }
        countsB[i] *= RIGHT_ROWS;
        disp += countsB[i];
    }

    if (rank == 0) {
        countsC = (int*)malloc(size * sizeof(int));
        displsC = (int*)malloc(size * sizeof(int));
        for(int i = 0; i < size; i++) {
            countsC[i] = countsA[i] / LEFT_COLUMNS * RIGHT_COLUMNS;
        }
        
        displsC[0] = 0;
        for(int i = 1; i < size; i++) {
            displsC[i] = displsC[i - 1] + countsC[i - 1];
        }   
    }

    int maxColsInTask = 0;

    for(int i = 0; i < size; i++) {
        if (countsB[i] > maxColsInTask) {
            maxColsInTask = countsB[i];
        }
    }

    double* local_A = (double*)malloc(countsA[rank] * sizeof(double)); // локальная матрица A, которую получает текущий процесс
    double* local_B = (double*)malloc(maxColsInTask * sizeof(double)); // Задаем максимальный размер, чтобы MPI корректно перезаписывал значения
    double* local_C = (double*)malloc(countsA[rank] / LEFT_COLUMNS * RIGHT_COLUMNS * sizeof(double)); // кусок результирующей матрицы
    // передаем каждому процессу его часть матрицы A
    MPI_Scatterv(A, countsA, displsA, MPI_DOUBLE, local_A, countsA[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // передаем каждому процессу его часть матрицы B (rank 0 => 0 столбец, 1 => 1, etc..)
    MPI_Scatterv(B, countsB, displsB, MPI_DOUBLE, local_B, countsB[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int iteration = 0; iteration < size; ++iteration) {
        int dataFromRank = (rank + size - iteration) % size;
        int startFromColumn = displsB[dataFromRank] / RIGHT_ROWS;
        int currentColumnsCount = countsB[dataFromRank] / RIGHT_ROWS;
        
        for (int i = 0; i < countsA[rank] / LEFT_COLUMNS; ++i) { // Проходимся по строке левой матрицы
            for (int j = 0; j < currentColumnsCount; ++j) { // Проходимся по столбцу правой матрицы
                local_C[i * RIGHT_COLUMNS + (startFromColumn + j)] = 0;
                for (int k = 0; k < RIGHT_ROWS; ++k) { // Проходимся по строке+колонке, чтобы получить один элемент в local_C
                    local_C[i * RIGHT_COLUMNS + (startFromColumn + j)] += (local_A[i * LEFT_COLUMNS + k] * local_B[j * RIGHT_ROWS + k]);
                }
            }
        }
        MPI_Sendrecv_replace(local_B, maxColsInTask, MPI_DOUBLE, next_rank, 0, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Gatherv(local_C, countsA[rank] / LEFT_COLUMNS * RIGHT_COLUMNS, MPI_DOUBLE, C, countsC, displsC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Runtime = %lf\n", end_time - start_time);
    }
    free(local_A);
    free(local_B);
    free(local_C);
    free(countsA);
    free(displsA);
    free(countsB);
    free(displsB);
    if (rank == 0) {
        free(countsC);
        free(displsC);
    }
}

void parallelByBlocks(double* A, double* B, double* C, int rank, int size) {
    // Переменные для вычисления времени работы алгоритма
    double end_time;
    double start_time = MPI_Wtime();

    // Переменные для сетки раздения матриц на блоки
    int grid_size;
    int grid_rows;
    int grid_cols;

    if (rank == 0) {
        grid_size = ceil(sqrt(size)); //исходя из количества потоков вычислим размер сетки
        grid_rows = ceil((float)LEFT_ROWS / (float)grid_size); //вычислим количество разделений по строкам
        grid_cols = ceil((float)RIGHT_COLUMNS /(float)grid_size); //вычислим количество разделений по столбцам
    }

    //передача значений сетки в каждый поток
    MPI_Bcast(&grid_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&grid_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&grid_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("%d, %d, %d\n", grid_size, grid_rows, grid_cols);

    int local_A_rows = LEFT_ROWS / grid_rows; // количество строк для блочной матрицы А
    int local_A_cols = LEFT_COLUMNS / grid_cols; // количество столбцов для блочной матрицы А
    int left_rows_number = LEFT_ROWS % grid_rows; // оставшееся количество строк А
    int left_cols_number = LEFT_COLUMNS % grid_cols; //оставшееся количество столбцов А

    double* ordered_A = NULL; //так как матрица у нас записана как вектор, то ее значения нужно упорядочить по значения в блоках это для матрицы А
    int* rows_division = (int*)malloc(grid_rows * sizeof(int)); //массив разделения на блоки по строкам для матрицы А
    int* columns_division = (int*)malloc(grid_cols * sizeof(int)); //массив разделения на блоки по столцам для матрицы А

    for (int i = 0; i < grid_rows; i++) {
        rows_division[i] = local_A_rows;
        if (left_rows_number > 0) { // если нацело не делится количество строк то в первых блок идет больше значений
            rows_division[i]++;
            left_rows_number--;
        }
    }

    for (int i = 0; i < grid_cols; i++) {
        columns_division[i] = local_A_cols;
        if (left_cols_number > 0) { // если нацело не делится количество столбцов то в первых блок идет больше значений
            columns_division[i]++;
            left_cols_number--;
        }
    }

    // формируем новый порядок значений в матрице А в соответствии с блоками
    if (rank == 0) {
        ordered_A = (double*)malloc(LEFT_ROWS * LEFT_COLUMNS * sizeof(double));

        int temp = 0; // индекс для упорядоченной матрицы
        int offset_row = 0; //смещение по строкам
        int offset_col = 0; //смещение по столбцам
        
        for (int i = 0; i < grid_rows; i++) {
            for (int j = 0; j < grid_cols; j++) {
                for (int k = 0; k < rows_division[i]; k++) {
                    for (int h = 0; h < columns_division[j]; h++) {
                        ordered_A[temp] = A[(k + offset_row) * LEFT_COLUMNS + h + offset_col]; // смешение по матрице А по количеству стоблцов в ней и по количеству элементов в блоке
                    }
                }
                offset_col = (offset_col + columns_division[j]) % LEFT_COLUMNS; // смещение по столбцам 
            }

            offset_row += rows_division[i];
        }
    }

    int* A_block_items = (int*)malloc(grid_cols * grid_rows * sizeof(int)); // массив количества элементов в блоке
    int* A_offsets = (int*)malloc(grid_cols * grid_rows * sizeof(int)); // массив смещений по вектору матрицы А
    int* B_block_items = (int*)malloc(grid_cols * grid_rows * sizeof(int)); // массив количества элементов в блок
    int* B_offsets = (int*)malloc(grid_cols * grid_rows * sizeof(int)); // массив смещений по вектору матрицы В

    // заполнение матрицы смещения и кол. элементов для матрицы А 
    int step = 0; //смещение (шаг) по вектору матрицы А
    int temp = 0; //индекс
    for (int i = 0; i < grid_rows; i++)
    {
        for (int j = 0; j < grid_cols; j++)
        {
            A_block_items[temp] = rows_division[i] * columns_division[j];
            A_offsets[temp] = step;
            step += A_block_items[temp];

            temp++;
        }
    }

    // заполнение матрицы смещения и кол. элементов для матрицы B
    step = 0; //смещение (шаг) по вектору матрицы B
    temp = 0;
    for (int i = 0; i < size; i++)
    {
        B_block_items[i] = columns_division[temp] * RIGHT_COLUMNS; // делим матрицу B в зависимости разделения матрицы А так как перемножение матриц должно быть с равным значением colmA = rowB
        B_offsets[i] = step;

        temp = (temp + 1) % grid_cols;
        if (temp == 0)
            step = 0;
        else
            step += B_block_items[i];
    }

    double* A_block = (double*)malloc(A_block_items[rank] * sizeof(double));
    double* B_block = (double*)malloc(B_block_items[rank] * sizeof(double));
    double* local_C = (double*)calloc(LEFT_ROWS * RIGHT_COLUMNS, sizeof(double));

    // передаем каждому процессу его часть матрицы A
    MPI_Scatterv(ordered_A, A_block_items, A_offsets, MPI_DOUBLE, A_block, A_block_items[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // передаем каждому процессу его часть матрицы B
    MPI_Scatterv(B, B_block_items, B_offsets, MPI_DOUBLE, B_block, B_block_items[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sum = 0;
    int local_b_rows = B_block_items[rank] / RIGHT_COLUMNS;
    int local_a_rows = A_block_items[rank] / local_b_rows;
    
    int offset = 0;
    int rows_count = rank;

    while (B_offsets[rows_count] != 0)
        rows_count--;

    for (int i = 0; i < rows_count; i++)
    {
        if (B_offsets[i] == 0)
            offset += A_block_items[i] / (B_block_items[i] / RIGHT_COLUMNS); //высчитываем сдвиг в результатирующей матрице исходят из сдвигов в матрицу В
    }

    //считаем по алгоритму фокса
    for (int i = 0; i < local_a_rows; i++)
    {
        for (int j = 0; j < RIGHT_COLUMNS; j++)
        {
            for (int k = 0; k < local_b_rows; k++)
            {
                local_sum += A_block[i * local_b_rows + k] * B_block[k * RIGHT_COLUMNS + j];
            }

            local_C[(i + offset) * RIGHT_COLUMNS + j] = local_sum;
            local_sum = 0;
        }
    }

    //Объединяем все локальные матрицы в основную матрицу С 
    MPI_Reduce(local_C, &C, LEFT_ROWS * RIGHT_COLUMNS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Runtime = %lf\n", end_time - start_time);
    }
    free(rows_division);
    free(columns_division);
    free(local_C);
    free(A_block);
    free(B_block);
    free(A_block_items);
    free(B_block_items);
    free(A_offsets);
    free(B_offsets);
    if (ordered_A)
        free(ordered_A);
}


int main(int argc, char** argv) {
    int i, j, k;
    int rank, size;
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int l = atoi(argv[1]);
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    LEFT_ROWS = l;
    LEFT_COLUMNS = m;
    RIGHT_ROWS = m;
    RIGHT_COLUMNS = n;
    // инициализируем матрицы A, B и C
    if (rank == 0) {
        A = (double*)malloc(LEFT_ROWS * LEFT_COLUMNS * sizeof(double));
        B = (double*)malloc(RIGHT_ROWS * RIGHT_COLUMNS * sizeof(double));
        C = (double*)malloc(LEFT_ROWS * RIGHT_COLUMNS * sizeof(double));
        for (i = 0; i < LEFT_ROWS; i++) {
            for (j = 0; j < LEFT_COLUMNS; j++) {
                A[i * LEFT_COLUMNS + j] = i * LEFT_COLUMNS + j;
            }
        }
        for (i = 0; i < RIGHT_ROWS; i++) {
            for (j = 0; j < RIGHT_COLUMNS; j++) {
                B[i * RIGHT_COLUMNS + j] = i * RIGHT_COLUMNS + j;
            }
        }
        for (i = 0; i < LEFT_ROWS; i++) {
            for (j = 0; j < RIGHT_COLUMNS; j++) {
                C[i * RIGHT_COLUMNS + j] = 0;
            }
        }
    
       /* printf("Initial matrices:\n");
        print_matrix(A, LEFT_ROWS, LEFT_COLUMNS);
        print_matrix(B, RIGHT_ROWS, RIGHT_COLUMNS);
        print_matrix(C, LEFT_ROWS, RIGHT_COLUMNS);*/
//        matrixMultiply(A, B, C);
    }

    parallelByColumns(A, B, C, rank, size);
    //parallelByRows(A, B, C, rank, size);

    //parallelByBlocks(A, B, C, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    // выводим результаты
    if (rank == -1) {
        printf("Result matrix C:\n");
        for (i = 0; i < LEFT_ROWS; i++) {
            for (j = 0; j < RIGHT_COLUMNS; j++) {
                printf("%lf\t", C[i * RIGHT_COLUMNS + j]);
            }
            printf("\n");
        }
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
