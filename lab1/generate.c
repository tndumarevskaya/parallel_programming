#include <stdlib.h>
#include <stdio.h>
#include <time.h>
 
#define N 10000
#define TS 100
int main() {
    FILE* file = fopen("input3.txt", "w+");

    srand(time(NULL));

    fprintf(file, "6.6743e-11 %d %d\n", N, TS);

    for (int i = 0; i < N; i++) {
        int mass = rand() % 1000;
        fprintf(file, "%d\n", mass);
        int x = rand() % 1000;
        int y = rand() % 1000;
        fprintf(file, "%d %d\n", x, y);
        fprintf(file, "0 0\n");
    }

    fclose(file);
};
