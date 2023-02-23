#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define DT 0.05
#define NUM_THREADS 16

typedef struct
{
    double x, y;
} vector;

int counter = 0;
pthread_mutex_t mutex;
pthread_cond_t cond_var;

char outputFilename[255];

int bodies, timeSteps;
double *masses, GravConstant;
vector *positions, *velocities, *accelerations;

void initOutputFile() {
    time_t rawtime;

    time (&rawtime);
    sprintf(outputFilename, "output-%dBodies-%d-Steps-%d-Timestamp.txt", bodies, timeSteps, (int)time(NULL));

    char *p = outputFilename;
    for (; *p; ++p)
    {
        if (*p == ' ')
            *p = '_';
    }

    FILE *S1;
    S1 = fopen(outputFilename, "w");
    fprintf(S1, "Body   :     x              y           vx              vy   ");
    fclose(S1);
}

void writeCurrentStateToFile(int step) {
    FILE *S1;
    S1 = fopen(outputFilename, "a");
    fprintf(S1, "Cycle %d\n", step);
    printf("Cycle %d\n", step);
    for (int i = 0; i < bodies; i++) {
        fprintf(S1, "Body %d : %lf\t%lf\t%lf\t%lf\n", i + 1, positions[i].x, positions[i].y, velocities[i].x, velocities[i].y);
        printf("Body %d : %lf\t%lf\t%lf\t%lf\n", i + 1, positions[i].x, positions[i].y, velocities[i].x, velocities[i].y);
    }
    fclose(S1);
}

vector addVectors(vector a, vector b)
{
    vector c = {a.x + b.x, a.y + b.y};

    return c;
}

vector scaleVector(double b, vector a)
{
    vector c = {b * a.x, b * a.y};

    return c;
}

vector subtractVectors(vector a, vector b)
{
    vector c = {a.x - b.x, a.y - b.y};

    return c;
}

double mod(vector a)
{
    return sqrt(a.x * a.x + a.y * a.y);
}

void initiateSystem(char *fileName)
{
    int i;
    FILE *fp = fopen(fileName, "r");

    fscanf(fp, "%lf%d%d", &GravConstant, &bodies, &timeSteps);

    masses = (double *)malloc(bodies * sizeof(double));
    positions = (vector *)malloc(bodies * sizeof(vector));
    velocities = (vector *)malloc(bodies * sizeof(vector));
    accelerations = (vector *)malloc(bodies * sizeof(vector));

    for (i = 0; i < bodies; i++)
    {
        fscanf(fp, "%lf", &masses[i]);
        fscanf(fp, "%lf%lf", &positions[i].x, &positions[i].y);
        fscanf(fp, "%lf%lf", &velocities[i].x, &velocities[i].y);
    }

    fclose(fp);
}

void resolveCollisions(int i, int j) {
    if (positions[i].x == positions[j].x && positions[i].y == positions[j].y) {
        vector temp = velocities[i];
        velocities[i] = velocities[j];
        velocities[j] = temp;
    }
}

void computeAccelerations(int i, int j) {
    if (i != j) {
        accelerations[i] = addVectors(accelerations[i], scaleVector(GravConstant * masses[j] / pow(mod(subtractVectors(positions[i], positions[j])), 3), subtractVectors(positions[j], positions[i])));
    }
}

void computeVelocities(int i) {
    velocities[i] = addVectors(velocities[i], scaleVector(DT, accelerations[i]));
}

void computePositions(int i) {
    positions[i] = addVectors(positions[i], scaleVector(DT,velocities[i]));
}

void *simulate(void *rank) {
    long my_rank = (long)rank;
    for (int step = 0; step < timeSteps; step++)
    {
        for (int i = my_rank; i < bodies; i = i + NUM_THREADS)
        {
            accelerations[i].x = 0;
            accelerations[i].y = 0;
            for (int j = 0; j < bodies; j++)
            {
                computeAccelerations(i, j);
            }
        }
        
        pthread_mutex_lock(&mutex);
        counter++;
        if (counter == NUM_THREADS)
        {
            counter = 0;
            pthread_cond_broadcast(&cond_var);
        }
        else
        {
            while (pthread_cond_wait(&cond_var, &mutex) != 0)
            {
            }
        }
        pthread_mutex_unlock(&mutex);

        for (int i = my_rank; i < bodies; i = i + NUM_THREADS)
        {
            computePositions(i);
            computeVelocities(i);
        }

        for (int i = my_rank; i < bodies; i = i + NUM_THREADS)
        {
            for (int j = i + 1; j < bodies; j++)
            {
                resolveCollisions(i, j);
            }
        }

        pthread_mutex_lock(&mutex);
        counter++;
        if (counter == NUM_THREADS)
        {
            writeCurrentStateToFile(step);
            counter = 0;
            pthread_cond_broadcast(&cond_var);
        }
        else
        {
            while (pthread_cond_wait(&cond_var, &mutex) != 0)
            {
            }
        }
        pthread_mutex_unlock(&mutex);
    }
    
    return NULL;
}

int main(int argC, char *argV[])
{
    int i, j;

    if (argC != 2)
        printf("Usage : %s <file name containing system configuration data>", argV[0]);
    else
    {
        initiateSystem(argV[1]);
        initOutputFile();

        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&cond_var, NULL);

        clock_t start, end;
        double execution_time;
        start = clock();
        pthread_t* tread_handles = malloc(NUM_THREADS * sizeof(pthread_t));

        printf("Body   :     x              y           vx              vy   ");
        for (long thread = 0; thread < NUM_THREADS; thread++)
        {
            pthread_create(&tread_handles[thread], NULL, simulate, (void*)thread);
        }

        for (long thread = 0; thread < NUM_THREADS; thread++)
        {
            pthread_join(tread_handles[thread], NULL);
        }
        free(tread_handles);
        end = clock();
        execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
        printf("%lf", execution_time);
    }
    return 0;
}
