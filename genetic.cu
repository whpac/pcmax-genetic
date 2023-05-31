// %%cu

#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

// Konfiguracja algorytmu
#define GREEDY_MUTATION_CHANCE 0.3            // Prawdopodobieństwo, że mutacja będzie zachłanna
#define GREEDY_MUTATION_ITERATIONS 5          // Liczba iteracji, podczas zachłannej mutacji
#define MAX_DURATION 300                      // Maksymalny czas pracy w sekundach
#define MAX_ITERATIONS 5000000                // Maksymalna liczba iteracji
#define MIGRATION_CHANCE 0.002                // Prawdopodobieństwo migracji
#define MUTATIONS_IN_SOLUTION 2               // Liczba mutacji w rozwiązaniu
#define POPULATION_SIZE 100                   // Rozmiar populacji
#define POPULATION_TO_DIE 0.15                // Odsetek populacji, który zginie w iteracji
#define RANDOM_SOLUTIONS 0.5                  // Odsetek losowych rozwiązań w pierwotnej populacji
#define SOLUTION_CROSSOVER_CHANCE 0.85        // Prawdopodobieństwo, że rozwiązanie się rozmnoży
#define SOLUTION_MUTATION_CHANCE 0.05         // Prawdopodobieństwo, że w rozwiązaniu zajdzie mutacja

#define MAX_TASKS 250
#define MAX_PROCESSORS 50
#define NUM_THREADS 32
#define NUM_BLOCKS 16

// Diagnostyka
#define ENABLE_EXPORT false                   // Czy eksportować wyniki na koniec
#define ENABLE_PRINT_STATS true               // Czy wyświetlać statystyki
#define PRINT_STATS_FREQ 5000                 // Co ile iteracji wyświetlać status

typedef unsigned char ProcId;
typedef unsigned int* RandSeed;

typedef struct {
    long long cmax;
    ProcId* tasks;
} Solution;

typedef struct {
    long long bestCmax;
    int iterations;
} CudaOutput;

// Dane problemu
__device__ __constant__ int processorCount;            // Liczba procesorów
__device__ __constant__ int taskCount;                 // Liczba zadań
__device__ __constant__ int executionTimes[MAX_TASKS]; // Tablica czasów wykonania zadań
__device__ bool isFinished;                            // Czy algorytm zakończył działanie



__global__ void genetic(const long long tresholdCmax, unsigned int randSeed, CudaOutput* cudaOut);
__device__ void doGeneticIteration(RandSeed seed, Solution* population, const Solution* bestSolution, int* executionTimes);
__device__ void generateInitialSolutions(RandSeed seed, Solution* population, int* executionTimes);
__device__ void buildSolutionRandom(RandSeed seed, Solution* solution, int* executionTimes);
__device__ void buildSolutionGreedy(RandSeed seed, Solution* solution, int* executionTimes);
__device__ void performCrossOvers(RandSeed seed, Solution* population, int* executionTimes);
__device__ void crossOver(RandSeed seed, Solution* parent1, Solution* parent2, Solution* child, int* executionTimes);
__device__ void performMutations(RandSeed seed, Solution* population, int* executionTimes);
__device__ void mutate(RandSeed seed, Solution* solution, int* executionTimes);
__device__ void greedyMutate(Solution* solution, int* executionTimes);
__device__ void updateBestSolution(Solution* population, Solution* bestSolution);
__device__ void performMigration(RandSeed seed, Solution* population, const Solution* bestSolution, int* executionTimes);
__device__ void measureSolutionCmax(Solution* solution, int* executionTimes);
void loadData(int** executionTimes, int* processorCount, int* taskCount);


#undef RAND_MAX
#define RAND_MAX 0xffffffff

// https://en.wikipedia.org/wiki/Linear-feedback_shift_register
__device__ unsigned int randInt(RandSeed seed) {
    unsigned int b = 1 & (*seed ^ (*seed >> 1) ^ (*seed >> 21) ^ (*seed >> 31));
    *seed = (*seed << 1) | b;
    return *seed;
}


/**
 * Entry point of the algorithm
 * @param startTime The time when the algorithm started
 * @param tresholdCmax When to stop the algorithm
 */
__global__ void genetic(const long long tresholdCmax, unsigned int randSeed, CudaOutput* cudaOut){
    int tId = threadIdx.x;
    randSeed ^= (tId & 0xfff) << 19;

    __shared__ volatile bool earlyStop;
    if(tId == 0) earlyStop = false;

    __shared__ CudaOutput cudaOut_local[NUM_THREADS];

    int iterations = 0;

    __shared__ int executionTimes_sh[MAX_TASKS];
    for(int i = tId; i < taskCount; i+= NUM_THREADS){
        executionTimes_sh[i] = executionTimes[i];
    }

    // Allocate the population
    __shared__ ProcId taskMem[MAX_TASKS * POPULATION_SIZE];
    __shared__ Solution population[POPULATION_SIZE];
    for(int i = tId; i < POPULATION_SIZE; i += NUM_THREADS){
        population[i].tasks = &taskMem[i * taskCount];
    }

    __syncthreads();

    // Prepare the population
    generateInitialSolutions(&randSeed, population, executionTimes_sh);

    // Save the best solution (and allocate the memory for it)
    __shared__ ProcId bestSolutionTasks[MAX_TASKS];
    __shared__ Solution bestSolution;
    bestSolution.tasks = bestSolutionTasks;
    bestSolution.cmax = LLONG_MAX;
    if(tId == 0) updateBestSolution(population, &bestSolution);

    while(iterations < MAX_ITERATIONS) {
        doGeneticIteration(&randSeed, population, &bestSolution, executionTimes_sh);
        if(tId == 0) updateBestSolution(population, &bestSolution);

        ++iterations;

        // If any thread finds a solution, stop the entire block
        if(bestSolution.cmax <= tresholdCmax){
            earlyStop = true;
            isFinished = true;
            __threadfence();
        }
        if (earlyStop) break;

        if(tId == 0 && (iterations & 0x3ff) == 0) {
            earlyStop = isFinished;
        }
    }

    cudaOut_local[tId].iterations = iterations;
    cudaOut_local[tId].bestCmax = bestSolution.cmax;
    // for(int i = 0; i < taskCount; i++) {
    //     cudaOut_local[tId].tasks[i] = bestSolution.tasks[i];
    // }

    __threadfence();
    __syncthreads();

    // Save the best solution
    if(tId == 0){
        cudaOut->bestCmax = LLONG_MAX;
        for(int i = 0; i < blockDim.x; ++i){
            if(cudaOut_local[i].bestCmax < cudaOut->bestCmax){
                cudaOut->bestCmax = cudaOut_local[i].bestCmax;
                cudaOut->iterations = cudaOut_local[i].iterations;
                // for(int j = 0; j < taskCount; j++) {
                //     cudaOut->tasks[j] = cudaOut_local[i].tasks[j];
                // }
            }
        }
    }
}


/**
 * Performs one iteration of genetic algorithm
 * @param seed The seed for the random number generator
 * @param population The current population
 * @param bestSolution The best solution in the history
 */
__device__ void doGeneticIteration(RandSeed seed, Solution* population, const Solution* bestSolution, int* executionTimes){
    performCrossOvers(seed, population, executionTimes);
    performMutations(seed, population, executionTimes);
    performMigration(seed, population, bestSolution, executionTimes);
}


/**
 * Generates an initial population of solutions
 * @param seed The seed for the random number generator
 * @param population The population to fill
 */
__device__ void generateInitialSolutions(RandSeed seed, Solution* population, int* executionTimes){
    const int randomCount = POPULATION_SIZE * RANDOM_SOLUTIONS;
    int tId = threadIdx.x;
    int i = tId;
    for(; i < randomCount; i+= NUM_THREADS){
        buildSolutionRandom(seed, &population[i], executionTimes);
    }
    for(; i < POPULATION_SIZE; i += NUM_THREADS){
        buildSolutionGreedy(seed, &population[i], executionTimes);
    }
}

/**
 * Builds a random solution
 * @param seed The seed for the random number generator
 * @param solution Where to save the solution
 */
__device__ void buildSolutionRandom(RandSeed seed, Solution* solution, int* executionTimes){
    long long sum = 0;
    for(int i = 0; i < taskCount; ++i){
        solution->tasks[i] = randInt(seed) % processorCount;
        sum++;
    }
    measureSolutionCmax(solution, executionTimes);
}


/**
 * Builds a solution using the greedy algorithm
 * @param seed The seed for the random number generator
 * @param solution Where to save the solution
 */
__device__ void buildSolutionGreedy(RandSeed seed, Solution* solution, int* executionTimes){
    long long processorUsage[MAX_PROCESSORS];
    int taskOrder[MAX_TASKS];

    for(int i = 0; i < processorCount; ++i){
        processorUsage[i] = 0;
    }

    // Randomize the order of processing tasks
    for(int i = 0; i < taskCount; ++i){
        taskOrder[i] = i;
    }
    for(int i = taskCount - 1; i > 0; --i){
        int randomIndex = randInt(seed) % i;
        int temp = taskOrder[i];
        taskOrder[i] = taskOrder[randomIndex];
        taskOrder[randomIndex] = temp;
    }

    // Assign tasks to the least used processor
    for(int i = 0; i < taskCount; ++i){
        int t = taskOrder[i];
        ProcId minIndex = 0;
        for(ProcId j = 1; j < processorCount; ++j){
            if(processorUsage[j] < processorUsage[minIndex]){
                minIndex = j;
            }
        }
        solution->tasks[t] = minIndex;
        processorUsage[minIndex] += executionTimes[t];
    }
    measureSolutionCmax(solution, executionTimes);
}


/**
 * Performs crossovers on the population.
 * Picks two parents and creates a children from them.
 * @param seed The seed for the random number generator
 * @param population The population to perform crossovers on
 */
__device__ void performCrossOvers(RandSeed seed, Solution* population, int* executionTimes){
    int crossOvers = POPULATION_SIZE * POPULATION_TO_DIE;
    int tId = threadIdx.x;

    for(int i = tId; i < crossOvers; i += NUM_THREADS){
        int p1, p2, c; // Indices of: parent1, parent2, child
        p1 = randInt(seed) % POPULATION_SIZE;
        do{
            p2 = randInt(seed) % POPULATION_SIZE;
        } while(p1 == p2);
        do{
            c = randInt(seed) % POPULATION_SIZE;
        } while(p1 == c || p2 == c);

        crossOver(seed, &population[p1], &population[p2], &population[c], executionTimes);
    }
}


/**
 * Performs a crossover on two parents and creates a child
 * @param seed The seed to use for random numbers
 * @param parent1 The first parent
 * @param parent2 The second parent
 * @param child Where to save the child
 */
__device__ void crossOver(RandSeed seed, Solution* parent1, Solution* parent2, Solution* child, int* executionTimes){
    int start = randInt(seed) % taskCount;
    int end = randInt(seed) % (taskCount - start) + start;

    for(int i = 0; i < taskCount; ++i){
        if(i >= start && i <= end){
            child->tasks[i] = parent2->tasks[i];
        } else {
            child->tasks[i] = parent1->tasks[i];
        }
    }
    measureSolutionCmax(child, executionTimes);
}


/**
 * Picks solutions and mutates them randomly
 * @param seed The seed to use for random numbers
 * @param population The population to mutate
 */
__device__ void performMutations(RandSeed seed, Solution* population, int* executionTimes){
    __shared__ int mutatingIdx[(int)(POPULATION_SIZE * SOLUTION_MUTATION_CHANCE)];
    int mutationCount = POPULATION_SIZE * SOLUTION_MUTATION_CHANCE;
    int tId = threadIdx.x;

    // Pick solutions to mutate
    for(int i = tId; i < mutationCount; i += NUM_THREADS){
        mutatingIdx[i] = randInt(seed) % POPULATION_SIZE;
    }

    // Mutate the solutions
    for(int i = tId; i < mutationCount; i += NUM_THREADS){
        int idx = mutatingIdx[i];
        if((randInt(seed) / (float)RAND_MAX) <= GREEDY_MUTATION_CHANCE){
            greedyMutate(&population[idx], executionTimes);
        } else {
            mutate(seed, &population[idx], executionTimes);
        }
    }
}


/**
 * Mutates a solution by swapping two tasks
 * @param seed The seed to use for random numbers
 * @param solution The solution to mutate
 */
__device__ void mutate(RandSeed seed, Solution* solution, int* executionTimes){
    for(int i = 0; i < MUTATIONS_IN_SOLUTION; ++i){
        int pos1 = randInt(seed) % taskCount;
        int pos2 = randInt(seed) % taskCount;
        ProcId temp = solution->tasks[pos1];
        solution->tasks[pos1] = solution->tasks[pos2];
        solution->tasks[pos2] = temp;
    }
    measureSolutionCmax(solution, executionTimes);
}


/**
 * Performs a greedy mutation on a solution
 * @param solution The solution to mutate
 */
__device__ void greedyMutate(Solution* solution, int* executionTimes){
    long long processorUsage[MAX_PROCESSORS];

    for(int i = 0; i < processorCount; ++i){
        processorUsage[i] = 0;
    }

    for(int i = 0; i < taskCount; ++i){
        processorUsage[solution->tasks[i]] += executionTimes[i];
    }

    for(int i = 0; i < GREEDY_MUTATION_ITERATIONS; ++i){
        ProcId procLeastInd = 0, procMostInd = 0;
        long long procLeastUsage = processorUsage[0];
        long long procMostUsage = processorUsage[0];

        for(ProcId j = 1; j < processorCount; ++j){
            if(processorUsage[j] < procLeastUsage){
                procLeastUsage = processorUsage[j];
                procLeastInd = j;
            }
            if(processorUsage[j] > procMostUsage){
                procMostUsage = processorUsage[j];
                procMostInd = j;
            }
        }

        int shortestTaskInd = 0, shortestTaskTime = INT_MAX;
        for(int j = 0; j < taskCount; ++j){
            if(solution->tasks[j] == procMostInd && executionTimes[j] < shortestTaskTime){
                shortestTaskTime = executionTimes[j];
                shortestTaskInd = j;
            }
        }

        solution->tasks[shortestTaskInd] = procLeastInd;
        processorUsage[procLeastInd] += shortestTaskTime;
        processorUsage[procMostInd] -= shortestTaskTime;
    }

    measureSolutionCmax(solution, executionTimes);
}


/**
 * Updates the best solution in the population
 * @param population The population to search
 * @param bestSolution Where to save the best solution
 */
__device__ void updateBestSolution(Solution* population, Solution* bestSolution){
    for(int i = 0; i < POPULATION_SIZE; ++i){
        if(population[i].cmax < bestSolution->cmax){
            bestSolution->cmax = population[i].cmax;
            for(int j = 0; j < taskCount; ++j){
                bestSolution->tasks[j] = population[i].tasks[j];
            }
        }
    }
}


/**
 * Performs migration of the best solution to the population
 * @param seed The seed to use for random numbers
 * @param population The population to migrate to
 * @param bestSolution The best solution to migrate
 */
__device__ void performMigration(RandSeed seed, Solution* population, const Solution* bestSolution, int* executionTimes){
    int tId = threadIdx.x;
    __shared__ bool doMigration;

    if(tId == 0){
        doMigration = (randInt(seed) / (float)RAND_MAX) < MIGRATION_CHANCE;
    }

    __syncthreads();
    if(!doMigration) return;


    __shared__ long long currentBestCmax;
    currentBestCmax = LLONG_MAX;

    __syncthreads();

    for(int i = tId; i < POPULATION_SIZE; i += NUM_THREADS){
        atomicMin(&currentBestCmax, population[i].cmax);
    }

    __syncthreads();

    if(currentBestCmax > bestSolution->cmax){
        population[POPULATION_SIZE - 1].cmax = bestSolution->cmax;

        for(int i = tId; i < taskCount; i += NUM_THREADS){
            population[POPULATION_SIZE - 1].tasks[i] = bestSolution->tasks[i];
        }
    } else {
        if(tId == 0) buildSolutionGreedy(seed, &population[POPULATION_SIZE - 1], executionTimes);
    }
}


/**
 * Measures the Cmax of a solution and saves it into the solution
 * @param solution The solution to measure
 */
__device__ void measureSolutionCmax(Solution* solution, int* executionTimes){
    long long processorUsage[MAX_PROCESSORS];

    for(int i = 0; i < processorCount; ++i){
        processorUsage[i] = 0;
    }

    for(int i = 0; i < taskCount; ++i){
        processorUsage[solution->tasks[i]] += executionTimes[i];
    }

    long long maxUsage = 0;
    for(int i = 0; i < processorCount; ++i){
        if(processorUsage[i] > maxUsage){
            maxUsage = processorUsage[i];
        }
    }
    solution->cmax = maxUsage;
}

/**
 * Loads data from stdin and saves it into global variables
 */
void loadData(int** executionTimes, int* processorCount, int* taskCount){
    if(!scanf("%d", processorCount)) exit(1);
    if(!scanf("%d", taskCount)) exit(1);

    *executionTimes = (int*)malloc(sizeof(int) * *taskCount);
    for(int i = 0; i < *taskCount; ++i){
        if(!scanf("%d", &(*executionTimes)[i])) exit(1);
    }
    fprintf(stderr, "Loaded data (%d proc, %d tasks)\n", *processorCount, *taskCount);
}

int main(int argc, char** argv){
    int procCnt = 10;
    int taskCnt = 200;
    int* execTimes;
    loadData(&execTimes, &procCnt, &taskCnt);

    long long treshold = 0;
    if(argc >= 2) {
        treshold = atoll(argv[1]);
    }

    bool FALSE = false;

    cudaMemcpyToSymbol(processorCount, &procCnt, sizeof(int));
    cudaMemcpyToSymbol(taskCount, &taskCnt, sizeof(int));
    cudaMemcpyToSymbol(executionTimes, execTimes, sizeof(int) * taskCnt);
    cudaMemcpyToSymbol(isFinished, &FALSE, sizeof(bool));

    CudaOutput cudaOutput_host;
    CudaOutput* cudaOutput_dev;
    cudaMalloc(&cudaOutput_dev, sizeof(CudaOutput));

    fprintf(stderr, "Before kernel: %s\n", cudaGetErrorString(cudaGetLastError()));

    const clock_t startTime = clock();
    const unsigned int seed = (unsigned int)(time(NULL) & 0x7fffffff);
    genetic<<<NUM_BLOCKS, NUM_THREADS>>>(treshold, seed, cudaOutput_dev);
    cudaDeviceSynchronize();

    fprintf(stderr, "After kernel: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(&cudaOutput_host, cudaOutput_dev, sizeof(CudaOutput), cudaMemcpyDeviceToHost);

    const clock_t workTime = clock() - startTime;
    //free(execTimes);

    fprintf(stderr, "Work time: %.1fs\n", (double)workTime / CLOCKS_PER_SEC);
    fprintf(stderr, "Best Cmax: %lld\n", cudaOutput_host.bestCmax);
    fprintf(stderr, "Iterations: %d\n", cudaOutput_host.iterations);
    /*fprintf(stderr, "Best solution:\n");

    for(int i = 0; i < taskCnt; i += 20){
        for(int j = 0; j < 20 && i + j < taskCnt; ++j){
            fprintf(stderr, "%3d ", cudaOutput_host.tasks[i + j]);
        }
        fprintf(stderr, "\n");
    }*/

    cudaFree(cudaOutput_dev);
}
