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
#define MAX_ITERATIONS 500000                 // Maksymalna liczba iteracji
#define MIGRATION_CHANCE 0.002                // Prawdopodobieństwo migracji
#define MUTATIONS_IN_SOLUTION 2               // Liczba mutacji w rozwiązaniu
#define POPULATION_SIZE 100                   // Rozmiar populacji
#define POPULATION_TO_DIE 0.15                // Odsetek populacji, który zginie w iteracji
#define RANDOM_SOLUTIONS 0.5                  // Odsetek losowych rozwiązań w pierwotnej populacji
#define SOLUTION_CROSSOVER_CHANCE 0.85        // Prawdopodobieństwo, że rozwiązanie się rozmnoży
#define SOLUTION_MUTATION_CHANCE 0.05         // Prawdopodobieństwo, że w rozwiązaniu zajdzie mutacja

#define MAX_TASKS 1000
#define MAX_PROCESSORS 50
#define NUM_THREADS 1024

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
__device__ __constant__ int processorCount;       // Liczba procesorów
__device__ __constant__ int taskCount;            // Liczba zadań
__device__ __constant__ int executionTimes[2048]; // Tablica czasów wykonania zadań



__global__ void genetic(const long long tresholdCmax, unsigned int randSeed, CudaOutput* cudaOut);
__device__ void doGeneticIteration(RandSeed seed, Solution* population, const Solution* bestSolution);
__device__ void generateInitialSolutions(RandSeed seed, Solution* population);
__device__ void buildSolutionRandom(RandSeed seed, Solution* solution);
__device__ void buildSolutionGreedy(RandSeed seed, Solution* solution);
__device__ void performCrossOvers(RandSeed seed, Solution* population);
__device__ void crossOver(RandSeed seed, Solution* parent1, Solution* parent2, Solution* child);
__device__ void performMutations(RandSeed seed, Solution* population);
__device__ void mutate(RandSeed seed, Solution* solution);
__device__ void greedyMutate(Solution* solution);
__device__ void updateBestSolution(Solution* population, Solution* bestSolution);
__device__ void performMigration(RandSeed seed, Solution* population, const Solution* bestSolution);
__device__ void measureSolutionCmax(Solution* solution);
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

    __syncthreads();

    int iterations = 0;

    // Allocate the population
    ProcId taskMem[MAX_TASKS * POPULATION_SIZE];
    Solution population[POPULATION_SIZE];
    for(int i = 0; i < POPULATION_SIZE; ++i){
        population[i].tasks = &taskMem[i * taskCount];
    }

    // Prepare the population
    generateInitialSolutions(&randSeed, population);

    // Save the best solution (and allocate the memory for it)
    ProcId bestSolutionTasks[MAX_TASKS];
    Solution bestSolution;
    bestSolution.tasks = bestSolutionTasks;
    bestSolution.cmax = LLONG_MAX;
    updateBestSolution(population, &bestSolution);

    while(iterations < MAX_ITERATIONS) {
        doGeneticIteration(&randSeed, population, &bestSolution);
        updateBestSolution(population, &bestSolution);

        ++iterations;

        // If any thread finds a solution, stop the entire block
        if(bestSolution.cmax <= tresholdCmax){
            earlyStop = true;
        }
        if (earlyStop) break;
    }

    cudaOut_local[tId].iterations = iterations;
    cudaOut_local[tId].bestCmax = bestSolution.cmax;

    __threadfence();
    __syncthreads();

    // Save the best solution
    if(tId == 0){
        cudaOut->bestCmax = LLONG_MAX;
        for(int i = 0; i < blockDim.x; ++i){
            if(cudaOut_local[i].bestCmax < cudaOut->bestCmax){
                cudaOut->bestCmax = cudaOut_local[i].bestCmax;
                cudaOut->iterations = cudaOut_local[i].iterations;
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
__device__ void doGeneticIteration(RandSeed seed, Solution* population, const Solution* bestSolution){
    performCrossOvers(seed, population);
    performMutations(seed, population);
    performMigration(seed, population, bestSolution);
}


/**
 * Generates an initial population of solutions
 * @param seed The seed for the random number generator
 * @param population The population to fill
 */
__device__ void generateInitialSolutions(RandSeed seed, Solution* population){
    const int randomCount = POPULATION_SIZE * RANDOM_SOLUTIONS;
    for(int i = 0; i < randomCount; ++i){
        buildSolutionRandom(seed, &population[i]);
    }
    for(int i = randomCount; i < POPULATION_SIZE; ++i){
        buildSolutionGreedy(seed, &population[i]);
    }
}

/**
 * Builds a random solution
 * @param seed The seed for the random number generator
 * @param solution Where to save the solution
 */
__device__ void buildSolutionRandom(RandSeed seed, Solution* solution){
    long long sum = 0;
    for(int i = 0; i < taskCount; ++i){
        solution->tasks[i] = randInt(seed) % processorCount;
        sum++;
    }
    measureSolutionCmax(solution);
}


/**
 * Builds a solution using the greedy algorithm
 * @param seed The seed for the random number generator
 * @param solution Where to save the solution
 */
__device__ void buildSolutionGreedy(RandSeed seed, Solution* solution){
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
    measureSolutionCmax(solution);
}


/**
 * Performs crossovers on the population.
 * Picks two parents and creates a children from them.
 * @param seed The seed for the random number generator
 * @param population The population to perform crossovers on
 */
__device__ void performCrossOvers(RandSeed seed, Solution* population){
    int crossOvers = POPULATION_SIZE * POPULATION_TO_DIE;

    while(crossOvers > 0){
        // TODO: Optimize
        if((randInt(seed) / (float)RAND_MAX) >= SOLUTION_CROSSOVER_CHANCE){
            continue;
        }

        int p1, p2, c; // Indices of: parent1, parent2, child
        p1 = randInt(seed) % POPULATION_SIZE;
        do{
            p2 = randInt(seed) % POPULATION_SIZE;
        } while(p1 == p2);
        do{
            c = randInt(seed) % POPULATION_SIZE;
        } while(p1 == c || p2 == c);

        crossOver(seed, &population[p1], &population[p2], &population[c]);
        --crossOvers;
    }
}


/**
 * Performs a crossover on two parents and creates a child
 * @param seed The seed to use for random numbers
 * @param parent1 The first parent
 * @param parent2 The second parent
 * @param child Where to save the child
 */
__device__ void crossOver(RandSeed seed, Solution* parent1, Solution* parent2, Solution* child){
    int start = randInt(seed) % taskCount;
    int end = randInt(seed) % (taskCount - start) + start;

    for(int i = 0; i < taskCount; ++i){
        if(i >= start && i <= end){
            child->tasks[i] = parent2->tasks[i];
        } else {
            child->tasks[i] = parent1->tasks[i];
        }
    }
    measureSolutionCmax(child);
}


/**
 * Picks solutions and mutates them randomly
 * @param seed The seed to use for random numbers
 * @param population The population to mutate
 */
__device__ void performMutations(RandSeed seed, Solution* population){
    for(int i = 0; i < POPULATION_SIZE; ++i){
        // TODO: Optimize
        if((randInt(seed) / (float)RAND_MAX) >= SOLUTION_MUTATION_CHANCE){
            continue;
        }
        if((randInt(seed) / (float)RAND_MAX) <= GREEDY_MUTATION_CHANCE){
            greedyMutate(&population[i]);
        } else {
            mutate(seed, &population[i]);
        }
    }
}


/**
 * Mutates a solution by swapping two tasks
 * @param seed The seed to use for random numbers
 * @param solution The solution to mutate
 */
__device__ void mutate(RandSeed seed, Solution* solution){
    for(int i = 0; i < MUTATIONS_IN_SOLUTION; ++i){
        int pos1 = randInt(seed) % taskCount;
        int pos2 = randInt(seed) % taskCount;
        ProcId temp = solution->tasks[pos1];
        solution->tasks[pos1] = solution->tasks[pos2];
        solution->tasks[pos2] = temp;
    }
    measureSolutionCmax(solution);
}


/**
 * Performs a greedy mutation on a solution
 * @param solution The solution to mutate
 */
__device__ void greedyMutate(Solution* solution){
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

    measureSolutionCmax(solution);
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
__device__ void performMigration(RandSeed seed, Solution* population, const Solution* bestSolution){
    if((randInt(seed) / (float)RAND_MAX) >= MIGRATION_CHANCE){
        return;
    }

    long long currentBestCmax = LLONG_MAX;
    for(int i = 0; i < POPULATION_SIZE; ++i){
        if(population[i].cmax < currentBestCmax){
            currentBestCmax = population[i].cmax;
        }
    }

    if(currentBestCmax > bestSolution->cmax){
        population[POPULATION_SIZE - 1].cmax = bestSolution->cmax;
        memcpy(population[POPULATION_SIZE - 1].tasks, bestSolution->tasks, sizeof(ProcId) * taskCount);
    } else {
        greedyMutate(&population[POPULATION_SIZE - 1]);
        buildSolutionGreedy(seed, &population[POPULATION_SIZE - 1]);
    }
}


/**
 * Measures the Cmax of a solution and saves it into the solution
 * @param solution The solution to measure
 */
__device__ void measureSolutionCmax(Solution* solution){
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
    int procCnt;
    int taskCnt;
    int* execTimes;
    loadData(&execTimes, &procCnt, &taskCnt);

    long long treshold = 0;
    if(argc >= 2) {
        treshold = atoll(argv[1]);
    }


    cudaMemcpyToSymbol(processorCount, &procCnt, sizeof(int));
    cudaMemcpyToSymbol(taskCount, &taskCnt, sizeof(int));
    cudaMemcpyToSymbol(executionTimes, execTimes, sizeof(int) * taskCnt);

    CudaOutput cudaOutput_host;
    CudaOutput* cudaOutput_dev;
    cudaMalloc(&cudaOutput_dev, sizeof(CudaOutput));

    fprintf(stderr, "Before kernel: %s\n", cudaGetErrorString(cudaGetLastError()));

    const clock_t startTime = clock();
    const unsigned int seed = (unsigned int)(time(NULL) & 0x7fffffff);
    genetic<<<1, NUM_THREADS>>>(treshold, seed, cudaOutput_dev);
    cudaDeviceSynchronize();

    fprintf(stderr, "After kernel: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(&cudaOutput_host, cudaOutput_dev, sizeof(CudaOutput), cudaMemcpyDeviceToHost);

    const clock_t workTime = clock() - startTime;
    //free(execTimes);

    fprintf(stderr, "Work time: %.1fs\n", (double)workTime / CLOCKS_PER_SEC);
    fprintf(stderr, "Best Cmax: %lld\n", cudaOutput_host.bestCmax);
    fprintf(stderr, "Iterations: %d\n", cudaOutput_host.iterations);

    cudaFree(cudaOutput_dev);
}
