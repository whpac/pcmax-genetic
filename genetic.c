#include <stdbool.h>
#include <limits.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "config.h"

typedef struct {
    long long cmax;
    unsigned int* tasks;
} Solution;
typedef unsigned int* RandSeed;

// Dane problemu
int processorCount;     // Liczba procesorów
int taskCount;          // Liczba zadań
int* executionTimes;    // Tablica czasów wykonania zadań



void genetic(const double startTime, Solution* bestSolution, const long long tresholdCmax, volatile bool* earlyStop);
void doGeneticIteration(RandSeed seed, Solution* population, const Solution* bestSolution);
void generateInitialSolutions(RandSeed seed, Solution* population);
void buildSolutionRandom(RandSeed seed, Solution* solution);
void buildSolutionGreedy(RandSeed seed, Solution* solution);
void performCrossOvers(RandSeed seed, Solution* population);
void crossOver(RandSeed seed, Solution* parent1, Solution* parent2, Solution* child);
void performMutations(RandSeed seed, Solution* population);
void mutate(RandSeed seed, Solution* solution);
void greedyMutate(Solution* solution);
unsigned long long getCurrentBestCmax(Solution* population);
void updateBestSolution(Solution* population, Solution* bestSolution);
void performMigration(RandSeed seed, Solution* population, const Solution* bestSolution);
void measureSolutionCmax(Solution* solution);
void printStats(const double startTime, const int iterations, const long long currentBestCmax, const long long allTimeBestCmax);
void printFinalStats(const double startTime, const int iterations, const long long firstCmax, const long long allTimeBestCmax);
void loadData(FILE* inFile);
void export(const Solution* bestSolution);

#undef RAND_MAX
#define RAND_MAX 0xffffffff

// https://en.wikipedia.org/wiki/Linear-feedback_shift_register
unsigned int randInt(RandSeed seed) {
    unsigned int b = 1 & (*seed ^ (*seed >> 1) ^ (*seed >> 21) ^ (*seed >> 31));
    *seed = (*seed << 1) | b;
    return *seed;
}


/**
 * Entry point of the algorithm
 * @param startTime The time when the algorithm started
 * @param bestSolution The best solution in the history (will allocate the memory for it,
 *      but needs freeing bestSolution->tasks)
 * @param tresholdCmax When to stop the algorithm
 * @param earlyStop Whether to stop the algorithm early
 */
void genetic(const double startTime, Solution* bestSolution, const long long tresholdCmax, volatile bool* earlyStop){
    unsigned int randSeed = time(NULL);
    unsigned int tId = omp_get_thread_num();
    randSeed ^= (tId & 0xf) << 27;

    const double endTime = startTime + MAX_DURATION;
    int iterations = 0;

    // Allocate the population
    int* taskMem = malloc(sizeof(int) * taskCount * POPULATION_SIZE);
    Solution population[POPULATION_SIZE];
    for(int i = 0; i < POPULATION_SIZE; ++i){
        population[i].tasks = &taskMem[i * taskCount];
    }

    // Prepare the population
    generateInitialSolutions(&randSeed, population);
    fprintf(stderr, "Initial population generated\n");

    // Save the best solution (and allocate the memory for it)
    bestSolution->tasks = malloc(sizeof(int) * taskCount);
    bestSolution->cmax = LLONG_MAX;
    updateBestSolution(population, bestSolution);
    const long long firstCmax = bestSolution->cmax;

    fprintf(stderr, "Starting genetic algorithm loop\n");
    fprintf(stderr, "Initial Cmax: %lld\n", firstCmax);
    while((iterations < MAX_ITERATIONS) && (omp_get_wtime() <= endTime) && (bestSolution->cmax > tresholdCmax)) {
        doGeneticIteration(&randSeed, population, bestSolution);
        updateBestSolution(population, bestSolution);

        ++iterations;
        printStats(startTime, iterations, getCurrentBestCmax(population), bestSolution->cmax);

        // Check the early stop once in a while not to cause simultaneous access to the variable
        // Exactly once every 1024 iterations
        if((iterations & 0x3ff) == 0) {
            if (*earlyStop) break;
        }
    }

    // Free the population task assignment
    free(taskMem);

    #pragma omp critical
    printFinalStats(startTime, iterations, firstCmax, bestSolution->cmax);
}


/**
 * Performs one iteration of genetic algorithm
 * @param population The current population
 * @param bestSolution The best solution in the history
 */
void doGeneticIteration(RandSeed seed, Solution* population, const Solution* bestSolution){
    performCrossOvers(seed, population);
    performMutations(seed, population);
    performMigration(seed, population, bestSolution);
}


/**
 * Generates an initial population of solutions
 * @param population The population to fill
 */
void generateInitialSolutions(RandSeed seed, Solution* population){
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
 * @param solution Where to save the solution
 */
void buildSolutionRandom(RandSeed seed, Solution* solution){
    for(int i = 0; i < taskCount; ++i){
        solution->tasks[i] = randInt(seed) % processorCount;
    }
    measureSolutionCmax(solution);
}


/**
 * Builds a solution using the greedy algorithm
 * @param solution Where to save the solution
 */
void buildSolutionGreedy(RandSeed seed, Solution* solution){
    long long processorUsage[processorCount];
    int taskOrder[taskCount];

    for(int i = 0; i < processorCount; ++i){
        processorUsage[i] = 0;
    }

    for(int i = 0; i < taskCount; ++i){
        taskOrder[i] = i;
    }
    for(int i = taskCount - 1; i > 0; --i){
        int randomIndex = randInt(seed) % i;
        int temp = taskOrder[i];
        taskOrder[i] = taskOrder[randomIndex];
        taskOrder[randomIndex] = temp;
    }

    for(int i = 0; i < taskCount; ++i){
        int t = taskOrder[i];
        int minIndex = 0;
        for(int j = 1; j < processorCount; ++j){
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
 * @param population The population to perform crossovers on
 */
void performCrossOvers(RandSeed seed, Solution* population){
    int crossOvers = POPULATION_SIZE * POPULATION_TO_DIE;

    while(crossOvers > 0){
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
 * @param parent1 The first parent
 * @param parent2 The second parent
 * @param child Where to save the child
 */
void crossOver(RandSeed seed, Solution* parent1, Solution* parent2, Solution* child){
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
 * @param population The population to mutate
 */
void performMutations(RandSeed seed, Solution* population){
    for(int i = 0; i < POPULATION_SIZE; ++i){
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
 * @param solution The solution to mutate
 */
void mutate(RandSeed seed, Solution* solution){
    for(int i = 0; i < MUTATIONS_IN_SOLUTION; ++i){
        int pos1 = randInt(seed) % taskCount;
        int pos2 = randInt(seed) % taskCount;
        int temp = solution->tasks[pos1];
        solution->tasks[pos1] = solution->tasks[pos2];
        solution->tasks[pos2] = temp;
    }
    measureSolutionCmax(solution);
}


/**
 * Performs a greedy mutation on a solution
 * @param solution The solution to mutate
 */
void greedyMutate(Solution* solution){
    long long processorUsage[processorCount];

    for(int i = 0; i < processorCount; ++i){
        processorUsage[i] = 0;
    }

    for(int i = 0; i < taskCount; ++i){
        processorUsage[solution->tasks[i]] += executionTimes[i];
    }

    for(int i = 0; i < GREEDY_MUTATION_ITERATIONS; ++i){
        int procLeastInd = 0, procMostInd = 0;
        long long procLeastUsage = processorUsage[0];
        long long procMostUsage = processorUsage[0];

        for(int j = 1; j < processorCount; ++j){
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
 * Returns the Cmax of the best solution in the current population
 * @param population The population to scan through
 */
unsigned long long getCurrentBestCmax(Solution* population) {
    unsigned long long best = LLONG_MAX;
    for(int i = 0; i < POPULATION_SIZE; ++i){
        if(population[i].cmax < best){
            best = population[i].cmax;
        }
    }

    return best;
}


/**
 * Updates the best solution in the population
 * @param population The population to search
 * @param bestSolution Where to save the best solution
 */
void updateBestSolution(Solution* population, Solution* bestSolution){
    for(int i = 0; i < POPULATION_SIZE; ++i){
        if(population[i].cmax < bestSolution->cmax){
            bestSolution->cmax = population[i].cmax;
            memcpy(bestSolution->tasks, population[i].tasks, sizeof(int) * taskCount);
        }
    }
}


/**
 * Performs migration of the best solution to the population
 * @param population The population to migrate to
 * @param bestSolution The best solution to migrate
 */
void performMigration(RandSeed seed, Solution* population, const Solution* bestSolution){
    if((randInt(seed) / (float)RAND_MAX) >= MIGRATION_CHANCE){
        return;
    }

    if(getCurrentBestCmax(population) > bestSolution->cmax){
        population[POPULATION_SIZE - 1].cmax = bestSolution->cmax;
        memcpy(population[POPULATION_SIZE - 1].tasks, bestSolution->tasks, sizeof(int) * taskCount);
    } else {
        buildSolutionGreedy(seed, &population[POPULATION_SIZE - 1]);
    }
}


/**
 * Measures the Cmax of a solution and saves it into the solution
 * @param solution The solution to measure
 */
void measureSolutionCmax(Solution* solution){
    long long processorUsage[processorCount];

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
 * Prints statistics every PRINT_STATS_FREQ iterations
 * @param startTime The time when the algorithm started
 * @param iterations The number of iterations performed
 * @param currentBestCmax The current best Cmax
 * @param allTimeBestCmax The all time best Cmax
 */
void printStats(const double startTime, const int iterations, const long long currentBestCmax, const long long allTimeBestCmax){
#if ENABLE_PRINT_STATS
    if(iterations % PRINT_STATS_FREQ != 0){
        return;
    }

    double duration = omp_get_wtime() - startTime;
    int tId = omp_get_thread_num();
    fprintf(stderr, "[%2d |%9d]: %6.1fs elapsed, Cmax: %lld, ATB: %lld\n", tId, iterations, duration, currentBestCmax, allTimeBestCmax);
#endif
}


/**
 * Prints final statistics
 * @param startTime The time when the algorithm started
 * @param iterations The number of iterations performed
 * @param firstCmax The Cmax of the first solution
 */
void printFinalStats(const double startTime, const int iterations, const long long firstCmax, const long long allTimeBestCmax){
#if ENABLE_PRINT_STATS
    long long totalExecTime = 0;
    for(int i = 0; i < taskCount; ++i){
        totalExecTime += executionTimes[i];
    }
    double optimum = (double)totalExecTime / processorCount;

    double duration = omp_get_wtime() - startTime;
    fprintf(stderr, "\nFinished job [%d]\n", omp_get_thread_num());
    fprintf(stderr, "    %d iterations performed\n", iterations);
    fprintf(stderr, "    %.1f seconds\n", duration);
    fprintf(stderr, "    \033[1;32m%lld = Cmax\033[0m\n", allTimeBestCmax);
    fprintf(stderr, "    %.1f = Cmax* <divisible tasks>\n", optimum);
    fprintf(stderr, "    %lld -> %lld genetic progress\n", firstCmax, allTimeBestCmax);
#endif
}

/**
 * Loads data from stdin and saves it into global variables
 */
void loadData(FILE* inFile){
    if(!fscanf(inFile, "%d", &processorCount)) exit(1);
    if(!fscanf(inFile, "%d", &taskCount)) exit(1);

    executionTimes = malloc(sizeof(int) * taskCount);
    for(int i = 0; i < taskCount; ++i){
        if(!fscanf(inFile, "%d", &executionTimes[i])) exit(1);
    }
    fprintf(stderr, "Loaded data (%d proc, %d tasks)\n", processorCount, taskCount);
}

/**
 * Exports the processor usage to the stdout
 * @param bestSolution The best solution to export
 */
void export(const Solution* bestSolution){
#if ENABLE_EXPORT
    long long processorUsage[processorCount];
    for(int i = 0; i < processorCount; ++i){
        processorUsage[i] = 0;
    }

    for(int i = 0; i < taskCount; ++i){
        processorUsage[bestSolution->tasks[i]] += executionTimes[i];
    }

    for(int i = 0; i < processorCount; ++i){
        printf("%lld\n", processorUsage[i]);
    }
#endif
}


int main(int argc, char** argv){
    long long treshold = 0;
    if(argc >= 2) {
        treshold = atoll(argv[1]);
    }

    FILE* file = stdin;
    if(argc >= 3) {
        char* fileName = argv[2];
        file = fopen(fileName, "r");
    }
    loadData(file);

    if(file != stdin){
        fclose(file);
    }

    omp_set_num_threads(4);

    volatile bool solutionFound = false;

    const double globalStartTime = omp_get_wtime();
    #pragma omp parallel firstprivate(taskCount, processorCount, executionTimes, treshold) shared(solutionFound)
    {
        Solution bestSolution;

        const double startTime = omp_get_wtime();
        genetic(startTime, &bestSolution, treshold, &solutionFound);
        solutionFound = true;

        export(&bestSolution);

        free(bestSolution.tasks);
    }
    const double globalTime = omp_get_wtime() - globalStartTime;
    free(executionTimes);

    fprintf(stderr, "Global time: %.1fs\n", globalTime);
}
