// Konfiguracja algorytmu
#define GREEDY_MUTATION_CHANCE 0.3            // Prawdopodobieństwo, że mutacja będzie zachłanna
#define GREEDY_MUTATION_ITERATIONS 5          // Liczba iteracji, podczas zachłannej mutacji
#define MAX_DURATION 300                      // Maksymalny czas pracy w sekundach
#define MAX_ITERATIONS 50000000               // Maksymalna liczba iteracji
#define MIGRATION_CHANCE 0.002                // Prawdopodobieństwo migracji
#define MUTATIONS_IN_SOLUTION 2               // Liczba mutacji w rozwiązaniu
#define POPULATION_SIZE 100                   // Rozmiar populacji
#define POPULATION_TO_DIE 0.15                // Odsetek populacji, który zginie w iteracji
#define RANDOM_SOLUTIONS 0.5                  // Odsetek losowych rozwiązań w pierwotnej populacji
#define SOLUTION_MUTATION_CHANCE 0.05         // Prawdopodobieństwo, że w rozwiązaniu zajdzie mutacja

// Diagnostyka
#define ENABLE_EXPORT false                   // Czy eksportować wyniki na koniec
#define ENABLE_PRINT_STATS true               // Czy wyświetlać statystyki
#define PRINT_STATS_FREQ 50000                // Co ile iteracji wyświetlać status