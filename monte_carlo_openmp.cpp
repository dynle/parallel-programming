#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>

int main() {
    long long num_samples = 100000000;
    long long circle_count = 0;

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        // Seeding with time + thread ID makes each run and each thread unique.
        // Avoid regularities in the random numbers by using a different seed for each thread.
        unsigned int seed = time(NULL) + omp_get_thread_num();

        #pragma omp for reduction(+:circle_count)
        for (long long i = 0; i < num_samples; ++i) {
            double x, y;

            x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;

            if (x * x + y * y <= 1.0) {
                circle_count++;
            }
        }
    }

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    double pi_estimate = 4.0 * circle_count / num_samples;

    std::cout << "Total Samples: " << num_samples << std::endl;
    std::cout << "Points in Circle: " << circle_count << std::endl;
    std::cout << "Estimated Pi: " << pi_estimate << std::endl;
    std::cout << "Execution Time: " << execution_time << " seconds" << std::endl;
    std::cout << "Accuracy: " << (100 - std::abs(pi_estimate - M_PI) / M_PI * 100) << "%" << std::endl;

    return 0;
}

