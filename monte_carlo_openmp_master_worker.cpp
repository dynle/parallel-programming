#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <ctime>
#include <cmath>

int main() {
    long long num_samples = 100000000;
    long long circle_count = 0;

    // Shared storage for random points, accessible by all threads.
    std::vector<double> x_coords(num_samples);
    std::vector<double> y_coords(num_samples);

    #pragma omp parallel
    {
        // === MASTER'S TASK: The random generator runs as the master. ===
        // This entire block is executed only by the master thread.
        #pragma omp master
        {
            // Use a high-quality random number engine to ensure accuracy.
            std::mt19937 engine(time(NULL));
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            for (long long i = 0; i < num_samples; ++i) {
                // The master generates ALL random points and stores them.
                x_coords[i] = dist(engine);
                y_coords[i] = dist(engine);
            }
        } // Implicit barrier: Workers wait here until the master is finished.

        // === WORKERS' TASK: Estimation is parallelized. ===
        // All threads, including the master, now work together on the generated data.
        #pragma omp for reduction(+:circle_count)
        for (long long i = 0; i < num_samples; ++i) {
            if (x_coords[i] * x_coords[i] + y_coords[i] * y_coords[i] <= 1.0) {
                circle_count++;
            }
        }
    } // End of the parallel region.

    double pi_estimate = 4.0 * circle_count / num_samples;
    std::cout << "Total Samples: " << num_samples << std::endl;
    std::cout << "Points in Circle: " << circle_count << std::endl;
    std::cout << "Estimated Pi: " << pi_estimate << std::endl;
    std::cout << "Accuracy: " << (100 - std::abs(pi_estimate - M_PI) / M_PI * 100) << "%" << std::endl;

    return 0;
}