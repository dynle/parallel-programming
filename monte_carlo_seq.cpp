#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>

int main() {
    long long num_samples = 100000000; // Total number of random points
    long long circle_count = 0;       // Points that fall inside the circle
    double x, y;

    // Seed the random number generator
    srand(time(NULL));

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    for (long long i = 0; i < num_samples; ++i) {
        // Generate random x, y in [-1, 1]
        x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        y = (double)rand() / RAND_MAX * 2.0 - 1.0;

        // Check if the point is in the circle
        if (x * x + y * y <= 1.0) {
            circle_count++;
        }
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double pi_estimate = 4.0 * circle_count / num_samples;
    
    std::cout << "Total Samples: " << num_samples << std::endl;
    std::cout << "Points in Circle: " << circle_count << std::endl;
    std::cout << "Estimated Pi: " << pi_estimate << std::endl;
    std::cout << "Execution Time: " << duration.count()/1000.0 << " seconds" << std::endl;
    std::cout << "Accuracy: " << (100 - std::abs(pi_estimate - M_PI) / M_PI * 100) << "%" << std::endl;

    return 0;
}