#include <omp.h>
#include <iostream>

int main() {
#pragma omp parallel
    {
#pragma omp critical
        {
            std::cout << "Printing from thread: " << omp_get_thread_num()
                      << "\n";
        }
    }
}
