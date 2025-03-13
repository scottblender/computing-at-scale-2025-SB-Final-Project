#include <cmath>  // For M_PI constant

// Function to wrap an angle to the range [0, 2π)
double wrapTo2Pi(double angle) {
    // Wrap the angle to the range [0, 2π)
    return fmod(angle, 2 * M_PI) + (fmod(angle, 2 * M_PI) < 0 ? 2 * M_PI : 0);
}
