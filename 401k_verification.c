#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define FEATURES 5
#define SAMPLES 10000 // Assuming this is the number of samples in the 401K dataset

// Structure to hold each sample's data
typedef struct {
    double features[FEATURES];
    int treatment;
    double outcome;
} Sample;

// Generate a uniform random number between 0 and 1
double uniform_rand() {
    return (double)rand() / RAND_MAX;
}

// Generate a standard normal random number using Box-Muller transform
double normal_rand() {
    double u1 = uniform_rand();
    double u2 = uniform_rand();
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

// Sigmoid function for logistic probability
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Generate simulation data according to the specified process
void generate_data(Sample* data, int n) {
    for (int i = 0; i < n; i++) {
        // Generate X ~ Unif([-2,2]^5)
        for (int j = 0; j < FEATURES; j++) {
            data[i].features[j] = -2 + 4 * uniform_rand();
        }
        
        // Generate A|X ~ Bern(sigma(0.75X_0 + 0.5))
        double p = sigmoid(0.75 * data[i].features[0] + 0.5);
        data[i].treatment = (uniform_rand() < p) ? 1 : 0;
        
        // Generate Y ~ N((2A-1)(X_0 + 1) - 2sin((4A-2)X_0), 1)
        double mean = (2 * data[i].treatment - 1) * (data[i].features[0] + 1) - 
                      2 * sin((4 * data[i].treatment - 2) * data[i].features[0]);
        data[i].outcome = mean + normal_rand();
    }
}

// Calculate the true Conditional Average Treatment Effect (CATE)
double true_cate(double x0) {
    return 2 * (x0 + 1) - 2 * (sin(2 * x0) - sin(-2 * x0));
}

// Estimate lower and upper bounds for CATE using B-Learner method
void estimate_bounds(Sample* data, int n, double log_gamma, double* lower_bound, double* upper_bound) {
    double gamma = exp(log_gamma);
    double lambda = gamma + 1;  // Lambda = gamma + 1
    double tau = gamma / lambda;

    for (int i = 0; i < n; i++) {
        // Estimate propensity score e(x)
        double e = sigmoid(0.75 * data[i].features[0] + 0.5);
        
        // Estimate outcome mu(x,a)
        double mu = (2 * data[i].treatment - 1) * (data[i].features[0] + 1) - 
                    2 * sin((4 * data[i].treatment - 2) * data[i].features[0]);
        
        // Calculate R(z,q) (simplified version without quantile estimation)
        double R = lambda * (data[i].outcome - mu) / 2;
        
        // Calculate rho(x,a) (simplified version)
        double rho = mu + R / lambda;
        
        // Calculate pseudo-outcome
        double pseudo_outcome = data[i].outcome - mu + 
                                (data[i].treatment - e) / (e * (1 - e)) * 
                                (data[i].outcome - mu);
        
        // Estimate bounds
        lower_bound[i] = pseudo_outcome - gamma * fabs(data[i].outcome - mu);
        upper_bound[i] = pseudo_outcome + gamma * fabs(data[i].outcome - mu);
    }
}

int main() {
    // Seed the random number generator
    srand(time(NULL));
    
    // Allocate memory for data and results
    Sample* data = malloc(SAMPLES * sizeof(Sample));
    double* lower_bounds = malloc(SAMPLES * sizeof(double));
    double* upper_bounds = malloc(SAMPLES * sizeof(double));
    double* true_cates = malloc(SAMPLES * sizeof(double));

    // Check if memory allocation was successful
    if (!data || !lower_bounds || !upper_bounds || !true_cates) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Generate simulation data
    generate_data(data, SAMPLES);

    // Calculate true CATEs
    for (int i = 0; i < SAMPLES; i++) {
        true_cates[i] = true_cate(data[i].features[0]);
    }

    // Print CSV header
    printf("log_gamma,avg_lower_bound,avg_upper_bound,avg_true_cate,coverage,pct_negative_lb\n");

    // Estimate bounds for different log_gamma values
    for (double log_gamma = 0; log_gamma <= 1.0; log_gamma += 0.1) {
        estimate_bounds(data, SAMPLES, log_gamma, lower_bounds, upper_bounds);
        
        // Calculate averages, coverage, and percentage of negative lower bounds
        double avg_lower = 0, avg_upper = 0, avg_true = 0;
        int covered = 0, negative_lb = 0;
        for (int i = 0; i < SAMPLES; i++) {
            avg_lower += lower_bounds[i];
            avg_upper += upper_bounds[i];
            avg_true += true_cates[i];
            if (lower_bounds[i] <= true_cates[i] && true_cates[i] <= upper_bounds[i]) {
                covered++;
            }
            if (lower_bounds[i] < 0) {
                negative_lb++;
            }
        }
        avg_lower /= SAMPLES;
        avg_upper /= SAMPLES;
        avg_true /= SAMPLES;
        double coverage = (double)covered / SAMPLES;
        double pct_negative_lb = (double)negative_lb / SAMPLES * 100;

        // Print results as CSV row
        printf("%.1f,%.4f,%.4f,%.4f,%.4f,%.2f\n", log_gamma, avg_lower, avg_upper, avg_true, coverage, pct_negative_lb);
    }

    // Free allocated memory
    free(data);
    free(lower_bounds);
    free(upper_bounds);
    free(true_cates);
    return 0;
}