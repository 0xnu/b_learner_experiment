#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define FEATURES 11
#define SAMPLES 10000 // Assuming this is the number of samples in the 401K dataset

// Structure to hold each sample's data
typedef struct {
    double age;
    double inc;
    double educ;
    double fsize;
    int marr;
    int two_earn;
    int db;
    int pira;
    int hown;
    int e401;
    double net_tfa;
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
        // Generate continuous features
        data[i].age = 18 + 47 * uniform_rand(); // Age between 18 and 65
        data[i].inc = 10000 + 190000 * uniform_rand(); // Income between 10k and 200k
        data[i].educ = 8 + 16 * uniform_rand(); // Education between 8 and 24 years
        data[i].fsize = 1 + 9 * uniform_rand(); // Family size between 1 and 10
        data[i].net_tfa = -100000 + 1100000 * uniform_rand(); // Net financial assets between -100k and 1M

        // Generate binary features
        data[i].marr = (uniform_rand() < 0.5) ? 0 : 1;
        data[i].two_earn = (uniform_rand() < 0.5) ? 0 : 1;
        data[i].db = (uniform_rand() < 0.3) ? 0 : 1;
        data[i].pira = (uniform_rand() < 0.4) ? 0 : 1;
        data[i].hown = (uniform_rand() < 0.6) ? 0 : 1;
        
        // Generate A|X ~ Bern(sigma(0.01*age + 0.00001*inc + 0.1*educ - 0.5))
        double p = sigmoid(0.01 * data[i].age + 0.00001 * data[i].inc + 0.1 * data[i].educ - 0.5);
        data[i].e401 = (uniform_rand() < p) ? 1 : 0;
        
        // Generate Y ~ N((2A-1)(age/10 + inc/50000), 1)
        double mean = (2 * data[i].e401 - 1) * (data[i].age/10 + data[i].inc/50000);
        data[i].outcome = mean + normal_rand();
    }
}

// Calculate the true Conditional Average Treatment Effect (CATE)
double true_cate(double age, double inc) {
    return 2 * (age/10 + inc/50000);
}

// Estimate lower and upper bounds for CATE using B-Learner method
void estimate_bounds(Sample* data, int n, double log_gamma, double* lower_bound, double* upper_bound) {
    double gamma = exp(log_gamma);
    double lambda = gamma + 1;  // Lambda = gamma + 1

    for (int i = 0; i < n; i++) {
        // Estimate propensity score e(x)
        double e = sigmoid(0.01 * data[i].age + 0.00001 * data[i].inc + 0.1 * data[i].educ - 0.5);
        
        // Estimate outcome mu(x,a)
        double mu = (2 * data[i].e401 - 1) * (data[i].age/10 + data[i].inc/50000);
        
        // Calculate R(z,q) (simplified version without quantile estimation)
        double R = lambda * (data[i].outcome - mu) / 2;
        
        // Calculate pseudo-outcome
        double pseudo_outcome = data[i].outcome - mu + 
                                (data[i].e401 - e) / (e * (1 - e)) * 
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
        true_cates[i] = true_cate(data[i].age, data[i].inc);
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