# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>
# include <float.h>

/*
* In this code, the weights  and biases
* are named a little differently from the
* provided diagram in Figure 4.
*
* Figure 4 | My Code
* w_{j}    | w_{j-1},    for 1 <= j <= N
* w_{j}    | m_{j-N-1},  for N+1 <= j <= 2N
* b_{j}    | b_{j-1},    for 1 <= j <= N
* b_{N+1}  | c_{0}
* 
* ws, bs, ms and c0 are collectively called
* "parameters."
*/


double activation_function(double x) {
    /* Activation function. */
    return log(1+exp(x));
}

double d_activation_function(double x) {
    /* Derivative of the activation function. */
    return exp(x)/(1+exp(x));
}

double y(double x, double* ws, double* bs, double* ms, double c0, int layer_size){
    /*
    * Given the parameters ws, bs, ms, c0 of the neural network
    * and an input x, evaluate its output y.
    */
    double sum = 0;
    int i;
    for (i = 0; i < layer_size; i++) {
        sum += ms[i] * activation_function(ws[i]*x+bs[i]);
    }
    return sum+c0;
}

void ys(double* xs, double*  ws, double* bs, double* ms, double c0, int layer_size, double* ys, int ys_len) {
    /* Evaluates y(x) for each element x in xs and saves it to ys.*/
    int i;
    for (i = 0; i < ys_len; i++) {
        ys[i] = y(xs[i], ws, bs, ms, c0, layer_size);
    }
}

double error(double y_observed, double y_predicted) {
    /* Error. */
    return y_observed - y_predicted;
}

double mse(double* ys_observed, double* ys_predicted, int ys_observed_len) {
    /* Mean Square Error (MSE). */
    double sum = 0;
    int i;
    for (i = 0; i < ys_observed_len; i++) {
        double err = error(ys_observed[i], ys_predicted[i]);
        sum += err*err;
    }
    return sum / ys_observed_len;
}


void fast_di_mse(double* ys_observed, double* ys_predicted, int ys_observed_len,
                double* xs, double*  ws, double* bs, double* ms, double c0, int layer_size,
                int j, double* grad_mse, double* multiplicative_factors) {
    /*
    * Computes the derivative of MSE with respect to w_{j}, b_{j}, m_{j} and saves them to grad_mse.
    * Takes advantage of common factors in each expression. This reduces execution time by ~8%.
    */
    double sum = 0;
    int i;

    /* bs */
    for (i = 0; i < ys_observed_len; i++) {
        multiplicative_factors[i] = ms[j] * d_activation_function(ws[j]*xs[i]+bs[j]);
        sum += multiplicative_factors[i] * error(ys_observed[i], ys_predicted[i]);
    }
    grad_mse[j+layer_size] = -2*sum/ys_observed_len;

    /* ws */
    sum = 0;
    for (i = 0; i < ys_observed_len; i++) {
        sum += xs[i]*multiplicative_factors[i] * error(ys_observed[i], ys_predicted[i]);
    }
    grad_mse[j] = -2*sum/ys_observed_len;

    /* ms */
    sum = 0;
    for (i = 0; i < ys_observed_len; i++) {
        sum +=  activation_function(ws[j]*xs[i]+bs[j]) * error(ys_observed[i], ys_predicted[i]);
    }
    grad_mse[j+2*layer_size] = -2*sum/ys_observed_len;
}

void step_w(double* params, int params_len, double rate, double* grad_mse) {
    /* Updates parameters based on the gradient of MSE. */
    int i;
    for (i = 0; i < params_len; i++) {
        params[i] = params[i] - rate * grad_mse[i];
    }
}

int stop_condition(double previous_mse, double current_mse, double* previous_params, double* current_params,
                    int params_len, double mse_threshold, double params_threshold, int max_iterations, int current_iteration) {
    /* Returns 1 if gradient descent should stop, and 0 otherwise. */
    
    /* MSE difference. */
    double mse_difference = fabs(current_mse-previous_mse);
    
    /* Magnitude^2 of parameter difference vector. */
    double params_difference = 0;
    int i;
    for (i = 0; i < params_len; i++) {
        double diff = current_params[i]-previous_params[i];
        params_difference += diff * diff;
    }
    /* Check if thresholds were surpassed or if next iteration will exceed the limit. */
    if ((mse_difference < mse_threshold && params_difference < params_threshold*params_threshold) || current_iteration >= max_iterations) {
        return 1;
    } else {
        return 0;
    }
}
/*
* We expose these two variables to the global scope
* because they will be exported together with the
* other network parameters.
*/
int current_iteration = 0;
double current_mse;

double fit(double* xs, double* ys_observed, int ys_observed_len, double* ws, double* bs, double* ms, double c0,
            int layer_size, double mse_threshold, double params_thresold, double rate, int max_iterations) {
    /* Finds the best parameters for the neural network. */

    /* Compute the initial predicted y values. */
    double* ys_predicted = (double*)malloc(sizeof(double)*ys_observed_len);
    ys(xs, ws, bs, ms, c0, layer_size, ys_predicted, ys_observed_len);


    /*
    * Initialize parameter array and gradient array.
    * Note that we keep track of a previous and current
    * parameters array in case we want to pass a threshold
    * to the parameters as we do for MSE. We also keep a 
    * multiplicative factors array to speed up calculations
    * of the MSE gradient by ~8%, as detailed in fast_di_mse().
    */
    int params_len = 3*layer_size + 1;
    double* grad_mse = (double*)malloc(sizeof(double)*params_len);
    double* params = (double*)malloc(sizeof(double)*params_len);
    double* previous_params = (double*)malloc(sizeof(double)*params_len);
    double* multiplicative_factors = (double*)malloc(sizeof(double)*ys_observed_len);
    
    int i;
    for (i = 0; i < layer_size; i++) {
        params[i] = ws[i];
        previous_params[i] = 0;
    }
    for (i = layer_size; i < 2*layer_size; i++) {
        params[i] = bs[i];
        previous_params[i] = 0;
    }
    for (i = 2*layer_size; i < 3*layer_size; i++) {
        params[i] = ms[i];
        previous_params[i] = 0;
    }
    previous_params[params_len-1] = 0;
    params[params_len-1] = c0;

    /* Compute the initial Mean Square Error (MSE). */
    double previous_mse = 0;
    current_mse = mse(ys_observed, ys_predicted, ys_observed_len);

    
    while (stop_condition(previous_mse, current_mse, previous_params, params, params_len,
                            mse_threshold, params_thresold, max_iterations, current_iteration) == 0) {
        
        /* Set previous parameters and MSE to the current parameters and MSE. */
        for (i = 0; i < params_len; i++) {
            previous_params[i] = params[i];
        }
        previous_mse = current_mse;

        /* Calculate the gradient. */
        for (i = 0; i < layer_size; i++) {
            fast_di_mse(ys_observed, ys_predicted, ys_observed_len, xs, ws, bs, ms, c0,
                        layer_size, i, grad_mse, multiplicative_factors);
        }

        /* Gradient of c0 */
        double sum = 0; /* This declaration should be optimized by the compiler. */
        for (i = 0; i < ys_observed_len; i++) {
            sum += error(ys_observed[i], ys_predicted[i]);
        }
        grad_mse[params_len-1] = -2*sum/ys_observed_len;

        /* Update parameters. */
        for (i = 0; i < layer_size; i++) {
            ws[i] = ws[i]-rate*grad_mse[i];
            params[i] = ws[i];

            bs[i] = bs[i]-rate*grad_mse[i+layer_size];
            params[i+layer_size] = ws[i];

            ms[i] = ms[i]-rate*grad_mse[i+2*layer_size];
            params[i+2*layer_size] = ms[i];
        }
        c0 = c0-rate*grad_mse[params_len-1];
        params[params_len-1] = c0;

        /* Update MSE and predicted y values. */
        ys(xs, ws, bs, ms, c0, layer_size, ys_predicted, ys_observed_len);
        current_mse = mse(ys_observed, ys_predicted, ys_observed_len);
        current_iteration += 1;

        }
    
    /* Free some memory :) */
    free(ys_predicted);
    free(grad_mse);
    free(params);
    free(previous_params);
    free(multiplicative_factors);

    printf("Fit Complete. MSE = %e\n", current_mse);
    printf("Total Iterations: %d\n", current_iteration);
    return c0;
}

/*
* These random functions come from Eric Roberts'
* "The Art and Science of C".
*/

void InitRandom(void)
{
  srand((int) time(NULL));
}

#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif

double RandomReal(double low, double high)
{
  double d;

  d = (double) rand() / ((double) RAND_MAX + 1);
  return (low + d * (high - low));
}

/*
* This is based on central limit theorem,
* so the distribution is capped to [-n, n]
* which should be good enough for large n
* values like 6 or 7. 
*/
double unit_gaussian(int n) {
    double sum = 0;
    int i;
    for (i = 0; i < 2*n; i++) {
        sum += RandomReal(0,1);
    }
    return sum-n;
}

int main(int argc, char *argv[]) {
    # define data_len 10 /* Number of data points*/
    int layer_size = atoi(argv[1]); /* Layer Size */
    
    /* Data arrays. */
    double xs_observed[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double ys_observed[] = {4.11, 6.07, 8.01, 9.46, 10, 9.46, 8.01, 6.07, 4.11, 2.49};

    /* Parameters arrays */
    double* xs = (double*) malloc(sizeof(double*) * data_len);
    double* ws = (double*) malloc(sizeof(double*) * layer_size);
    double* bs = (double*) malloc(sizeof(double*) * layer_size);
    double* ms = (double*) malloc(sizeof(double*) * layer_size);
    double c0 = 0;
    int i;
    for(i=0; i < data_len; i++) {
        xs[i] = xs_observed[i];
    }

    /* 
    * Initialize the weights to a Normal distribution
    * (capped to [-6, 6]), which should be good enough
    */
    InitRandom();
    
    for (i=0; i < layer_size; i++) {
        ws[i] = unit_gaussian(6);
        ms[i] = unit_gaussian(6);
        bs[i] = 0;
    }

    /* Thresholds and other scalar constants. */
    double mse_threshold = pow(10, -6);
    double ws_threshold = HUGE_VAL;
    double rate = pow(10, -4);
    int max_iterations = pow(10,6);

    /*Calculate the time taken by fit()*/
    clock_t t;
    t = clock();
    c0 = fit(xs, ys_observed, data_len, ws, bs, ms, c0, layer_size, mse_threshold, ws_threshold, rate, max_iterations);
    t = clock() - t;
    /* time_taken is measured in seconds */
    double time_taken = ((double)t)/CLOCKS_PER_SEC; 
    
    /* The follwoing can be uncommented for a more verbose output. */
    /*
    for (i = 0; i < layer_size; i++) {
        printf("ws[%d] = %.15f\n", i, ws[i]);
        printf("bs[%d] = %.15f\n", i, bs[i]);
        printf("ms[%d] = %.15f\n", i, ms[i]);
    }
    printf("c0 = %.15f\n", c0);
    */
   
    printf("fit() took %f seconds to execute \n", time_taken);

    /*
    * Here, we process data to be exported, merging
    * all ws and ms to ws2 and all bs and c0 to bs2.
    */
    double* ys_predicted = (double*)malloc(sizeof(double)*data_len);
    ys(xs, ws, bs, ms, c0, layer_size, ys_predicted, data_len);
    double* ws2 = (double*) malloc(sizeof(double*) * 2*layer_size);
    double* bs2 = (double*) malloc(sizeof(double*) * (layer_size+1));\
    for(i=0; i < layer_size; i++) {
        ws2[i] = ws[i];
    }
    for(i=layer_size; i < 2*layer_size; i++) {
        ws2[i] = ms[i-layer_size];
    }
    for(i=0; i < layer_size; i++) {
        bs2[i] = bs[i];
    }
    bs2[layer_size] = c0;

    /* Free unused memory :) */
    free(ws);
    free(bs);
    free(ms);

    /*
    * Export the data as a .csv file. Note
    * the various conditionals because of
    * the possible difference in sizes for
    * data_len and layer_size.
    */
    FILE *file = fopen(argv[2], "a");
    fprintf(file, "x,y,yfit,w,b,k,mse,t\n");

    if (layer_size+1 <= data_len) {
        fprintf(file, "%.16e,%.16e,%.16e,%.16e,%.16e,%d,%.16e,%.16e\n", xs_observed[0], ys_observed[0], ys_predicted[0], ws2[0], bs2[0], current_iteration, current_mse, time_taken);
        for (i=1; i < layer_size+1; i++) {
            fprintf(file, "%.16e,%.16e,%.16e,%.16e,%.16e,,,\n", xs_observed[i], ys_observed[i], ys_predicted[i], ws2[i], bs2[i]);
        }
                
        if (2*layer_size <= data_len) {
            for (i = layer_size+1; i < 2*layer_size; i++) {
             fprintf(file, "%.16e,%.16e,%.16e,%.16e,,,,\n", xs_observed[i], ys_observed[i], ys_predicted[i], ws2[i]);
            }
            for (i = 2*layer_size; i < data_len; i++) {
                fprintf(file, "%.16e,%.16e,%.16e,,,,,\n", xs_observed[i], ys_observed[i], ys_predicted[i]);
            }
        } else {
            for (i = layer_size+1; i < data_len; i++) {
             fprintf(file, "%.16e,%.16e,%.16e,%.16e,,,,\n", xs_observed[i], ys_observed[i], ys_predicted[i], ws2[i]);
            }
            for (i = data_len; i < 2*layer_size; i++) {
                fprintf(file, ",,,%.16e,,,,\n", ws2[i]);
            }
        }

    } else {
        fprintf(file, "%.16e,%.16e,%.16e,%.16e,%.16e,%d,%.16e,%.16e\n", xs_observed[0], ys_observed[0], ys_predicted[0], ws2[0], bs2[0], current_iteration, current_mse, time_taken);        
        for (i = 1; i < data_len; i++) {
            fprintf(file, "%.16e,%.16e,%.16e,%.16e,%.16e,,,\n", xs_observed[i], ys_observed[i], ys_predicted[i], ws2[i], bs2[i]);
        }
        for (i = data_len; i < layer_size+1; i++) {
            fprintf(file, ",,,%.16e,%.16e,,,\n", ws2[i], bs2[i]);
        }
        for (i = layer_size+1; i < 2*layer_size; i++) {
            fprintf(file, ",,,%.16e,,,,\n", ws2[i]);
        }
    }

    fclose(file);

    /* Free more memory :) */
    free(xs);
    free(ys_predicted);
    free(ws2);
    free(bs2);

    return 0;
}