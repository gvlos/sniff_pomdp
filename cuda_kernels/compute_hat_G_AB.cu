#include <iostream>
#include "cuda_utils.h"

using namespace std;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n
const int numbeliefs = 2; // |B|

__global__ void compute_hat_G_AB(double *hat_G_AB, int hat_G_AB_size, double *beliefvec_set, double *G_AB) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < hat_G_AB_size){

        int j = (int) l_idx / numactions;
        int a = l_idx - j*numactions;

        double sum = 0;
        double g_a_bj, bj;
        for (int s=0; s<numstates; s++){
            g_a_bj = G_AB[j*numactions*numstates + a*numstates + s];
            bj = beliefvec_set[j*numstates + s];
            sum += bj * g_a_bj;
        }

        hat_G_AB[l_idx] = sum;

        l_idx += blockDim.x * gridDim.x;
    }



}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    // B
    double beliefvec_set[numbeliefs][numstates] = {
            0.7, 0.3,
            0.4, 0.6,
            };
    int beliefvec_set_size = numbeliefs * numstates;
    int beliefvec_set_space = beliefvec_set_size * sizeof(double);

    double *dev_beliefvec_set;
    HANDLE_ERROR(cudaMalloc((void **) &dev_beliefvec_set, beliefvec_set_space));
    HANDLE_ERROR(cudaMemcpy(dev_beliefvec_set, beliefvec_set, beliefvec_set_space, cudaMemcpyHostToDevice));

    // G_AB
    double G_AB[numbeliefs][numactions][numstates] = {
            {
                0.2, 1.8,
                2.2, 1.2,
            },
            {
                0, 2.2,
                2.2, 1.2,
            }
    };

    int G_AB_size = numactions * numstates * numbeliefs;
    int G_AB_space = G_AB_size * sizeof(double);

    double *dev_G_AB;
    HANDLE_ERROR(cudaMalloc((void **) &dev_G_AB, G_AB_space));
    HANDLE_ERROR(cudaMemcpy(dev_G_AB, G_AB, G_AB_space, cudaMemcpyHostToDevice));

    // Init hat_G_AB
    double hat_G_AB[numbeliefs][numactions];
    int hat_G_AB_size = numbeliefs * numactions;
    int hat_G_AB_space = hat_G_AB_size * sizeof(double);

    double *dev_hat_G_AB;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_G_AB, hat_G_AB_space));

    // Num blocks
    int num_blocks = (int) (hat_G_AB_size + num_threads - 1) / num_threads;

    // call kernel
    compute_hat_G_AB<<<num_blocks, num_threads>>>(dev_hat_G_AB, hat_G_AB_size, dev_beliefvec_set, dev_G_AB);

    // copy back
    HANDLE_ERROR( cudaMemcpy( hat_G_AB, dev_hat_G_AB, hat_G_AB_space, cudaMemcpyDeviceToHost ) );


    // hat_star_G_AB
    int hat_star_G_AB[numbeliefs];

    for (int j=0; j<numbeliefs; j++){
        double max = -999;
        int argmax = -1;
        for (int a=0; a<numactions; a++){
            if (hat_G_AB[j][a] > max){
                max = hat_G_AB[j][a];
                argmax = a;
            }
        }
        hat_star_G_AB[j] = argmax;
    }

    // Print
    cout << "Size hat_G_AB: " << hat_G_AB_size << endl;
    cout << endl;

    cout << "hat_G_AB" << endl;
    cout << endl;

    for(int j=0; j<numbeliefs; j++){
        for(int a=0; a<numactions; a++){
            cout << hat_G_AB[j][a] << " ";
        }
        cout << endl;
    }

    cout << "Size hat_star_G_AB: " << numbeliefs << endl;
    cout << endl;

    cout << "hat_G_AB" << endl;
    cout << endl;

    for(int j=0; j<numbeliefs; j++){
        cout << hat_star_G_AB[j] << " ";
    }
    cout << endl;

    return 0;
}
