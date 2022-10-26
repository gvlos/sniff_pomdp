#include <iostream>
#include "cuda_utils.h"

using namespace std;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n
const int numbeliefs = 2; // |B|

__global__ void compute_backup(double *B_KP, int B_KP_size, double *G_AB, int *hat_star_G_AB){

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < B_KP_size){

        int j = (int) l_idx / numstates;
        int a_star_bj = hat_star_G_AB[j];
        int s = l_idx - j*numstates;

        B_KP[l_idx] = G_AB[j*numactions*numstates + a_star_bj*numstates + s];

        l_idx += blockDim.x * gridDim.x;
    }

}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

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
    int hat_star_G_AB[numbeliefs] = {
            1,1,
    };

    int hat_star_G_AB_size = numbeliefs;
    int hat_star_G_AB_space = hat_star_G_AB_size * sizeof(int);

    int *dev_hat_star_G_AB;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_star_G_AB, hat_star_G_AB_space));
    HANDLE_ERROR(cudaMemcpy(dev_hat_star_G_AB, hat_star_G_AB, hat_star_G_AB_space, cudaMemcpyHostToDevice));

    // Compute backup
    double B_KP[numbeliefs][numstates];

    int B_KP_size = numbeliefs * numstates;
    int B_KP_space = B_KP_size * sizeof(double);

    double *dev_B_KP;
    HANDLE_ERROR(cudaMalloc((void **) &dev_B_KP, B_KP_space));

    // Num blocks
    int num_blocks = (int) (B_KP_size + num_threads - 1) / num_threads;

    compute_backup<<<num_blocks, num_threads>>>(dev_B_KP, B_KP_size, dev_G_AB, dev_hat_star_G_AB);

    HANDLE_ERROR( cudaMemcpy( B_KP, dev_B_KP, B_KP_space, cudaMemcpyDeviceToHost ) );

    // Print
    cout << "Size B_KP: " << B_KP_size << endl;
    cout << endl;

    cout << "B_KP" << endl;
    cout << endl;

    for(int j=0; j<numbeliefs; j++){
        for(int s=0; s<numstates; s++){
            cout << B_KP[j][s] << " ";
        }
        cout << endl;
    }

    return 0;
}
