#include <iostream>
#include "cuda_utils.h"

using namespace std;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n

__global__ void compute_t_ao(double *t_ao, int *trans_matrix, double *observ_matrix, int a, int o) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < numstates){

        int s_prime = trans_matrix[a*numstates + l_idx];
        t_ao[l_idx] = observ_matrix[o*numactions*numstates + a*numstates + s_prime];

        l_idx += blockDim.x * gridDim.x;
    }
}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    // Initialize inverse transition matrix, observation matrix, alpha vecs
    int trans_matrix[numactions][numstates] = {
            0, 1,
            1, 1,
    };
    int *dev_trans_matrix;
    int space_trans_matrix = numactions * numstates * sizeof(int);
    HANDLE_ERROR(cudaMalloc((void **) &dev_trans_matrix, space_trans_matrix));
    HANDLE_ERROR(cudaMemcpy(dev_trans_matrix, trans_matrix, space_trans_matrix, cudaMemcpyHostToDevice));


    double observ_matrix[numobservs][numactions][numstates] = {
            { {0.1, 0.2}, {0.1, 0.2} },
            { {0.1, 0.2}, {0.1, 0.2} }
    };
    double *dev_observ_matrix;
    int space_observ_matrix = numactions * numstates * numobservs * sizeof(double);
    HANDLE_ERROR(cudaMalloc((void **) &dev_observ_matrix, space_observ_matrix));
    HANDLE_ERROR(cudaMemcpy(dev_observ_matrix, observ_matrix, space_observ_matrix, cudaMemcpyHostToDevice));

    // a, o
    int a = 1;
    int o = 0;

    // Initialize t_ao
    double t_ao[numstates];

    int space_t_ao = numstates * sizeof(double);
    double *dev_t_ao;
    HANDLE_ERROR(cudaMalloc((void **) &dev_t_ao, space_t_ao));

    // Num blocks
    int num_blocks = (int) (numstates + num_threads - 1) / num_threads;

    //
    compute_t_ao<<<num_blocks, num_threads>>>(dev_t_ao, dev_trans_matrix, dev_observ_matrix, a, o);

    HANDLE_ERROR( cudaMemcpy( t_ao, dev_t_ao, space_t_ao, cudaMemcpyDeviceToHost ) );

    // Compute P(o|b,a)
    double belief[numstates] = {
            0.1, 0.9,
            };

    double dot = 0;
    for (int s=0; s<numstates; s++){
        dot += belief[s] * t_ao[s];
    }

    // Print
    cout << "Size t_ao: " << numstates << endl;
    cout << endl;

    cout << "t_ao" << endl;
    cout << endl;

    for (int s=0; s<numstates; s++){
        cout << t_ao[s] << " ";
    }
    cout << endl;
    cout << endl;

    cout << "p(o|b,a) = " << dot << endl;

    return 0;
}
