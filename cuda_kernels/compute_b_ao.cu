#include <iostream>
#include "cuda_utils.h"

using namespace std;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n

__global__ void compute_hat_b_ao(double *b_ao, double *belief, double *observ_matrix, int *inv_trans_matrix, int a, int o) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < numstates){

        int s = inv_trans_matrix[a*numstates + l_idx];
        b_ao[l_idx] = observ_matrix[o*numactions*numstates + a*numstates + l_idx] * belief[s];

        l_idx += blockDim.x * gridDim.x;
    }
}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    // Initialize inverse transition matrix, observation matrix, alpha vecs
    int inv_trans_matrix[numactions][numstates] = {
            0, 1,
            0, 1,
    };
    int *dev_inv_trans_matrix;
    int space_inv_trans_matrix = numactions * numstates * sizeof(int);
    HANDLE_ERROR(cudaMalloc((void **) &dev_inv_trans_matrix, space_inv_trans_matrix));
    HANDLE_ERROR(cudaMemcpy(dev_inv_trans_matrix, inv_trans_matrix, space_inv_trans_matrix, cudaMemcpyHostToDevice));


    double observ_matrix[numobservs][numactions][numstates] = {
            { {0.1, 0.2}, {0.1, 0.2} },
            { {0.1, 0.2}, {0.1, 0.2} }
    };
    double *dev_observ_matrix;
    int space_observ_matrix = numactions * numstates * numobservs * sizeof(double);
    HANDLE_ERROR(cudaMalloc((void **) &dev_observ_matrix, space_observ_matrix));
    HANDLE_ERROR(cudaMemcpy(dev_observ_matrix, observ_matrix, space_observ_matrix, cudaMemcpyHostToDevice));

    double belief[numstates] = {
            0.1, 0.9,
    };
    double *dev_belief;
    int space_belief = numstates * sizeof(double);
    HANDLE_ERROR(cudaMalloc((void **) &dev_belief, space_belief));
    HANDLE_ERROR(cudaMemcpy(dev_belief, belief, space_belief, cudaMemcpyHostToDevice));

    // a, o
    int a = 1;
    int o = 0;

    // Initialize new belief b_ao
    double b_ao[numstates];

    double *dev_b_ao;
    HANDLE_ERROR(cudaMalloc((void **) &dev_b_ao, space_belief));

    // Num blocks
    int num_blocks = (int) (numstates + num_threads - 1) / num_threads;

    //
    compute_hat_b_ao<<<num_blocks, num_threads>>>(dev_b_ao, dev_belief, dev_observ_matrix, dev_inv_trans_matrix, a, o);

    HANDLE_ERROR( cudaMemcpy( b_ao, dev_b_ao, space_belief, cudaMemcpyDeviceToHost ) );

    // Apply normalization
    double sum = 0;
    for (int s=0; s<numstates; s++){
        sum += b_ao[s];
    }

    double beta_norm = 1/sum;

    for (int s=0; s<numstates; s++){
        b_ao[s] *= beta_norm;
    }

    // Print
    cout << "Size b_ao: " << numstates << endl;
    cout << endl;

    cout << "b_ao" << endl;
    cout << endl;

    for (int s=0; s<numstates; s++){
        cout << b_ao[s] << " ";
    }
    cout << endl;


    return 0;
}
