#include <iostream>
#include "cuda_utils.h"

using namespace std;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n

__global__ void compute_G_AO(double *alphavec_cache, int alphavec_cache_size, double *old_alphavec_set,
                             double *observ_matrix, int *trans_matrix) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < alphavec_cache_size){

        int i = (int) l_idx / (numactions * numobservs * numstates);
        int m_idx = i * numactions * numobservs * numstates;
        int o = (int) (l_idx - m_idx) / (numactions * numstates);
        int a = (int) (l_idx - m_idx - o*numactions*numstates) / numstates;
        int s = (int) (l_idx - m_idx - o*numactions*numstates) - (a * numstates);

        // Access transition matrix to get s_prime
        int t_idx = a*numstates + s;
        int s_prime = trans_matrix[t_idx];

        // Access observation matrix to get O(a,s',o)
        int o_idx = o*numactions*numstates + a*numstates + s_prime;
        double observ_prob = observ_matrix[o_idx];

        // Access alpha vector set to get alpha[i][s']
        int alpha_idx = i*numstates + s_prime;
        double alpha_i_sprime = old_alphavec_set[alpha_idx];

        alphavec_cache[l_idx] = observ_prob * alpha_i_sprime;

        l_idx += blockDim.x * gridDim.x;

//        // Compute linear range to access the alphavec matrix
//        int alphavec_start_idx = i*numstates;
//        int alphavec_end_idx = i*numstates + numstates-1;
//
//        int _num_blocks = (int) ((numstates+num_threads-1) / num_threads );
//
//        range_scalar_product<<<_num_blocks, num_threads>>>(observ_prob, old_alphavec_set, numalphavecs*numstates,
//                                                          dev_prod, alphavec_start_idx, alphavec_end_idx);
//
//        int start_idx = l_idx;
//        int end_idx = l_idx + numstates-1;
//
//        assign2to1_range<<<_num_blocks, num_threads>>>(alphavec_cache, dev_prod, numstates, start_idx, end_idx);


    }



}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    // Initialize transition matrix, observation matrix, alpha vecs
    int trans_matrix[numactions][numstates] = {
            0, 1,
            1, 1,
    };

    double observ_matrix[numobservs][numactions][numstates] = {
            { {0.1, 0.2}, {0.1, 0.2} },
            { {0.1, 0.2}, {0.1, 0.2} }
    };

    double old_alphavec_set[numalphavecs][numstates] = {
            1, 2,
            0, 3,
    };

    // Copy transition and observation matrix, alpha vecs
    int *dev_trans_matrix;
    double *dev_observ_matrix, *dev_old_alphavec_set;

    int space_trans_matrix = numactions * numstates * sizeof(int);
    int space_observ_matrix = numactions * numstates * numobservs * sizeof(double);
    int space_alphavecs_set = numstates * numalphavecs * sizeof(double);

    HANDLE_ERROR(cudaMalloc((void **) &dev_trans_matrix, space_trans_matrix));
    HANDLE_ERROR(cudaMemcpy(dev_trans_matrix, trans_matrix, space_trans_matrix, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &dev_observ_matrix, space_observ_matrix));
    HANDLE_ERROR(cudaMemcpy(dev_observ_matrix, observ_matrix, space_observ_matrix, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &dev_old_alphavec_set, space_alphavecs_set));
    HANDLE_ERROR(cudaMemcpy(dev_old_alphavec_set, old_alphavec_set, space_alphavecs_set, cudaMemcpyHostToDevice));

    // Initialize G_AO
    double alphavec_cache[numalphavecs][numobservs][numactions][numstates];
    int alphavec_cache_size = numactions * numobservs * numstates * numalphavecs;
    int alphavec_cache_space = alphavec_cache_size * sizeof(double);

    double *dev_alphavec_cache;
    HANDLE_ERROR(cudaMalloc((void **) &dev_alphavec_cache, alphavec_cache_space));

    // Num blocks
    int num_blocks = (int) (alphavec_cache_size + num_threads - 1) / num_threads;

//    // Initialize dev_prod for inner scalar product computation
//    double *dev_prod;
//    int alphavec_space = numstates * sizeof(double);
//    HANDLE_ERROR(cudaMalloc((void **) &dev_prod, alphavec_space));

    //
    compute_G_AO<<<num_blocks, num_threads>>>(dev_alphavec_cache, alphavec_cache_size, dev_old_alphavec_set,
                                              dev_observ_matrix, dev_trans_matrix);

    HANDLE_ERROR( cudaMemcpy( alphavec_cache, dev_alphavec_cache,
                              alphavec_cache_space, cudaMemcpyDeviceToHost ) );


    // Print
    cout << "Size G_AO: " << alphavec_cache_size << endl;
    cout << "Num blocks: " << num_blocks << endl;
    cout << endl;

    cout << "G_AO" << endl;
    cout << endl;

    for(int i=0; i<numalphavecs; i++){
        cout << "Alphavecs i = " << i << endl;
        cout << endl;
        for(int o=0; o<numobservs; o++){
            cout << "Layer o = " << o << endl;
            for(int a=0; a<numactions; a++){
                for(int s=0; s<numstates; s++){
                    cout << alphavec_cache[i][o][a][s] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }


    return 0;
}
