#include <iostream>
#include "cuda_utils.h"

using namespace std;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n
const int numbeliefs = 2; // |B|

__global__ void compute_hat_G_AO(double *hat_G_AO, int hat_G_AO_size, double *alphavec_cache, double *beliefvec_set,
                                 double *b_j, double *g_ao_i, double *partial_dot) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < hat_G_AO_size){

        int j = (int) l_idx / (numactions * numalphavecs * numobservs);
        int m_idx = j * numactions * numalphavecs * numobservs;
        int o = (int) (l_idx - m_idx) / (numactions * numalphavecs);
        int a = (int) (l_idx - m_idx - o*numactions*numalphavecs) / numalphavecs;
        int i = (int) (l_idx - m_idx - o*numactions*numalphavecs) - (a * numalphavecs);

        // Compute CUDA dims for internal kernel calls
//        int num_threads = 256;
//        int num_blocks = (int) (numstates + num_threads - 1) / num_threads;

        // Access beliefvec_set to get b_j
        int B_start_idx = j*numstates;
//        int B_end_idx = B_start_idx + numstates - 1;
//        assign2to1_range<<<num_blocks, num_threads>>>(b_j, beliefvec_set, numstates, B_start_idx, B_end_idx);
//        cudaDeviceSynchronize();

        // Access G_AO to get g_ao^i
        int G_AO_start_idx = i*numactions*numobservs*numstates + o*numactions*numstates + a*numstates;
//        int G_AO_end_idx = G_AO_start_idx + numstates - 1;
//        assign2to1_range<<<num_blocks, num_threads>>>(g_ao_i, alphavec_cache, numstates, G_AO_start_idx, G_AO_end_idx);
//        cudaDeviceSynchronize();

        double dot = 0;

        // Compute and store dot product
//        dot_product<<<num_blocks, num_threads>>>(b_j, g_ao_i, partial_dot, numstates);
//       cudaDeviceSynchronize();

//        dot_product_offset<<<num_blocks, num_threads>>>(beliefvec_set, alphavec_cache, partial_dot, numstates, B_start_idx, G_AO_start_idx);
//        cudaDeviceSynchronize();
//        for (int r=0; r<num_blocks; r++) {
//            dot += partial_dot[r];
//       }

       for (int r=0; r<numstates; r++) {
           dot += beliefvec_set[B_start_idx + r] * alphavec_cache[G_AO_start_idx + r];
       }

        hat_G_AO[l_idx] = dot;

        l_idx += blockDim.x * gridDim.x;
    }



}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    // Initialize G_AO matrix, belief set
    int alphavec_cache_size = numalphavecs * numobservs * numactions * numstates;
    double alphavec_cache[numalphavecs][numobservs][numactions][numstates] = {
            { // i = 0
                { // o = 0
                    0.1, 0.4,
                    0.4, 0.4,
                },
                { // o = 1
                    0.1, 0.4,
                    0.4, 0.4,
                }
            },
            { // i = 1
                { // o = 0
                    0, 0.6,
                    0.6, 0.6
                },
                { // o = 1
                    0, 0.6,
                    0.6, 0.6
                }
            }
    };

    double beliefvec_set[numbeliefs][numstates] = {
            0.7, 0.3,
            0.4, 0.6,
            };

    // Copy G_AO, beliefvec_set
    double *dev_alphavec_cache, *dev_beliefvec_set;

    int space_alphavec_cache = numalphavecs * numobservs * numactions * numstates * sizeof(double);
    int space_beliefvec_set = numbeliefs * numstates * sizeof(double);

    HANDLE_ERROR(cudaMalloc((void **) &dev_alphavec_cache, space_alphavec_cache));
    HANDLE_ERROR(cudaMemcpy(dev_alphavec_cache, alphavec_cache, space_alphavec_cache, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &dev_beliefvec_set, space_beliefvec_set));
    HANDLE_ERROR(cudaMemcpy(dev_beliefvec_set, beliefvec_set, space_beliefvec_set, cudaMemcpyHostToDevice));

    // Initialize hat_G_AO
    double hat_G_AO[numbeliefs][numobservs][numactions][numalphavecs];
    int hat_G_AO_size = numbeliefs * numobservs * numactions * numalphavecs;
    int hat_G_AO_space = hat_G_AO_size * sizeof(double);

    double *dev_hat_G_AO;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_G_AO, hat_G_AO_space));

    // Num blocks
    int num_blocks = (int) (hat_G_AO_size + num_threads - 1) / num_threads;

    // Initialize b_j, g_ao_i, partial_dot
    double *dev_b_j, *dev_g_ao_i, *dev_partial_dot;
    HANDLE_ERROR(cudaMalloc((void **) &dev_b_j, numstates * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_g_ao_i, numstates * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_partial_dot, num_blocks * sizeof(double)));

    // call kernel
    compute_hat_G_AO<<<num_blocks, num_threads>>>(dev_hat_G_AO, hat_G_AO_size, dev_alphavec_cache, dev_beliefvec_set,
                                                  dev_b_j, dev_g_ao_i, dev_partial_dot);

    HANDLE_ERROR( cudaMemcpy( hat_G_AO, dev_hat_G_AO, hat_G_AO_space, cudaMemcpyDeviceToHost ) );


    // Print
    cout << "Size G_AO: " << alphavec_cache_size << endl;
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


    cout << "Size hat_G_AO: " << hat_G_AO_size << endl;
    cout << "Num blocks: " << num_blocks << endl;
    cout << endl;

    cout << "hat_G_AO" << endl;
    cout << endl;

    for(int j=0; j<numbeliefs; j++){
        cout << "Belief j = " << j << endl;
        cout << endl;
        for(int o=0; o<numobservs; o++){
            cout << "Layer o = " << o << endl;
            for(int a=0; a<numactions; a++){
                for(int i=0; i<numalphavecs; i++){
                    cout << hat_G_AO[j][o][a][i] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }


    return 0;
}
