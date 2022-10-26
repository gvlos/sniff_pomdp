#include <iostream>
#include "cuda_utils.h"

using namespace std;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n
const int numbeliefs = 2; // |B|

__global__ void compute_G_AB(double *G_AB, int G_AB_size, int *hat_star_G_AO, double *alphavec_cache, double *R,
                             double gamma) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < G_AB_size){

        int j = (int) l_idx / (numstates * numactions);
        int w_idx = j * numstates * numactions;
        int a = (int) (l_idx - w_idx) / numstates;
        int s = (l_idx - w_idx) - a*numstates;

        double sum = 0;
        int t_k, p_k;
        for (int k=0; k<numobservs; k++){
            t_k = j * numactions * numobservs + a * numobservs + k;
            p_k = hat_star_G_AO[t_k] * numactions * numstates * numobservs + k * numactions * numstates + a * numstates
                    + s;
            sum += alphavec_cache[p_k];
        }

        G_AB[l_idx] = R[a*numstates + s] + gamma * sum;

        l_idx += blockDim.x * gridDim.x;
    }



}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    // G_AO
    int alphavec_cache_size = numalphavecs * numobservs * numactions * numstates;
    int alphavec_cache_space = alphavec_cache_size * sizeof(double);
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

    double *dev_alphavec_cache;
    HANDLE_ERROR(cudaMalloc((void **) &dev_alphavec_cache, alphavec_cache_space));
    HANDLE_ERROR(cudaMemcpy(dev_alphavec_cache, alphavec_cache, alphavec_cache_space, cudaMemcpyHostToDevice));

    // hat_star_G_AO
    int hat_star_G_AO_size = numactions * numobservs * numbeliefs;
    int hat_star_G_AO_space = hat_star_G_AO_size * sizeof(int);

    int hat_star_G_AO[numbeliefs][numactions][numobservs] = {
            { // j=0
                0,0,
                1,1,
            },
            { // j=1
                1,1,
                1,1,
            }
    };

    int *dev_hat_star_G_AO;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_star_G_AO, hat_star_G_AO_space));
    HANDLE_ERROR(cudaMemcpy(dev_hat_star_G_AO, hat_star_G_AO, hat_star_G_AO_space, cudaMemcpyHostToDevice));

    // R
    int R_size = numactions * numstates;
    int R_space = R_size * sizeof(double);
    double R [numactions][numstates] = {
            0,1,
            1,0,
    };

    double *dev_R;
    HANDLE_ERROR(cudaMalloc((void **) &dev_R, R_space));
    HANDLE_ERROR(cudaMemcpy(dev_R, R, R_space, cudaMemcpyHostToDevice));

    double gamma = 1;

    // Initialize G_AB
    int G_AB_size = numactions * numstates * numbeliefs;
    int G_AB_space = G_AB_size * sizeof(double);

    double G_AB[numbeliefs][numactions][numstates];
    double *dev_G_AB;
    HANDLE_ERROR(cudaMalloc((void **) &dev_G_AB, G_AB_space));

    // Num blocks
    int num_blocks = (int) (G_AB_size + num_threads - 1) / num_threads;

    // call kernel
    compute_G_AB<<<num_blocks, num_threads>>>(dev_G_AB, G_AB_size, dev_hat_star_G_AO, dev_alphavec_cache, dev_R, gamma);

    // copy back
    HANDLE_ERROR( cudaMemcpy( G_AB, dev_G_AB, G_AB_space, cudaMemcpyDeviceToHost ) );

    // Print
    cout << "Size G_AB: " << G_AB_size << endl;
    cout << endl;

    cout << "G_AB" << endl;
    cout << endl;

    for(int j=0; j<numbeliefs; j++){
        cout << "Belief j = " << j << endl;
        cout << endl;
        for(int a=0; a<numactions; a++){
            for(int s=0; s<numobservs; s++){
                cout << G_AB[j][a][s] << " ";
            }
            cout << endl;
        }
        cout << endl;
        cout << endl;
    }


    return 0;
}
