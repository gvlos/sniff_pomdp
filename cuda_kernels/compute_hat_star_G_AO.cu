#include <iostream>
#include "cuda_utils.h"

using namespace std;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n
const int numbeliefs = 2; // |B|

__global__ void compute_hat_star_G_AO(int *hat_star_G_AO, int hat_star_G_AO_size, double *hat_G_AO) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < hat_star_G_AO_size){

        int j = (int) l_idx / (numactions * numobservs);
        int w_idx = j * numactions * numobservs;
        int a = (int) (l_idx - w_idx) / numobservs;
        int o = (l_idx - w_idx) - (a * numobservs);

        int start_idx = j * numactions * numalphavecs * numobservs + o * numactions * numalphavecs + a * numalphavecs;

        double max = -999;
        int argmax = -1;
        for(int r=0; r<numalphavecs; r++){
            if(hat_G_AO[start_idx + r] > max){
                max = hat_G_AO[start_idx + r];
                argmax = r;
            }
        }

        hat_star_G_AO[l_idx] = argmax;

        l_idx += blockDim.x * gridDim.x;
    }



}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    // Initialize G_AO matrix, belief set
    int hat_G_AO_size = numactions * numalphavecs * numobservs * numbeliefs;
    double hat_G_AO[numbeliefs][numobservs][numactions][numalphavecs] = {
            { // j = 0
                { // o = 0
                    0.19, 0.18,
                    0.4, 0.6,
                },
                { // o = 1
                    0.19, 0.18,
                    0.4, 0.6,
                }
            },
            { // j = 1
                { // o = 0
                    0.28, 0.36,
                    0.4, 0.6
                },
                { // o = 1
                    0.28, 0.36,
                    0.4, 0.6
                }
            }
    };

    // Copy G_AO,
    double *dev_hat_G_AO;

    int hat_G_AO_space = hat_G_AO_size * sizeof(double);
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_G_AO, hat_G_AO_space));
    HANDLE_ERROR(cudaMemcpy(dev_hat_G_AO, hat_G_AO, hat_G_AO_space, cudaMemcpyHostToDevice));

    // Initialize hat_star_G_AO
    int hat_star_G_AO_size = numactions * numobservs * numbeliefs;
    int hat_star_G_AO_space = numactions * numobservs * numbeliefs * sizeof(int);

    int hat_star_G_AO[numbeliefs][numactions][numobservs];
    int *dev_hat_star_G_AO;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_star_G_AO, hat_star_G_AO_space));

    // Num blocks
    int num_blocks = (int) (hat_star_G_AO_size + num_threads - 1) / num_threads;

    // call kernel
    compute_hat_star_G_AO<<<num_blocks, num_threads>>>(dev_hat_star_G_AO, hat_star_G_AO_size, dev_hat_G_AO);

    // copy back
    HANDLE_ERROR( cudaMemcpy( hat_star_G_AO, dev_hat_star_G_AO, hat_star_G_AO_space, cudaMemcpyDeviceToHost ) );


    // Print
    cout << "Size hat_star_G_AO: " << hat_star_G_AO_size << endl;
    cout << endl;

    cout << "hat_star_G_AO" << endl;
    cout << endl;

    for(int j=0; j<numbeliefs; j++){
        cout << "Belief j = " << j << endl;
        cout << endl;
        for(int a=0; a<numactions; a++){
            for(int o=0; o<numobservs; o++){
                cout << hat_star_G_AO[j][a][o] << " ";
            }
            cout << endl;
        }
        cout << endl;
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
