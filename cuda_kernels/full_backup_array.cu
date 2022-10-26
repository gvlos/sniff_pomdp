#include <iostream>
#include "cuda_u.h"
#include <vector>

using namespace std;

void cuda_backup(int trans_matrix[numactions][numstates], double observ_matrix[numobservs][numactions][numstates],
                 double old_alphavec_set[numalphavecs][numstates], double beliefvec_set[numbeliefs][numstates],
                 double R[numactions][numstates], double gamma){

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

    // --------------------------------------------------------------------------------------------------
    // Compute G_AO
    double alphavec_cache[numalphavecs][numobservs][numactions][numstates];
    int alphavec_cache_size = numactions * numobservs * numstates * numalphavecs;
    int alphavec_cache_space = alphavec_cache_size * sizeof(double);

    double *dev_alphavec_cache;
    HANDLE_ERROR(cudaMalloc((void **) &dev_alphavec_cache, alphavec_cache_space));

    // Num blocks
    int num_blocks = (int) (alphavec_cache_size + num_threads - 1) / num_threads;

    // call kernel
    compute_G_AO<<<num_blocks, num_threads>>>(dev_alphavec_cache, alphavec_cache_size, dev_old_alphavec_set,
                                              dev_observ_matrix, dev_trans_matrix);

    // copy back
    HANDLE_ERROR( cudaMemcpy( alphavec_cache, dev_alphavec_cache,
                              alphavec_cache_space, cudaMemcpyDeviceToHost ) );

    // --------------------------------------------------------------------------------------------------
    // Compute hat_G_AO
    double *dev_beliefvec_set;
    int space_beliefvec_set = numbeliefs * numstates * sizeof(double);
    HANDLE_ERROR(cudaMalloc((void **) &dev_beliefvec_set, space_beliefvec_set));
    HANDLE_ERROR(cudaMemcpy(dev_beliefvec_set, beliefvec_set, space_beliefvec_set, cudaMemcpyHostToDevice));

    double hat_G_AO[numbeliefs][numobservs][numactions][numalphavecs];
    int hat_G_AO_size = numbeliefs * numobservs * numactions * numalphavecs;
    int hat_G_AO_space = hat_G_AO_size * sizeof(double);

    double *dev_hat_G_AO;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_G_AO, hat_G_AO_space));

    // Num blocks
    num_blocks = (int) (hat_G_AO_size + num_threads - 1) / num_threads;

    // call kernel
    compute_hat_G_AO<<<num_blocks, num_threads>>>(dev_hat_G_AO, hat_G_AO_size, dev_alphavec_cache, dev_beliefvec_set);

    HANDLE_ERROR( cudaMemcpy( hat_G_AO, dev_hat_G_AO, hat_G_AO_space, cudaMemcpyDeviceToHost ) );

    // --------------------------------------------------------------------------------------------------
    // Compute hat_star_G_AO
    int hat_star_G_AO[numbeliefs][numactions][numobservs];
    int hat_star_G_AO_size = numactions * numobservs * numbeliefs;
    int hat_star_G_AO_space = numactions * numobservs * numbeliefs * sizeof(int);
    int *dev_hat_star_G_AO;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_star_G_AO, hat_star_G_AO_space));

    // Num blocks
    num_blocks = (int) (hat_star_G_AO_size + num_threads - 1) / num_threads;

    // call kernel
    compute_hat_star_G_AO<<<num_blocks, num_threads>>>(dev_hat_star_G_AO, hat_star_G_AO_size, dev_hat_G_AO);

    // copy back
    HANDLE_ERROR( cudaMemcpy( hat_star_G_AO, dev_hat_star_G_AO, hat_star_G_AO_space, cudaMemcpyDeviceToHost ) );

    // --------------------------------------------------------------------------------------------------
    // R
    int R_size = numactions * numstates;
    int R_space = R_size * sizeof(double);
    double *dev_R;
    HANDLE_ERROR(cudaMalloc((void **) &dev_R, R_space));
    HANDLE_ERROR(cudaMemcpy(dev_R, R, R_space, cudaMemcpyHostToDevice));

    // Compute G_AB
    int G_AB_size = numactions * numstates * numbeliefs;
    int G_AB_space = G_AB_size * sizeof(double);

    double G_AB[numbeliefs][numactions][numstates];
    double *dev_G_AB;
    HANDLE_ERROR(cudaMalloc((void **) &dev_G_AB, G_AB_space));

    // Num blocks
    num_blocks = (int) (G_AB_size + num_threads - 1) / num_threads;

    // call kernel
    compute_G_AB<<<num_blocks, num_threads>>>(dev_G_AB, G_AB_size, dev_hat_star_G_AO, dev_alphavec_cache, dev_R, gamma);

    // copy back
    HANDLE_ERROR( cudaMemcpy( G_AB, dev_G_AB, G_AB_space, cudaMemcpyDeviceToHost ) );

    // --------------------------------------------------------------------------------------------------
    // Compute hat_G_AB
    double hat_G_AB[numbeliefs][numactions];
    int hat_G_AB_size = numbeliefs * numactions;
    int hat_G_AB_space = hat_G_AB_size * sizeof(double);

    double *dev_hat_G_AB;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_G_AB, hat_G_AB_space));

    // Num blocks
    num_blocks = (int) (hat_G_AB_size + num_threads - 1) / num_threads;

    // call kernel
    compute_hat_G_AB<<<num_blocks, num_threads>>>(dev_hat_G_AB, hat_G_AB_size, dev_beliefvec_set, dev_G_AB);

    // copy back
    HANDLE_ERROR( cudaMemcpy( hat_G_AB, dev_hat_G_AB, hat_G_AB_space, cudaMemcpyDeviceToHost ) );

    // --------------------------------------------------------------------------------------------------
    // Compute hat_star_G_AB
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

    int hat_star_G_AB_size = numbeliefs;
    int hat_star_G_AB_space = hat_star_G_AB_size * sizeof(int);

    int *dev_hat_star_G_AB;
    HANDLE_ERROR(cudaMalloc((void **) &dev_hat_star_G_AB, hat_star_G_AB_space));
    HANDLE_ERROR(cudaMemcpy(dev_hat_star_G_AB, hat_star_G_AB, hat_star_G_AB_space, cudaMemcpyHostToDevice));

    // --------------------------------------------------------------------------------------------------
    // Compute backup
    double B_KP[numbeliefs][numstates];
    int B_KP_size = numbeliefs * numstates;
    int B_KP_space = B_KP_size * sizeof(double);
    double *dev_B_KP;
    HANDLE_ERROR(cudaMalloc((void **) &dev_B_KP, B_KP_space));

    // Num blocks
    num_blocks = (int) (B_KP_size + num_threads - 1) / num_threads;

    compute_backup<<<num_blocks, num_threads>>>(dev_B_KP, B_KP_size, dev_G_AB, dev_hat_star_G_AB);

    HANDLE_ERROR( cudaMemcpy( B_KP, dev_B_KP, B_KP_space, cudaMemcpyDeviceToHost ) );


    // --------------------------------------------------------------------------------------------------
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
    cout << endl;

    cout << "Size hat_star_G_AB: " << numbeliefs << endl;
    cout << endl;

    cout << "hat_G_AB" << endl;
    cout << endl;

    for(int j=0; j<numbeliefs; j++){
        cout << hat_star_G_AB[j] << " ";
    }
    cout << endl;

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

    // B
    double beliefvec_set[numbeliefs][numstates] = {
            0.7, 0.3,
            0.4, 0.6,
            };

    double R[numactions][numstates] = {
            0,1,
            1,0,
            };

    double gamma = 1;

    cuda_backup(trans_matrix, observ_matrix, old_alphavec_set, beliefvec_set, R, gamma);

    return 0;
}
