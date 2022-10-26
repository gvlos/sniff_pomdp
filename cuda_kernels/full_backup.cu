#include <iostream>
#include "cuda_u.h"
#include <vector>

using namespace std;

vector<vector<float> > cuda_backup(vector<vector<int> > trans_matrix, vector<vector<vector<float> > >observ_matrix,
                                    vector<vector<float> > old_alphavec_set, vector<vector<float> > beliefvec_set,
                                    vector<vector<float> > R, float gamma){

    // Copy transition and observation matrix, alpha vecs
    int *dev_trans_matrix;
    float *dev_observ_matrix, *dev_old_alphavec_set;

    int space_trans_matrix = numactions * numstates * sizeof(int);
    int space_observ_matrix = numactions * numstates * numobservs * sizeof(float);
    int space_alphavecs_set = numstates * numalphavecs * sizeof(float);

    HANDLE_ERROR(cudaMalloc((void **) &dev_trans_matrix, space_trans_matrix));
    int tm[numactions][numstates];
    for(int a=0; a<numactions; a++){
        for(int s=0; s<numstates; s++){
            tm[a][s] = trans_matrix[a][s];
        }
    }
    HANDLE_ERROR(cudaMemcpy(dev_trans_matrix, tm, space_trans_matrix, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &dev_observ_matrix, space_observ_matrix));
    float obm[numobservs][numactions][numstates];
    for(int o=0; o<numobservs; o++){
        for(int a=0; a<numactions; a++){
            for(int s=0; s<numstates; s++){
                obm[o][a][s] = observ_matrix[o][a][s];
            }
        }
    }
    HANDLE_ERROR(cudaMemcpy(dev_observ_matrix, obm, space_observ_matrix, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &dev_old_alphavec_set, space_alphavecs_set));
    float ovs[numalphavecs][numstates];
    for(int i=0; i<numalphavecs; i++){
        for(int s=0; s<numstates; s++){
            ovs[i][s] = old_alphavec_set[i][s];
        }
    }
    HANDLE_ERROR(cudaMemcpy(dev_old_alphavec_set, ovs, space_alphavecs_set, cudaMemcpyHostToDevice));

    // --------------------------------------------------------------------------------------------------
    // Compute G_AO
    float alphavec_cache[numalphavecs][numobservs][numactions][numstates];
    int alphavec_cache_size = numactions * numobservs * numstates * numalphavecs;
    int alphavec_cache_space = alphavec_cache_size * sizeof(float);

    float *dev_alphavec_cache;
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
    float *dev_beliefvec_set;
    int space_beliefvec_set = numbeliefs * numstates * sizeof(float);
    HANDLE_ERROR(cudaMalloc((void **) &dev_beliefvec_set, space_beliefvec_set));
    float bvs[numbeliefs][numstates];
    for(int j=0; j<numbeliefs; j++){
        for(int s=0; s<numstates; s++){
            bvs[j][s] = beliefvec_set[j][s];
        }
    }
    HANDLE_ERROR(cudaMemcpy(dev_beliefvec_set, bvs, space_beliefvec_set, cudaMemcpyHostToDevice));

    float hat_G_AO[numbeliefs][numobservs][numactions][numalphavecs];
    int hat_G_AO_size = numbeliefs * numobservs * numactions * numalphavecs;
    int hat_G_AO_space = hat_G_AO_size * sizeof(float);

    float *dev_hat_G_AO;
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
    int R_space = R_size * sizeof(float);
    float *dev_R;
    HANDLE_ERROR(cudaMalloc((void **) &dev_R, R_space));
    float R_mat[numactions][numstates];
    for(int a=0; a<numactions; a++){
        for(int s=0; s<numstates; s++){
            R_mat[a][s] = R[a][s];
        }
    }
    HANDLE_ERROR(cudaMemcpy(dev_R, R_mat, R_space, cudaMemcpyHostToDevice));

    // Compute G_AB
    int G_AB_size = numactions * numstates * numbeliefs;
    int G_AB_space = G_AB_size * sizeof(float);

    float G_AB[numbeliefs][numactions][numstates];
    float *dev_G_AB;
    HANDLE_ERROR(cudaMalloc((void **) &dev_G_AB, G_AB_space));

    // Num blocks
    num_blocks = (int) (G_AB_size + num_threads - 1) / num_threads;

    // call kernel
    compute_G_AB<<<num_blocks, num_threads>>>(dev_G_AB, G_AB_size, dev_hat_star_G_AO, dev_alphavec_cache, dev_R, gamma);

    // copy back
    HANDLE_ERROR( cudaMemcpy( G_AB, dev_G_AB, G_AB_space, cudaMemcpyDeviceToHost ) );

    // --------------------------------------------------------------------------------------------------
    // Compute hat_G_AB
    float hat_G_AB[numbeliefs][numactions];
    int hat_G_AB_size = numbeliefs * numactions;
    int hat_G_AB_space = hat_G_AB_size * sizeof(float);

    float *dev_hat_G_AB;
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
        float max = -999;
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
    float B_KP[numbeliefs][numstates];
    int B_KP_size = numbeliefs * numstates;
    int B_KP_space = B_KP_size * sizeof(float);
    float *dev_B_KP;
    HANDLE_ERROR(cudaMalloc((void **) &dev_B_KP, B_KP_space));

    // Num blocks
    num_blocks = (int) (B_KP_size + num_threads - 1) / num_threads;

    compute_backup<<<num_blocks, num_threads>>>(dev_B_KP, B_KP_size, dev_G_AB, dev_hat_star_G_AB);

    HANDLE_ERROR( cudaMemcpy( B_KP, dev_B_KP, B_KP_space, cudaMemcpyDeviceToHost ) );

    vector<vector<float> > backup_matrix;
    backup_matrix.resize(numbeliefs);
    for(int j=0; j<numbeliefs; j++){
        backup_matrix[j].resize(numstates);
        for(int s=0; s<numstates; s++){
            backup_matrix[j][s] = B_KP[j][s];
        }
    }


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

    return backup_matrix;

}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    // Initialize transition matrix, observation matrix, alpha vecs
    int tm[numactions][numstates] = {
            0, 1,
            1, 1,
    };

    vector<vector<int> > transition_matrix;
    transition_matrix.resize(numactions);
    for(int a=0; a<numactions; a++){
        transition_matrix[a].resize(numstates);
        for(int s=0; s<numstates; s++){
            transition_matrix[a][s] = tm[a][s];
        }
    }

    float om[numobservs][numactions][numstates] = {
            { {0.1, 0.2}, {0.1, 0.2} },
            { {0.1, 0.2}, {0.1, 0.2} }
    };

    vector<vector<vector<float > > > observation_matrix;
    observation_matrix.resize(numobservs);
    for(int o=0; o<numobservs; o++){
        observation_matrix[o].resize(numactions);
        for(int a=0; a<numactions; a++){
            observation_matrix[o][a].resize(numstates);
            for(int s=0; s<numstates; s++){
                observation_matrix[o][a][s] = om[o][a][s];
            }
        }
    }

    float oas[numalphavecs][numstates] = {
            1, 2,
            0, 3,
    };

    vector<vector<float> > old_alphavec_set;
    old_alphavec_set.resize(numalphavecs);
    for(int i=0; i<numalphavecs; i++){
        old_alphavec_set[i].resize(numstates);
        for(int s=0; s<numstates; s++){
            old_alphavec_set[i][s] = oas[i][s];
        }
    }

    // B
    float bvs[numbeliefs][numstates] = {
            0.7, 0.3,
            0.4, 0.6,
            };

    vector<vector<float> > beliefvec_set;
    beliefvec_set.resize(numbeliefs);
    for(int j=0; j<numbeliefs; j++){
        beliefvec_set[j].resize(numstates);
        for(int s=0; s<numstates; s++){
            beliefvec_set[j][s] = bvs[j][s];
        }
    }

    float R_mat[numactions][numstates] = {
            0,1,
            1,0,
            };

    vector<vector<float> > R;
    R.resize(numactions);
    for(int a=0; a<numactions; a++){
        R[a].resize(numstates);
        for(int s=0; s<numstates; s++){
            R[a][s] = R_mat[a][s];
        }
    }

    float gamma = 1;

    vector<vector<float> > B_KP = cuda_backup(transition_matrix, observation_matrix, old_alphavec_set, beliefvec_set, R, gamma);

//    cout << "Size B_KP: " <<  << endl;
//    cout << endl;

    cout << "B_KP" << endl;
    cout << endl;

    for(int j=0; j<numbeliefs; j++){
        for(int s=0; s<numstates; s++){
            cout << B_KP[j][s] << " ";
        }
        cout << endl;
    }

//    cout << "Trans matrix" << endl;
//    cout << endl;
//
//    for(int a=0; a<numactions; a++){
//        for(int s=0; s<numstates; s++){
//            cout << transition_matrix[a][s] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//
//    cout << "Observ matrix" << endl;
//    cout << endl;
//
//    for(int o=0; o<numobservs; o++){
//        cout << "Layer o = " << o << endl;
//        for(int a=0; a<numactions; a++){
//            for(int s=0; s<numstates; s++){
//                cout << observation_matrix[o][a][s] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//    cout << endl;
//
//    cout << "Old alpha vec set" << endl;
//    cout << endl;
//
//    for(int i=0; i<numalphavecs; i++){
//        for(int s=0; s<numstates; s++){
//            cout << old_alphavec_set[i][s] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//
//    cout << "Beliefvec set" << endl;
//    cout << endl;
//
//    for(int j=0; j<numbeliefs; j++){
//        for(int s=0; s<numstates; s++){
//            cout << beliefvec_set[j][s] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//
//    cout << "Reward matrix" << endl;
//    cout << endl;
//
//    for(int a=0; a<numactions; a++){
//        for(int s=0; s<numstates; s++){
//            cout << R[a][s] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;

    return 0;
}
