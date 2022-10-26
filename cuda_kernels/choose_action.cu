#include <iostream>
#include "cuda_u.h"

using namespace std;

void compute_action(int trans_matrix[numactions][numstates], int inv_trans_matrix[numactions][numstates],
                    double observ_matrix[numobservs][numactions][numstates], double belief[numstates],
                    double R[numactions][numstates], double gamma, double old_alphavec_set[numalphavecs][numstates]){

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

    // Inverse transition matrix
    int *dev_inv_trans_matrix;
    int space_inv_trans_matrix = numactions * numstates * sizeof(int);
    HANDLE_ERROR(cudaMalloc((void **) &dev_inv_trans_matrix, space_inv_trans_matrix));
    HANDLE_ERROR(cudaMemcpy(dev_inv_trans_matrix, inv_trans_matrix, space_inv_trans_matrix, cudaMemcpyHostToDevice));

    // Belief
    double *dev_belief;
    int space_belief = numstates * sizeof(double);
    HANDLE_ERROR(cudaMalloc((void **) &dev_belief, space_belief));
    HANDLE_ERROR(cudaMemcpy(dev_belief, belief, space_belief, cudaMemcpyHostToDevice));

    // R
    int R_size = numactions * numstates;
    int R_space = R_size * sizeof(double);
    double *dev_R;
    HANDLE_ERROR(cudaMalloc((void **) &dev_R, R_space));
    HANDLE_ERROR(cudaMemcpy(dev_R, R, R_space, cudaMemcpyHostToDevice));

    // --------------------------------------------------------------------------------------------------

    double* r_a;
    double* alpha_i;
    int space_r_a;
    int num_blocks = (int) (numstates + num_threads - 1) / num_threads;
    double b_ao[numstates], v_values[numstates], partial_v_i[num_blocks], v_i, v_star, t_ao[numstates], final_sum;
    double partial_c1[num_blocks], partial_c2[num_blocks];
    double c1, c2;
    int partial_dot_space = num_blocks*sizeof(double);
    int alpha_i_space = numstates * sizeof(double);
    int space_t_ao = numstates * sizeof(double);

    double a_max = -999;
    int a_star = -1;

    for (int a=0; a<numactions; a++){

        // --------------------------------------------------------------------------------------------------
        // Compute c1

        // Select row of R
        r_a = R[a];

        space_r_a = numstates * sizeof(double);
        double *dev_r_a;
        HANDLE_ERROR(cudaMalloc((void **) &dev_r_a, space_r_a));
        HANDLE_ERROR(cudaMemcpy(dev_r_a, r_a, space_r_a, cudaMemcpyHostToDevice));

        // Init partial dot product
        double *dev_partial_c1;
        HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c1, partial_dot_space) );

        dot_product<<<num_blocks, num_threads>>>(dev_r_a, dev_belief, dev_partial_c1, numstates);

        // copy the array 'c' back from the GPU to the CPU
        HANDLE_ERROR( cudaMemcpy( partial_c1, dev_partial_c1, partial_dot_space, cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaFree( dev_r_a ) );
        HANDLE_ERROR( cudaFree( dev_partial_c1 ) );

        // finish up on the CPU side
        c1 = 0;
        for (int i=0; i<num_blocks; i++) {
            c1 += partial_c1[i];
        }

        // --------------------------------------------------------------------------------------------------
        // Compute c2

        c2 = 0;
        for (int o=0; o<numobservs; o++){

            // Compute b_ao
            double *dev_b_ao;
            HANDLE_ERROR(cudaMalloc((void **) &dev_b_ao, space_belief));

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

            HANDLE_ERROR(cudaMemcpy(dev_b_ao, b_ao, space_belief, cudaMemcpyHostToDevice));

            // --------------------------------------------------------------------------------------------------
            // Compute v_star

            for (int i=0; i<numalphavecs; i++){

                alpha_i = old_alphavec_set[i];
                double *dev_alpha_i;
                HANDLE_ERROR( cudaMalloc( (void**)&dev_alpha_i, alpha_i_space) );
                HANDLE_ERROR(cudaMemcpy(dev_alpha_i, alpha_i, alpha_i_space, cudaMemcpyHostToDevice));

                // Init partial dot product
                double *dev_partial_v_i;
                HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_v_i, partial_dot_space) );

                dot_product<<<num_blocks, num_threads>>>(dev_alpha_i, dev_b_ao, dev_partial_v_i, numstates);

                HANDLE_ERROR( cudaMemcpy( partial_v_i, dev_partial_v_i, partial_dot_space, cudaMemcpyDeviceToHost ) );

                // finish up on the CPU side
                v_i = 0;
                for (int r=0; r<num_blocks; r++) {
                    v_i += partial_v_i[r];
                }

                v_values[i] = v_i;


                HANDLE_ERROR( cudaFree( dev_alpha_i ) );
                HANDLE_ERROR( cudaFree( dev_partial_v_i ) );

            }

            v_star = -999;
            for (int r=0; r<numstates; r++){
                if (v_values[r] > v_star){
                    v_star = v_values[r];
                }
            }

            // --------------------------------------------------------------------------------------------------
            // Compute p(o|b,a)

            // Initialize t_ao
            double *dev_t_ao;
            HANDLE_ERROR(cudaMalloc((void **) &dev_t_ao, space_t_ao));

            compute_t_ao<<<num_blocks, num_threads>>>(dev_t_ao, dev_trans_matrix, dev_observ_matrix, a, o);

            HANDLE_ERROR( cudaMemcpy( t_ao, dev_t_ao, space_t_ao, cudaMemcpyDeviceToHost ) );

            double *dev_partial_c2;
            HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c2, partial_dot_space) );

            dot_product<<<num_blocks, num_threads>>>(dev_belief, dev_t_ao, dev_partial_c2, numstates);

            // copy the array 'c' back from the GPU to the CPU
            HANDLE_ERROR( cudaMemcpy( partial_c2, dev_partial_c2, partial_dot_space, cudaMemcpyDeviceToHost ) );

            HANDLE_ERROR( cudaFree( dev_partial_c2 ) );

            // finish up on the CPU side
            c2 = 0;
            for (int i=0; i<num_blocks; i++) {
                c2 += partial_c2[i];
            }

            cout << "a = " << a << ", o = " << o << endl;

            cout << "belief" << endl;
            for (int s=0; s<numstates; s++){
                cout << belief[s] << " ";
            }
            cout << endl;

            cout << "t_ao" << endl;
            cout << endl;

            for (int s=0; s<numstates; s++){
                cout << t_ao[s] << " ";
            }
            cout << endl;
            cout << endl;

            cout << "c2 = " << c2 << endl;

            c2 *= v_star;

            // --------------------------------------------------------------------------------------------------
            // Compute final sum

            final_sum = c1 + gamma * c2;
            if (final_sum > a_max){
                a_max = final_sum;
                a_star = a;
            }

            cout << "final sum: " << final_sum << endl;
            cout << "a_max: " << a_max << endl;
            cout << "a_star: " << a_star << endl;

        }


    }

}

int main() {
    std::cout << "Hello, World! This is GPU test" << std::endl;

    int trans_matrix[numactions][numstates] = {
            0, 1,
            1, 1,
            };

    // Initialize inverse transition matrix
    int inv_trans_matrix[numactions][numstates] = {
            0, 1,
            0, 1,
    };

    double observ_matrix[numobservs][numactions][numstates] = {
            { {0.1, 0.2}, {0.1, 0.2} },
            { {0.1, 0.2}, {0.1, 0.2} }
    };

    double belief[numstates] = {
            0.1, 0.9,
    };

    double R[numactions][numstates] = {
            0,1,
            1,0,
            };

    double gamma = 1;

    double old_alphavec_set[numalphavecs][numstates] = {
            1, 2,
            0, 3,
            };

    compute_action(trans_matrix, inv_trans_matrix, observ_matrix, belief, R, gamma, old_alphavec_set);

}
