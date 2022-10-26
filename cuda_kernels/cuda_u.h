
// Test matrix - vector assignment and then dot product with that assignment (kernels assign2to1range and dot_product)

#ifndef SNIFF_GPU_TEST_CUDA_UTILS_H
#define SNIFF_GPU_TEST_CUDA_UTILS_H

#include <stdio.h>

#define imin(a,b) (a<b?a:b)

const int num_threads = 256;

const int numactions = 2; // |A|
const int numobservs = 2; // |\Omega|
const int numstates = 2; // |S|
const int numalphavecs = 2; // n
const int numbeliefs = 2; // |B|

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void compute_t_ao(float *t_ao, int *trans_matrix, float *observ_matrix, int a, int o) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < numstates){

        int s_prime = trans_matrix[a*numstates + l_idx];
        t_ao[l_idx] = observ_matrix[o*numactions*numstates + a*numstates + s_prime];

        l_idx += blockDim.x * gridDim.x;
    }
}

__global__ void compute_hat_b_ao(float *b_ao, float *belief, float *observ_matrix, int *inv_trans_matrix, int a, int o) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < numstates){

        int s = inv_trans_matrix[a*numstates + l_idx];
        b_ao[l_idx] = observ_matrix[o*numactions*numstates + a*numstates + l_idx] * belief[s];

        l_idx += blockDim.x * gridDim.x;
    }
}

__global__ void compute_G_AO(float *alphavec_cache, int alphavec_cache_size, float *old_alphavec_set,
                             float *observ_matrix, int *trans_matrix) {

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
        float observ_prob = observ_matrix[o_idx];

        // Access alpha vector set to get alpha[i][s']
        int alpha_idx = i*numstates + s_prime;
        float alpha_i_sprime = old_alphavec_set[alpha_idx];

        alphavec_cache[l_idx] = observ_prob * alpha_i_sprime;

        l_idx += blockDim.x * gridDim.x;
    }
}

__global__ void compute_hat_G_AO(float *hat_G_AO, int hat_G_AO_size, float *alphavec_cache, float *beliefvec_set) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < hat_G_AO_size){

        int j = (int) l_idx / (numactions * numalphavecs * numobservs);
        int m_idx = j * numactions * numalphavecs * numobservs;
        int o = (int) (l_idx - m_idx) / (numactions * numalphavecs);
        int a = (int) (l_idx - m_idx - o*numactions*numalphavecs) / numalphavecs;
        int i = (int) (l_idx - m_idx - o*numactions*numalphavecs) - (a * numalphavecs);

        // Access beliefvec_set to get b_j
        int B_start_idx = j*numstates;

        // Access G_AO to get g_ao^i
        int G_AO_start_idx = i*numactions*numobservs*numstates + o*numactions*numstates + a*numstates;

        // Dot product
        float dot = 0;
        for (int r=0; r<numstates; r++) {
            dot += beliefvec_set[B_start_idx + r] * alphavec_cache[G_AO_start_idx + r];
        }

        hat_G_AO[l_idx] = dot;

        l_idx += blockDim.x * gridDim.x;
    }
}

__global__ void compute_hat_star_G_AO(int *hat_star_G_AO, int hat_star_G_AO_size, float *hat_G_AO) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < hat_star_G_AO_size){

        int j = (int) l_idx / (numactions * numobservs);
        int w_idx = j * numactions * numobservs;
        int a = (int) (l_idx - w_idx) / numobservs;
        int o = (l_idx - w_idx) - (a * numobservs);

        int start_idx = j * numactions * numalphavecs * numobservs + o * numactions * numalphavecs + a * numalphavecs;

        float max = -999;
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

__global__ void compute_G_AB(float *G_AB, int G_AB_size, int *hat_star_G_AO, float *alphavec_cache, float *R,
                             float gamma) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < G_AB_size){

        int j = (int) l_idx / (numstates * numactions);
        int w_idx = j * numstates * numactions;
        int a = (int) (l_idx - w_idx) / numstates;
        int s = (l_idx - w_idx) - a*numstates;

        float sum = 0;
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

__global__ void compute_hat_G_AB(float *hat_G_AB, int hat_G_AB_size, float *beliefvec_set, float *G_AB) {

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < hat_G_AB_size){

        int j = (int) l_idx / numactions;
        int a = l_idx - j*numactions;

        float sum = 0;
        float g_a_bj, bj;
        for (int s=0; s<numstates; s++){
            g_a_bj = G_AB[j*numactions*numstates + a*numstates + s];
            bj = beliefvec_set[j*numstates + s];
            sum += bj * g_a_bj;
        }

        hat_G_AB[l_idx] = sum;

        l_idx += blockDim.x * gridDim.x;
    }
}

__global__ void compute_backup(float *B_KP, int B_KP_size, float *G_AB, int *hat_star_G_AB){

    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (l_idx < B_KP_size){

        int j = (int) l_idx / numstates;
        int a_star_bj = hat_star_G_AB[j];
        int s = l_idx - j*numstates;

        B_KP[l_idx] = G_AB[j*numactions*numstates + a_star_bj*numstates + s];

        l_idx += blockDim.x * gridDim.x;
    }
}

__global__ void assign2to1_range(float *vec1, float *vec2, int vec2_size, int start_idx, int end_idx){
    // Assign vector number 2 to vector number 1
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < vec2_size){
        if((tid >= start_idx) && (tid <= end_idx)){
            vec1[tid - start_idx] = vec2[tid];
        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void assign2to1(float *vec1, float *vec2, int size){
    // Assign vector number 2 to vector number 1
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size){
        vec1[tid] = vec2[tid];
        tid += blockDim.x * gridDim.x;
    }
}


__global__ void range_scalar_product(float scalar, float *vec, int vec_size, float *prod, int idx_start, int idx_end){
    // Scalar product in a specified linear range
    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (l_idx < vec_size){
        if((l_idx >= idx_start) && (l_idx <= idx_end)){
            prod[l_idx] = scalar * vec[l_idx];
        }
        l_idx += blockDim.x * gridDim.x;
    }
}

__global__ void scalar_product(int scalar, int *vec, int vec_size, int *prod){
    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (l_idx < vec_size){
        prod[l_idx] = scalar * vec[l_idx];
        l_idx += blockDim.x * gridDim.x;
    }
}

__global__ void dot_product( float *a, float *b, float *c, int size) {
    __shared__ float cache[num_threads];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < size) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

__global__ void dot_product_offset( float *a, float *b, float *c, int size, int start_idx_a, int start_idx_b){
    __shared__ float cache[num_threads];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < size) {
        temp += a[start_idx_a + tid] * b[start_idx_b + tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

#endif //SNIFF_GPU_TEST_CUDA_UTILS_H
