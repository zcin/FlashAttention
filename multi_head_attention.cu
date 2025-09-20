#include <cstdio>
#include <fstream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <iomanip>

#define CEIL_DIV(x,y) (x+y-1)/(y)

__device__ void fill(float* ptr, int N, float val) {
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) {
        ptr[i] = val;
    }
}

// Br=32 threads loading Br x d or Bc x d
__device__ void f_load_matrix_block(float* dst, const float* parent, int rows, int cols,
        int row_offset, int col_offset, int block_rows, int block_cols) {

    int tid = threadIdx.x;
    for (int i = tid; i < block_rows * block_cols; i += blockDim.x) {
        int r = i / block_cols;
        int c = i % block_cols;
        if (row_offset + r < rows) // guaranteed that `block_cols` divides `cols`
            dst[r * block_cols + c] = parent[(row_offset + r)*cols + (col_offset + c)];
        else
            dst[r * block_cols + c] = 0.0f;
    }
}

// Load in transposed tile
// numRows = Bc or <Bc (for the last tile)
__device__ void f_load_matrix_block_transpose(float* dst, const float* parent, int rows, int cols,
        int row_offset, int col_offset, int block_rows, int block_cols) {

    int tid = threadIdx.x;
    for (int i = tid; i < block_rows * block_cols; i += blockDim.x) {
        int r = i / block_cols;
        int c = i % block_cols;
        if (row_offset + r < rows) // guaranteed that `block_cols` divides `cols`
            dst[c * block_rows + r] = parent[(row_offset + r)*cols + (col_offset + c)];
        else
            dst[c * block_rows + r] = 0.0f;
    }
}

__device__ void f_store_matrix_block(float* src, float* parent, int rows, int cols,
        int row_offset, int col_offset, int block_rows, int block_cols) {

    int tid = threadIdx.x;
    for (int i = tid; i < block_rows * block_cols; i += blockDim.x) {
        int r = i / block_cols;
        int c = i % block_cols;
        if (row_offset + r < rows) // guaranteed that `block_cols` divides `cols`
            parent[(row_offset + r)*cols + (col_offset + c)] = src[r * block_cols + c];
    }
}

// A(M,K), B(K,N), C(M,N) are pointers to shared memory
template <bool is_acc = false>
__device__ void matrix_multiply(float* A, float* B, float* C, int M, int K, int N) {
    int tid = threadIdx.x;
    for (int i = tid; i < M*N; i += blockDim.x) {
        int r = i / N;
        int c = i % N;
        float acc = is_acc ? C[i] : 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[r*K + k] * B[k*N + c];
        }
        C[i] = acc;
    }
}

__device__ void scale_matrix(float* A, float scalar, int N) {
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) {
        A[i] *= scalar;
    }
}

// Sij (Br,Bc), m_new (Br), m_old (Br)
__device__ void reduce_max(float* Sij, float* m_new, float* m_old, int Br, int Bc) {
    int tid = threadIdx.x;
    for (int i = tid; i < Br; i += blockDim.x) {
        float maxval = m_old[i];
        for (int j = 0; j < Bc; j++) {
            maxval = max(maxval, Sij[i*Bc + j]);
        }
        m_new[i] = maxval;
    }
}

// Sij (Br, Bc), mi (Br)
__device__ void softmax(float* Sij, float* mi, int Br, int Bc) {
    int tid = threadIdx.x;
    for (int i = tid; i < Br*Bc; i += blockDim.x) {
        int r = i / Bc;
        Sij[i] = __expf(Sij[i] - mi[r]);
    }
}

// Sij (Br,Bc)
__device__ void scale_exp_m(float* Pij, float* li, float* m_new, float* m_old, int Br, int Bc) {
    int tid = threadIdx.x;
    for (int r = tid; r < Br; r += blockDim.x) {
        float acc = li[r] * __expf(m_old[r] - m_new[r]);
        for (int c = 0; c < Bc; c++) {
            acc += Pij[r * Bc + c];
        }
        li[r] = acc;
    }
}

// A(R,C), m_new(R), m_old(R)
__device__ void broadcasted_scale_exp_m(float* A, float* m_new, float* m_old, int R, int C) {
    int tid = threadIdx.x;
    for (int i = tid; i < R*C; i += blockDim.x) {
        int r = i / C;
        A[i] *= __expf(m_old[r] - m_new[r]);
    }
}

// A(R,C), vector(R)
__device__ void broadcasted_scale_div(float* A, float* vector, int R, int C) {
    int tid = threadIdx.x;
    for (int i = tid; i < R*C; i += blockDim.x) {
        int r = i / C;
        A[i] /= vector[r];
    }
}

__global__ void attention_kernel(const float* Q, const float* K, const float* V, float* output,
        int N, int d_model, int dk, int h, int Br, int Bc, int Tr, int Tc) {
    int i = blockIdx.x / h;
    int head = blockIdx.x % h;
    int realBr = min(Br, N - i*Br);

    // Shared memory to store Qi, Kj, Sij, Oi
    extern __shared__ float sram[];
    // Allocate
    float* Qi = sram;
    float* Kj = &Qi[Br * dk]; // size(Qi) offset
    float* Sij = &Kj[Bc * dk]; // size(Kj) offset
    float* Oi = &Sij[Br * Bc]; // size(Sij) offset
    float* m_new = &Oi[Br * dk];
    float* m_old = &m_new[Br];
    float* li = &m_old[Br];

    // Load + Initialize
    f_load_matrix_block(Qi, Q, /*rows*/N, /*cols*/d_model, /*row_offset*/i*Br, /*col_offset*/head*dk, /*block_rows*/realBr, /*block_cols*/dk);
    fill(Oi, Br*dk, 0.0f);
    fill(m_old, Br, -FLT_MAX);
    fill(li, Br, 0.0f);
    __syncthreads();

    for (int j = 0; j < Tc; j++) {
        int realBc = min(Bc, N - j*Bc);

        // load Kj.T
        f_load_matrix_block_transpose(Kj, K, /*rows*/N, /*cols*/d_model, /*row_offset*/j*Bc, /*col_offset*/head*dk, /*block_rows*/realBc, /*block_cols*/dk);
        __syncthreads();

        // calculate Sij = QiKj.T / sqrt(d)
        matrix_multiply(Qi, Kj, Sij, realBr, dk, realBc);
        __syncthreads();
        scale_matrix(Sij, 1.0f / sqrtf(dk), realBr * realBc);
        __syncthreads();

        // calculate m_new = max(m_old, max(Sij)) across rows
        reduce_max(Sij, m_new, m_old, realBr, realBc);
        __syncthreads();

        // calculate Pij = exp(Sij-m_new)
        softmax(Sij, m_new, realBr, realBc);
        __syncthreads();

        // calculate li = exp(m_old - m_new)li + Pij.sum()
        scale_exp_m(Sij, li, m_new, m_old, realBr, realBc);
        __syncthreads();

        // Note: Kj is now actually Vj, b/c we are loading tile of V into Kj
        f_load_matrix_block(Kj, V, /*rows*/N, /*cols*/d_model, /*row_offset*/j*Bc, /*col_offset*/head*dk, /*block_rows*/realBc, /*block_cols*/dk);
        __syncthreads();

        // calculate Oi = exp(m_old - m_new)Oi + PijVj
        // first: Oi *= exp(m_old - m_new)
        broadcasted_scale_exp_m(Oi, m_new, m_old, realBr, dk);
        __syncthreads();
        // second: Oi += PijVj
        matrix_multiply<true>(Sij, Kj, Oi, realBr, realBc, dk);
        __syncthreads();

        // Instead of copying m_new into m_old, just swap the pointers
        float* tmp = m_old;
        m_old = m_new;
        m_new = tmp;
    }

    // Oi /= li
    broadcasted_scale_div(Oi, li, realBr, dk);
    __syncthreads();

    // store Oi to output
    f_store_matrix_block(Oi, output, /*rows*/N, /*cols*/d_model, /*row_offset*/i*Br, /*col_offset*/head*dk, /*block_rows*/realBr, /*block_cols*/dk);

    __syncthreads();
}

// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("compute attention with N=%d, d_model=%d, num heads=%d\n", N, d_model, h);
    cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    int dk = d_model / h;
    int Bc = CEIL_DIV(25000, 4 * dk);
    int Br = min(Bc, dk);
    int Tr = CEIL_DIV(N, Br);
    int Tc = CEIL_DIV(N, Bc);
    printf("Br=%d, Bc=%d, Tr=%d, Tc=%d\n", Br, Bc, Tr, Tc);
    int sharedSize = (Br*Bc + 3*Bc*dk + 3*Br) * sizeof(float);
    attention_kernel<<<Tr*h, 512, sharedSize>>>(Q, K, V, output, N, d_model, dk, h, Br, Bc, Tr, Tc);
}

void readFile(const std::string& filename, float* ptr, int size) {
    std::ifstream file(filename);
    for (int i = 0; i < size; ++i)
        file >> ptr[i];
    file.close();
}

void writeToFile(const std::string& filename, float* data, int size) {
    std::ofstream file(filename);

    file << std::fixed << std::setprecision(12);
    for (int i = 0; i < size; ++i)
        file << data[i] << "\n";

    file.close();
}

int main(int argc, char* argv[]) {
    int N = std::atoi(argv[1]);
    int d = std::atoi(argv[2]);
    int h = std::atoi(argv[3]);

    float *Q, *K, *V, *output, *d_Q, *d_K, *d_V, *d_output;
    Q = new float[N*d];
    K = new float[N*d];
    V = new float[N*d];
    output = new float[N*d];
    readFile("Q.txt", Q, N*d);
    readFile("K.txt", K, N*d);
    readFile("V.txt", V, N*d);
    cudaMalloc(&d_Q, N*d * sizeof(float));
    cudaMalloc(&d_K, N*d * sizeof(float));
    cudaMalloc(&d_V, N*d * sizeof(float));
    cudaMalloc(&d_output, N*d * sizeof(float));
    cudaMemcpy(d_Q, Q, N*d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N*d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N*d * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_Q, d_K, d_V, d_output, N, d, h);
    cudaMemcpy(output, d_output, N*d * sizeof(float), cudaMemcpyDeviceToHost);
    writeToFile("output.txt", output, N*d);

    // Free up memory
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] output;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
}
