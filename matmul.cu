#include "matmul.h"

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define TILE_SIZE 16
#define THREAD_M 8
#define THREAD_N 8
#define NGPU 4
#define DOUBLE_BUFFER_DIV 4 

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int j = blockDim.x * blockIdx.x * THREAD_N + threadIdx.x; // col
  int i = blockDim.y * blockIdx.y * THREAD_M + threadIdx.y * THREAD_M; // row
  __shared__ float local_a[TILE_SIZE * THREAD_M][TILE_SIZE];  
  __shared__ float local_b[TILE_SIZE][TILE_SIZE * THREAD_N];
  if (i >= M || j >= N) return;
  float sum[THREAD_M][THREAD_N] = {0.0,};
  float reg_a[THREAD_M];
  float reg_b[THREAD_N];

  for( int t = 0; t < K / TILE_SIZE; t++){
    const int row_tile = TILE_SIZE * t + threadIdx.y;
    const int col_tile = TILE_SIZE * t + threadIdx.x;
    
    for(int tm=0;tm<THREAD_M;tm++)
      local_a[threadIdx.y * THREAD_M + tm][threadIdx.x] = A[(i + tm)*K + col_tile];  
    
    for(int tn=0;tn<THREAD_N;tn++)
      local_b[threadIdx.y][threadIdx.x + TILE_SIZE * tn] = B[row_tile*N + (j + TILE_SIZE * tn)];


    __syncthreads();
  
    for(int tt=0;tt<TILE_SIZE;tt++){

      for(int reg_idx_a=0; reg_idx_a<THREAD_M;reg_idx_a++)
        reg_a[reg_idx_a] = local_a[threadIdx.y * THREAD_M + reg_idx_a][tt];

      // to avoid bank conflict as possible
      //thread 0 : 0, 32, 64, 96, 128  ...
      //thread 1 : 1, 33, 65, 97, 129,...
      for(int reg_idx_b=0; reg_idx_b<THREAD_N;reg_idx_b++)      
        reg_b[reg_idx_b] = local_b[tt][threadIdx.x + reg_idx_b * TILE_SIZE];

      for(int tm=0;tm<THREAD_M;tm++)
        for(int tn=0;tn<THREAD_N;tn++)
          sum[tm][tn] += reg_a[tm] * reg_b[tn];
    }

    __syncthreads();
  }
    for(int tm=0;tm<THREAD_M;tm++)
      for(int tn=0;tn<THREAD_N;tn++)
        C[(i + tm) * N + (j + tn * TILE_SIZE)] = sum[tm][tn];
}


static int Mbegin[NGPU], Mend[NGPU];
static int ngpu;
static cudaStream_t streams[NGPU];
static cudaStream_t streams_2[NGPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];
static float *A_gpu2[NGPU], *C_gpu2[NGPU];

// A : M * K
// B : K * N
// C : M * N



void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Async memcpy H->D on each GPU
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], B, K * N * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int j = 0; j < DOUBLE_BUFFER_DIV; j++){
      int a_base_idx = Mbegin[i] * K;
      int a_size_per_iter = (Mend[i] - Mbegin[i]) * K / DOUBLE_BUFFER_DIV;

      int c_base_idx = Mbegin[i] * N;
      int c_size_per_iter = (Mend[i] - Mbegin[i]) * N / DOUBLE_BUFFER_DIV;
      if ( j % 2 == 0){
        CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &A[ a_base_idx + j * a_size_per_iter],
                                a_size_per_iter * sizeof(float),
                                cudaMemcpyHostToDevice, streams[i]));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + (TILE_SIZE * THREAD_N) - 1) / (TILE_SIZE * THREAD_N), ((Mend[i] - Mbegin[i]) / DOUBLE_BUFFER_DIV + (TILE_SIZE * THREAD_M) - 1) / (TILE_SIZE * THREAD_M));
        matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
            A_gpu[i], B_gpu[i], C_gpu[i], (Mend[i] - Mbegin[i]) / DOUBLE_BUFFER_DIV, N, K);



        CHECK_CUDA(cudaMemcpyAsync(&C[c_base_idx + j * c_size_per_iter], C_gpu[i],
                                  c_size_per_iter * sizeof(float),
                                  cudaMemcpyDeviceToHost, streams[i]));
      }
      else{ 
        CHECK_CUDA(cudaMemcpyAsync(A_gpu2[i], &A[ a_base_idx + j * a_size_per_iter],
                                a_size_per_iter * sizeof(float),
                                cudaMemcpyHostToDevice, streams_2[i]));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + (TILE_SIZE * THREAD_N) - 1) / (TILE_SIZE * THREAD_N), ((Mend[i] - Mbegin[i]) / DOUBLE_BUFFER_DIV + (TILE_SIZE * THREAD_M) - 1) / (TILE_SIZE * THREAD_M));
        matmul_kernel<<<gridDim, blockDim, 0, streams_2[i]>>>(
            A_gpu2[i], B_gpu[i], C_gpu2[i], (Mend[i] - Mbegin[i]) / DOUBLE_BUFFER_DIV, N, K);



        CHECK_CUDA(cudaMemcpyAsync(&C[c_base_idx + j * c_size_per_iter], C_gpu2[i],
                                  c_size_per_iter * sizeof(float),
                                  cudaMemcpyDeviceToHost, streams_2[i]));
      }

      
    }


    CHECK_CUDA(cudaGetLastError());
  }

  for (int i = 0; i < ngpu; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }
}




void matmul_initialize(int M, int N, int K) {

  CHECK_CUDA(cudaGetDeviceCount(&ngpu));

  cudaDeviceProp props[4];
  for (int i = 0; i < ngpu; ++i) {
    CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
  }

  for (int i = 0; i < ngpu; i++) {
    Mbegin[i] = M / ngpu  * i;
    Mend[i] = M / ngpu * (i + 1);
    if (i == ngpu - 1) Mend[i] = M ;
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaStreamCreate(&streams_2[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(
        cudaMalloc(&A_gpu[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(
        cudaMalloc(&C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));

    CHECK_CUDA(
        cudaMalloc(&A_gpu2[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CHECK_CUDA(
        cudaMalloc(&C_gpu2[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }
}


void matmul_finalize() {
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
    CHECK_CUDA(cudaFree(A_gpu2[i]));
    CHECK_CUDA(cudaFree(C_gpu2[i]));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
    CHECK_CUDA(cudaStreamDestroy(streams_2[i]));
  }
}
