
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define TILE_WIDTH1 24
#define TILE_WIDTH1 24
#define TILE_WIDTH2 16
#define CUDA_MAX_NUM_THREADS 1024

__constant__ float weightMatrix[24*12*7*7];

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil((W_out)/16.0);

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int b, m, h, w, c, p, q;
	b = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
	w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;
	float acc = 0.;
	if (b < B && m < M && h < H_out && w < W_out) {
	  for (c = 0; c < C; c++) { // sum over all input channels
		for (p = 0; p < K; p++) {// loop over KxK filter
		  for (q = 0; q < K; q++) {
			acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
		  }
		}
	  }
	  y4d(b, m, h, w) = acc;
	}
	#undef y4d
	#undef x4d
	#undef k4d
}

__global__ void forward_kernel_constant(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil((W_out)/16.0);

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	#define ck4d(i3, i2, i1, i0) weightMatrix[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int b, m, h, w, c, p, q;
	b = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
	w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;
	float acc = 0.;
	if (b < B && m < M && h < H_out && w < W_out) {
	  for (c = 0; c < C; c++) { // sum over all input channels
		for (p = 0; p < K; p++) {// loop over KxK filter
		  for (q = 0; q < K; q++) {
			acc += x4d(b, c, h + p, w + q) * ck4d(m, c, p, q);
		  }
		}
	  }
	  y4d(b, m, h, w) = acc;
	}
	#undef y4d
	#undef x4d
	#undef k4d
 	#undef ck4d
}

__global__ void forward_kernel_shared(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

	const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	int W_grid = ceil((W_out)/16.0);

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int b, m, h0, w0, h_base, w_base, h, w;
	int X_tile_width = TILE_WIDTH + K-1;
	extern __shared__ float shared_mem[];
	float* X_shared = &shared_mem[0];
	float* W_shared = &shared_mem[X_tile_width * X_tile_width];
	b = blockIdx.x;
	m = blockIdx.y;
	h0 = threadIdx.y;
	w0 = threadIdx.x;
	h_base = (blockIdx.z / W_grid) * TILE_WIDTH; // vertical base out data index for the block
	w_base = (blockIdx.z % W_grid) * TILE_WIDTH; // horizontal base out data index for the block
	h = h_base + h0;
	w = w_base + w0;
	float acc = 0.0;
	for(int c = 0; c < C; c++) {	// sum over all input channels
									// load weights for W [m, c,..],
									// h0 and w0 used as shorthand for threadIdx.x
									// and threadIdx.y
		if ((h0 < K) && (w0 < K)) {
			W_shared[h0 * K + w0]= k4d(m, c, h0, w0);
		}
		__syncthreads();

		// load tile from X[n, c,…] into shared memory
		for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
			for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
				if (i < H && j < W) {
					X_shared[(i - h_base) * X_tile_width + (j - w_base)] = x4d(b, c, i, j);
				} else {
					X_shared[(i - h_base) * X_tile_width + (j - w_base)] = 0;
				}
			}
		}
		__syncthreads();

		for(int p = 0; p < K; p++) {
			for(int q = 0; q < K; q++) {
				if((h0 + p) < X_tile_width && (w0 + q) < X_tile_width) {
					acc += X_shared[(h0 + p) * X_tile_width + (w0 + q)] * W_shared[p * K + q];
				}
			}
		}
		__syncthreads();
	}

	if (b < B && m < M && h < H_out && w < W_out) {
		y4d(b, m, h, w) = acc;
	}
	#undef y4d
	#undef x4d
	#undef k4d
}

__global__ void unroll_Kernel(int C, int H, int W, int b, int K, float* x, float* X_unroll)
{
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	int c, s, row_out, col_out, col_unroll, row_unroll, w_base, p, q;
	int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_unroll = H_out * W_out;
	if (t < C * W_unroll) {
		c = t / W_unroll;
		s = t % W_unroll;
		row_out = s / W_out;
		col_out = s % W_out;
		col_unroll = row_out * W_out + col_out;
		w_base = c * K * K;
		for(p = 0; p < K; p++) {
			for(q = 0; q < K; q++) {
				row_unroll = w_base + p * K + q;
				X_unroll[row_unroll * W_unroll + col_unroll] = x4d(b, c, row_out + p, col_out + q);
			}
		}
	}
	#undef x4d
}

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  int col = threadIdx.x + blockIdx.x * TILE_WIDTH;
  int row = threadIdx.y + blockIdx.y * TILE_WIDTH;

  float val = 0;

  for(int i = 0; i < ceil(1.0 * numAColumns / TILE_WIDTH); i++) {
    if(row < numARows && (i * TILE_WIDTH + threadIdx.x) < numAColumns) {
      tileA[threadIdx.y][threadIdx.x] = A[row * numAColumns + (i * TILE_WIDTH + threadIdx.x)];
    } else {
      tileA[threadIdx.y][threadIdx.x] = 0;
    }

    if(col < numBColumns && (i * TILE_WIDTH + threadIdx.y) < numBRows) {
      tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * numBColumns + col];
    } else {
      tileB[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    for(int j = 0; j < TILE_WIDTH; j++) {
      val += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
    }

    __syncthreads();
  }

  if(row < numCRows && col < numCColumns) {
    C[row * numCColumns + col] = val;
  }
}

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if((col < numCColumns) && (row < numCRows)) {
    float val = 0;

    for(int i = 0; i < numAColumns; i++) {
      val += A[row * numAColumns + i] * B[i * numBColumns + col];
    }

    C[row * numCColumns + col] = val;
  }
}

__global__ void matrixMultiplyUnroll(int C, int H, int W, int b, int K, float * x, float *y, float *k,
																																				int numARows, int numAColumns,
																																				int numBRows,	int numBColumns,
																																				int numCRows, int numCColumns) {
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int H_out = H - K + 1;
	int W_out = W - K + 1;

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if((col < numCColumns) && (row < numCRows)) {
    float val = 0;

    for(int i = 0; i < numAColumns; i++) {
			int x_unrolled_index = i * numBColumns + col;

			int x_index_c =  i / (K * K);
			int x_index_s = i % (K * K);

			int x_index_row = (x_index_s / K) + (col / W_out);
			int x_index_col = (x_index_s % K) + (col % W_out);

      val += k[row * numAColumns + i] * x4d(b, x_index_c, x_index_row, x_index_col);
    }

    y[row * numCColumns + col] = val;
  }

	#undef y4d
	#undef x4d
	#undef k4d
}

__global__ void matrixMultiplySharedUnroll1(int M, int C, int H, int W, int K, const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ k,
																																							int numKRows, int numKColumns,
																																							int numXRows,	int numXColumns,
																																							int numYRows, int numYColumns) {

	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

  __shared__ float tileK[TILE_WIDTH1][TILE_WIDTH1];
  __shared__ float tileX[TILE_WIDTH1][TILE_WIDTH1];

  int col = threadIdx.x + blockIdx.x * TILE_WIDTH1;
  int row = threadIdx.y + blockIdx.y * TILE_WIDTH1;

	int W_out = W - 6;
	int H_out = H - 6;

	int kernel_size = 49;
	int row_offset = col / W_out;
	int col_offset = col % W_out;

  float val = 0;

	int num_tiles = ceil(1.0 * numKColumns / TILE_WIDTH1);

  for(int i = 0; i < num_tiles; i++) {
		int col_index = (i * TILE_WIDTH1 + threadIdx.x);
    if(row < numKRows && col_index < numKColumns) {
      tileK[threadIdx.y][threadIdx.x] = k[row * numKColumns + col_index];
    } else {
      tileK[threadIdx.y][threadIdx.x] = 0;
    }

		int row_index = (i * TILE_WIDTH1 + threadIdx.y);
    if(col < numXColumns && row_index < numXRows) {
			int x_index_c = row_index / kernel_size;
			int x_index_s = row_index % kernel_size;

			int x_index_row = (x_index_s / 7) + row_offset;
			int x_index_col = (x_index_s % 7) + col_offset;

      tileX[threadIdx.y][threadIdx.x] = x4d(blockIdx.z, x_index_c, x_index_row, x_index_col);
    } else {
      tileX[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < TILE_WIDTH1; j++) {
      val += tileK[threadIdx.y][j] * tileX[j][threadIdx.x];
    }

    __syncthreads();
  }

  if(row < numYRows && col < numYColumns) {
    y[blockIdx.z * (M * H_out * W_out) + row * numYColumns + col] = val;
  }

	#undef y4d
	#undef x4d
	#undef k4d
}

__global__ void matrixMultiplySharedUnroll2(int M, int C, int H, int W, int K, const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ k,
																																							int numKRows, int numKColumns,
																																							int numXRows,	int numXColumns,
																																							int numYRows, int numYColumns) {

	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

  __shared__ float tileK[TILE_WIDTH2][TILE_WIDTH2];
  __shared__ float tileX[TILE_WIDTH2][TILE_WIDTH2];

  int col = threadIdx.x + blockIdx.x * TILE_WIDTH2;
  int row = threadIdx.y + blockIdx.y * TILE_WIDTH2;

	int W_out = W - 6;
	int H_out = H - 6;

	int kernel_size = 49;
	int row_offset = col / W_out;
	int col_offset = col % W_out;

  float val = 0;

	int num_tiles = ceil(1.0 * numKColumns / TILE_WIDTH2);

  for(int i = 0; i < num_tiles; i++) {
		int col_index = (i * TILE_WIDTH2 + threadIdx.x);
    if(row < numKRows && col_index < numKColumns) {
      tileK[threadIdx.y][threadIdx.x] = k[row * numKColumns + col_index];
    } else {
      tileK[threadIdx.y][threadIdx.x] = 0;
    }

		int row_index = (i * TILE_WIDTH2 + threadIdx.y);
    if(col < numXColumns && row_index < numXRows) {
			int x_index_c = row_index / kernel_size;
			int x_index_s = row_index % kernel_size;

			int x_index_row = (x_index_s / 7) + row_offset;
			int x_index_col = (x_index_s % 7) + col_offset;

      tileX[threadIdx.y][threadIdx.x] = x4d(blockIdx.z, x_index_c, x_index_row, x_index_col);
    } else {
      tileX[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < TILE_WIDTH; j++) {
      val += tileK[threadIdx.y][j] * tileX[j][threadIdx.x];
    }

    __syncthreads();
  }

  if(row < numYRows && col < numYColumns) {
    y[blockIdx.z * (M * H_out * W_out) + row * numYColumns + col] = val;
  }

	#undef y4d
	#undef x4d
	#undef k4d
}

__global__ void matrixMultiplySharedConstUnroll(int M, int C, int H, int W, int K, const float* __restrict__ x, float* __restrict__ y,
																																							int numKRows, int numKColumns,
																																							int numXRows,	int numXColumns,
																																							int numYRows, int numYColumns) {

	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

  __shared__ float tileX[TILE_WIDTH][TILE_WIDTH];

  int col = threadIdx.x + blockIdx.x * TILE_WIDTH;
  int row = threadIdx.y + blockIdx.y * TILE_WIDTH;

	int H_out = H - K + 1;
	int W_out = W - K + 1;

  float val = 0;

  for(int i = 0; i < ceil(1.0 * numKColumns / TILE_WIDTH); i++) {
    if(col < numXColumns && (i * TILE_WIDTH + threadIdx.y) < numXRows) {
			int x_index_c =  (i * TILE_WIDTH + threadIdx.y) / (K * K);
			int x_index_s = (i * TILE_WIDTH + threadIdx.y) % (K * K);

			int x_index_row = (x_index_s / K) + (col / W_out);
			int x_index_col = (x_index_s % K) + (col % W_out);

      tileX[threadIdx.y][threadIdx.x] = x4d(blockIdx.z, x_index_c, x_index_row, x_index_col);
    } else {
      tileX[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < TILE_WIDTH; j++) {
      val += weightMatrix[row * numKColumns + (i * TILE_WIDTH + j)] * tileX[j][threadIdx.x];
    }

    __syncthreads();
  }

  if(row < numYRows && col < numYColumns) {
    y[blockIdx.z * (M * H_out * W_out) + row * numYColumns + col] = val;
  }

	#undef y4d
	#undef x4d
	#undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - 7 + 1;
    const int W_out = W - 7 + 1;

	/*
	int W_grid = ceil((W_out)/16.0);
    int H_grid = ceil((H_out)/16.0);
    int Z = H_grid*W_grid;

    // Set the kernel dimensions
    dim3 gridDim(B,M,Z);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
	*/

    // Call the kernel
    //forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    /*
    const float* kernel = w.dptr_;
    float hostKernel[M*C*K*K];
    cudaMemcpy(hostKernel, kernel, M*C*K*K* sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpyToSymbol(weightMatrix,hostKernel,sizeof(float)*M*C*K*K);
    forward_kernel_constant<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    */


    /*
	  size_t shared_size = sizeof(float) * ((TILE_WIDTH + K-1) * (TILE_WIDTH + K-1) + K * K);
    forward_kernel_shared<<<gridDim, blockDim, shared_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    */

	/*
	int W_unroll = H_out * W_out;
	int H_unroll = C * K * K;

	float* X_unrolled;
	cudaMalloc((void **) &X_unrolled, W_unroll * H_unroll * sizeof(float));

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid(ceil((1.0 * W_unroll)/TILE_WIDTH), ceil((1.0 * M)/TILE_WIDTH), 1);

	int num_blocks = ceil((1.0 * C * H_out * W_out) / CUDA_MAX_NUM_THREADS);

	for (int b = 0; b < B; b++) {
		float* curr_output = &y.dptr_[b * M * H_out * W_out];
		unroll_Kernel<<<num_blocks, CUDA_MAX_NUM_THREADS>>>(C, H, W, b, K, x.dptr_, X_unrolled);

		matrixMultiplyShared<<<dimGrid, dimBlock>>>(w.dptr_, X_unrolled, curr_output,
													M, H_unroll,
													H_unroll, W_unroll,
													M, W_unroll);
	}
	*/

	int W_unroll = H_out * W_out;
	int H_unroll = C * 7 * 7;

	if(W_unroll < 1000) {
		dim3 dimBlock(TILE_WIDTH1, TILE_WIDTH1, 1);
		dim3 dimGrid(ceil((1.0 * W_unroll)/TILE_WIDTH1), ceil((1.0 * M)/TILE_WIDTH1), B);

		matrixMultiplySharedUnroll1<<<dimGrid, dimBlock>>>(M, C, H, W, 7, x.dptr_, y.dptr_, w.dptr_,
														M, H_unroll,
														H_unroll, W_unroll,
														M, W_unroll);

	} else{
		dim3 dimBlock(TILE_WIDTH2, TILE_WIDTH2, 1);
		dim3 dimGrid(ceil((1.0 * W_unroll)/TILE_WIDTH2), ceil((1.0 * M)/TILE_WIDTH2), B);

		matrixMultiplySharedUnroll2<<<dimGrid, dimBlock>>>(M, C, H, W, 7, x.dptr_, y.dptr_, w.dptr_,
																M, H_unroll,
																H_unroll, W_unroll,
																M, W_unroll);
	}

	/*
	const float* kernel = w.dptr_;
	float hostKernel[M*C*K*K];
	cudaMemcpy(hostKernel, kernel, M*C*K*K* sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpyToSymbol(weightMatrix,hostKernel,sizeof(float)*M*C*K*K);

	int W_unroll = H_out * W_out;
	int H_unroll = C * K * K;

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid(ceil((1.0 * W_unroll)/TILE_WIDTH), ceil((1.0 * M)/TILE_WIDTH), B);

	matrixMultiplySharedConstUnroll<<<dimGrid, dimBlock>>>(M, C, H, W, K, x.dptr_, y.dptr_,
													M, H_unroll,
													H_unroll, W_unroll,
													M, W_unroll);
	*/
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
	MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
