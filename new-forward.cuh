
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
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

	int n, m, h0, w0, h_base, w_base, h, w;
	int X_tile_width = TILE_WIDTH + K-1;
	extern __shared__ float shared_mem[];
	float* X_shared = &shared_mem[0];
	float* W_shared = &shared_mem[X_tile_width * X_tile_width];
	n = blockIdx.x;
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
					X_shared[(i - h_base) * X_tile_width + (j - w_base)] = x4d(n, c, i, j);
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

	if (n < B && m < M && h < H_out && w < W_out) {
		y4d(n, m, h, w) = acc;
	}
	#undef y4d
	#undef x4d
	#undef k4d
}

__global__ void unroll_Kernel(int C, int H, int W, int n, int K, float* X, float* X_unroll)
{
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	int c, s, h_out, w-out, h_unroll, w_base, p, q;
	int t = blockIdx.x * 1024 + threadIdx.x;
	int H_out = H – K + 1;
	int W_out = W – K + 1;
	int W_unroll = H_out * W_out;
	if (t < C * W_unroll) {
		c = t / W_unroll;
		s = t % W_unroll;
		h_out = s / W_out;
		w_out = s % W_out;
		h_unroll = h_out * W_out + w_out;
		w_base = c * K * K;
		for(p = 0; p < K; p++) {
			for(q = 0; q < K; q++) {
				w_unroll = w_base + p * K + q;
				X_unroll(h_unroll*W_unroll + w_unroll) = x4d(n, c, h_out + p, w_out + q);
			}
		}
	}
	#undef x4d
}

void unroll_gpu(int C, int H, int W, int n, int K, float* X, float* X_unroll)
{
	int H_out = H – K + 1;
	int W_out = W – K + 1;
	int num_threads = C * H_out * W_out;
	int num_blocks = ceil((C * H_out * W_out) / 1024);
	unroll_Kernel<<<num_blocks, 1024>>>(C, H, W, n, K, X, X_unroll);
}

void convLayer_forward(int N, int M, int C, int H, int W, int K, float* X, float* W_unroll, float* Y)
{
	int W_out = W – K + 1;
	int H_out = H – K + 1;
	int W_unroll = C * K * K;
	int H_unroll = H_out * W_out;
	float* X_unrolled = malloc(W_unroll * H_unroll * sizeof(float));
	for (int n=0; n < N; n++) {
		unroll_gpu(C, H, W, K, n, X, X_unrolled);
		gemm(H_unroll, M, W_unroll, X_unrolled, W, Y[n]);
	}
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

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil((W_out)/16.0);
    int H_grid = ceil((H_out)/16.0);
    int Z = H_grid*W_grid;

    // Set the kernel dimensions
    dim3 gridDim(B,M,Z);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);

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
