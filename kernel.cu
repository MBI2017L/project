/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "kernel.cuh"
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <helper_timer.h>

#define MAX_LEN 512

uint8_t *query_d = NULL, *target_d = NULL;
int *qp_d = NULL;
int8_t *mat_d = NULL;
size_t pitch;
short *H_d;
int *results_d;
int *b_d, *bi_d;

__global__ void genqp_kernel(int *res, uint8_t *query, int qlen, int m,
		int8_t *mat) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < qlen) {
		for (int k = 0; k < m; ++k)
			res[i + k * qlen] = mat[k * m + query[i]];
	}
}

__device__ __forceinline__ short maxshort(short a, short b) {
	return (a >= b) ? a : b;
}

__global__ void sw_kernel2(int* qp, int qlen, uint8_t *query, int tlen,
		uint8_t *target, int o_del, int e_del, short* H_d, int pitch,
		int minsc, int endsc, int* b, int* bi, int *results) {

	int j = threadIdx.x;

	int n = ((tlen + 1) >> 1) * 2;
	short* H = (short*)((char*)H_d);
	short* H1 = (short*)((char*)H + pitch);
	short* Hp = (short*)((char*)H1 + pitch);
	short* E = (short*)((char*)Hp + pitch);
	short* F = (short*)((char*)E + pitch);
	short* Fp = (short*)((char*)F + pitch);

	__shared__ int iteration, max, ind, te_ind;
	__shared__ bool anyoneBetter;

	if (j == 0) {
		iteration = 0;
		te_ind = 0;
		anyoneBetter = false;
		results[0] = -1;	//score
		results[1] = -1;	//te
		results[2] = -1;	//qe
		results[3] = -1;	//te_ind
	}

	while(iteration < qlen) {

		if (j > 0 && j <= tlen) {
			int *q = &qp[target[j - 1] * qlen];
			F[j] = maxshort(Fp[j], Hp[j] - o_del);
			F[j] -= e_del;
			H1[j] = maxshort(Hp[j - 1] + q[iteration], F[j]);
			H1[j] = maxshort(H1[j], 0);
			__syncthreads();
			E[j] = maxshort(j * e_del, H1[j - 1] + (j - 1) * e_del);
		}

		__syncthreads();
		for (int offset = 1; offset < n; offset *= 2) {
			if (j >= offset && j <= tlen)
				E[j] = maxshort(E[j], E[j - offset]);
			__syncthreads();
		}

		if (j > 0 && j <= tlen) {
			E[j] -= j * e_del;
			H[j] = maxshort(H1[j], E[j] - o_del);
			Hp[j] = H[j];
			Fp[j] = F[j];
			if (H[j] > results[0]) anyoneBetter = true;
		}
		__syncthreads();

		if (j == 0) {
			if (anyoneBetter) {

				ind = -1;
				max = -1;
				for (int k = 0; k <= tlen; ++k) {
					if (H[k] > max) {
						max = H[k];
						ind = k;

						if (max >= minsc) {
							if (te_ind == 0 || bi[te_ind - 1] + 1 != ind) {
								b[te_ind] = max;
								bi[te_ind++] = ind;
							}
							else if (max > b[te_ind - 1]) {
								b[te_ind - 1] = max;
								bi[te_ind - 1] = ind;
							}
						}
					}
				}

				if (max > results[0]) {
					results[0] = max;
					results[1] = ind - 1;
					results[2] = iteration;
					results[3] = te_ind;

					if (max >= endsc) iteration = qlen;
				}

			}
			iteration++;
			anyoneBetter = false;
		}
		__syncthreads();
	}
}

/*#define max(a,b) \
		({ __typeof__ (a) _a = (a); \
		__typeof__ (b) _b = (b); \
		_a > _b ? _a : _b; })*/

void runTest(int* qp, int qlen, uint8_t *query, int tlen, uint8_t *target,
		int o_del, int e_del) {

	int H[MAX_LEN][MAX_LEN];
	int H1[MAX_LEN][MAX_LEN];
	int F[MAX_LEN][MAX_LEN];
	int E[MAX_LEN][MAX_LEN];

	for (int i = 0; i <= qlen; ++i)
		for (int j = 0; j <= tlen; ++j) {
			H[i][j] = 0;
			H1[i][j] = 0;
			F[i][j] = 0;
			E[i][j] = 0;
		}

	for (int i = 1; i <= qlen; ++i) {
		for (int j = 1; j <= tlen; ++j) {
			F[i][j] = max(F[i - 1][j], H[i - 1][j] - o_del);
			F[i][j] -= e_del;
			H1[i][j] = max(H[i - 1][j - 1] + qp[target[j - 1] * qlen + i - 1], F[i][j]);
			H1[i][j] = max(H1[i][j], 0);
			for (int k = 1; k < j; ++k) {
				//printf("%d %d\n", H1[i][j-k], E[i][j]);
				E[i][j] = max(H1[i][j - k] - k * e_del, E[i][j]);
			}
			H[i][j] = max(H1[i][j], E[i][j] - o_del);
			//printf("%d %d %d %d \t", H[i][j], H1[i][j], F[i][j], E[i][j]);
			//printf("%d ", E[i][j]);
		}
		//printf("\n");
	}
	//	for(int i = 0; i<=qlen; ++i){
	//		for(int j = 0; j<=tlen;++j){
	//			printf("%d ", H[i][j]);
	//		}
	//		printf("\n");
	//	}
	//	printf("\n");
}


int nextPow2( int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void initCUDA(){

	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors(cudaMalloc((void ** )&query_d, MAX_LEN * sizeof(uint8_t)));
	checkCudaErrors(cudaMalloc((void ** )&mat_d, MAX_LEN * MAX_LEN * sizeof(int8_t)));
	checkCudaErrors(cudaMalloc((void ** )&qp_d, MAX_LEN * MAX_LEN * sizeof(int)));
	checkCudaErrors(cudaMalloc((void ** )&target_d, MAX_LEN * sizeof(uint8_t)));
	checkCudaErrors(cudaMallocPitch((void ** )&H_d, &pitch, MAX_LEN * sizeof(short), 6));

	checkCudaErrors(cudaMalloc((void **)&b_d, MAX_LEN * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&bi_d, MAX_LEN * sizeof(int)));

	checkCudaErrors(cudaMalloc((void **)&results_d, 4 * sizeof(int)));
}

void freeCUDA(){
	checkCudaErrors(cudaFree(target_d));
	checkCudaErrors(cudaFree(H_d));
	checkCudaErrors(cudaFree(query_d));
	checkCudaErrors(cudaFree(qp_d));
	checkCudaErrors(cudaFree(mat_d));
	checkCudaErrors(cudaFree(results_d));
	checkCudaErrors(cudaFree(b_d));
	checkCudaErrors(cudaFree(bi_d));
}


kswr_t sw_kernel(int qlen, uint8_t *query, int tlen, uint8_t *target, int m,
		const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins,
		int minsc, int endsc) {

	int *qp = (int *) malloc(qlen * m * sizeof(int));
	kswr_t r = { 0, -1, -1, -1, -1, -1, -1 };
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);


	//runTest(qp, qlen, query, tlen, target, o_del, e_del);
	if(tlen < MAX_LEN){

		int *b = (int *)malloc(tlen * sizeof(int));
		int *bi = (int *)malloc(tlen * sizeof(int));
		int *results = (int*)malloc(4 * sizeof(int));

		checkCudaErrors(cudaMemcpy(query_d, query, qlen * sizeof(uint8_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mat_d, mat, m * m * sizeof(int8_t), cudaMemcpyHostToDevice));

		// ################# query profile
		int threadsPerBlock = 256;
		int blocksPerGrid = (qlen + threadsPerBlock - 1) / threadsPerBlock;
		genqp_kernel<<<blocksPerGrid, threadsPerBlock>>>(qp_d, query_d, qlen, m, mat_d);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpy(qp, qp_d, m * qlen * sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(target_d, target, tlen * sizeof(uint8_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset2D(H_d, pitch, 0, (tlen + 1) * sizeof(short), 6));

		sw_kernel2<<<1, nextPow2(tlen)>>>(qp_d, qlen, query_d, tlen, target_d, o_del, e_del, H_d, pitch, minsc, endsc, b_d, bi_d, results_d);
		checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaMemcpy(results, results_d, 4 * sizeof(int), cudaMemcpyDeviceToHost));
		r.score = results[0];
		r.te = results[1];
		r.qe = results[2];

		int te_ind = results[3];
		checkCudaErrors(cudaMemcpy(b, b_d, tlen * sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(bi, bi_d, tlen * sizeof(int), cudaMemcpyDeviceToHost));

		int mmax = 0;
		for (int i = 0; i < m*m; ++i) // get the max score
			mmax = mmax > mat[i] ? mmax : mat[i];
		if (te_ind>0) {
			int i = (r.score + mmax - 1) / mmax;
			int low = r.te - i; int high = r.te + i;
			for (i = 0; i < te_ind; ++i) {
				int e = bi[i];
				if ((e < low || e > high) && b[i] > r.score2)
					r.score2 = b[i], r.te2 = e;
			}
		}
	}
	free(qp);
	return r;
}
