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

#define MAX_SEQ_LEN 154

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
		int iteration) {

	int n = ((tlen + 1) >> 1) * 2;
	int j = threadIdx.x;
	short* H = (short*) ((char*) H_d);
	short* H1 = (short*) ((char*) H + pitch);
	short* Hp = (short*) ((char*) H1 + pitch);
	short* E = (short*) ((char*) Hp + pitch);
	short* F = (short*) ((char*) E + pitch);
	short* Fp = (short*) ((char*) F + pitch);

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
	}

}

#define max(a,b) \
		({ __typeof__ (a) _a = (a); \
		__typeof__ (b) _b = (b); \
		_a > _b ? _a : _b; })

void runTest(int* qp, int qlen, uint8_t *query, int tlen, uint8_t *target,
		int o_del, int e_del) {

	int H[512][512];
	int H1[512][512];
	int F[512][512];
	int E[512][512];

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

kswr_t sw_kernel(int qlen, uint8_t *query, int tlen, uint8_t *target, int m,
		const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins,
		int minsc, int endsc) {

	int *qp = (int *) malloc(qlen * m * sizeof(int));
	kswr_t r = { 0, -1, -1, -1, -1, -1, -1 };
	int score = -1, te = -1, qe = -1, score2 = -1, te2 = -1;
	int *b = (int *)malloc(tlen*qlen*sizeof(int));
	int *bi = (int *)malloc(tlen*qlen*sizeof(int));
	int te_ind = 0;
	uint8_t *query_d = NULL, *target_d = NULL;
	int *qp_d = NULL;
	int8_t *mat_d = NULL;
	size_t pitch;
	short *H, *H_d;

	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors(cudaMalloc((void ** )&query_d, qlen * sizeof(uint8_t)));
	checkCudaErrors(cudaMalloc((void ** )&mat_d, m * m * sizeof(int8_t)));
	checkCudaErrors(cudaMalloc((void ** )&qp_d, qlen * m * sizeof(int)));
	checkCudaErrors(cudaMemcpy(query_d, query, qlen * sizeof(uint8_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(mat_d, mat, m * m * sizeof(int8_t), cudaMemcpyHostToDevice));

	// ################# query profile
	int threadsPerBlock = 256;
	int blocksPerGrid = (qlen + threadsPerBlock - 1) / threadsPerBlock;
	genqp_kernel<<<blocksPerGrid, threadsPerBlock>>>(qp_d, query_d, qlen, m, mat_d);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(qp, qp_d, m * qlen * sizeof(int), cudaMemcpyDeviceToHost));

	//runTest(qp, qlen, query, tlen, target, o_del, e_del);

	int MAX_SIZE = 512;
	if (tlen < MAX_SIZE) {
		H = (short*) malloc((tlen + 1) * sizeof(short));
		checkCudaErrors(cudaMalloc((void ** )&target_d, tlen * sizeof(uint8_t)));
		checkCudaErrors(cudaMallocPitch((void ** )&H_d, &pitch, (tlen + 1) * sizeof(short), 6));
		checkCudaErrors(cudaMemset2D(H_d, pitch, 0, (tlen + 1) * sizeof(short), 6));
		checkCudaErrors(cudaMemcpy(target_d, target, tlen * sizeof(uint8_t), cudaMemcpyHostToDevice));

		for (int i = 0; i < qlen; ++i) {
			sw_kernel2<<<1, MAX_SIZE>>>(qp_d, qlen, query_d, tlen, target_d, o_del, e_del, H_d, pitch, i);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaMemcpy(H, H_d, (tlen + 1) * sizeof(short), cudaMemcpyDeviceToHost));
			int ind = -1;
			int max = -1;
			for (int j = 0; j <= tlen; ++j){
				if (H[j] > max) {
					max = H[j];
					ind = j;
					if(max >= minsc){
						if(te_ind == 0 ||  bi[te_ind-1] + 1 != ind){
							b[te_ind] = max;
							bi[te_ind++] = ind;
						}
						else if(max > b[te_ind-1]){
							b[te_ind-1] = max;
							bi[te_ind-1] = ind;
						}
					}
				}
			}
			if (max > score) {
				score = max;
				te = ind;
				qe = i;
				if(score >= endsc) break;
			}
		}

		r.score = score;
		r.te = te - 1;
		r.qe = qe;
		int mmax = 0;
		for (int i = 0; i < m*m; ++i) // get the max score
			mmax = mmax > mat[i]? mmax : mat[i];
		if (te_ind>0) {
			int i = (r.score + mmax - 1) / mmax;
			int low = r.te - i; int high = r.te + i;
			for (i = 0; i < te_ind; ++i) {
				int e = bi[i];
				if ((e < low || e > high) && b[i] > r.score2)
					r.score2 = b[i], r.te2 = e;
			}
		}

		checkCudaErrors(cudaFree(target_d));
		checkCudaErrors(cudaFree(H_d));
		checkCudaErrors(cudaFree(query_d));
		checkCudaErrors(cudaFree(qp_d));
		checkCudaErrors(cudaFree(mat_d));
		checkCudaErrors(cudaDeviceReset());
		free(qp);
		free(H);
		free(b);
		free(bi);
	}
	return r;
}
