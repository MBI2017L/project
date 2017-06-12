
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


__global__ void
genqp_kernel(int *res, uint8_t *query, int qlen, int m, int8_t *mat)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < qlen)
	{
		for(int k = 0;k<m;++k)
			res[i + k * qlen] = mat[k * m + query[i]];
	}
}


__device__ __forceinline__ short maxshort(short a, short b)
{
	return (a >= b) ? a : b;
}



__global__ void
sw_kernel(int* qp, int qlen, uint8_t *query, int tlen, uint8_t *target, int o_del, int e_del, int o_ins, int e_ins, int minsc, int endsc, int* result, int* result_ind){

	__shared__ short iteration;
	__shared__ short E[MAX_SEQ_LEN+1]; //cols
	__shared__ short F[MAX_SEQ_LEN+1]; //rows
	__shared__ short Fp[MAX_SEQ_LEN+1]; //rows
	__shared__ short H[MAX_SEQ_LEN+1]; //rows
	__shared__ short H1[MAX_SEQ_LEN+1]; //rows
	__shared__ short Hp[MAX_SEQ_LEN+1]; //rows
	__shared__ short maxInRow[MAX_SEQ_LEN+1]; //rows

	int n = (tlen>>1)*2;
	int j = threadIdx.x;
	H[j] = 0;
	F[j] = 0;
	Fp[j] = 0;
	E[j] = 0;
	Hp[j] = 0;
	H1[j] = 0;
	maxInRow[j] = 0;

	if (j == 0)
		iteration = 1;

	__syncthreads();

	while(iteration<=qlen){

		if (j > 0 && j <= tlen){
			int *q = &qp[target[j-1] * qlen];
			F[j] = maxshort(Fp[j], Hp[j] - o_del);
			F[j] -= e_del;
			H1[j] = maxshort(Hp[j-1] + q[iteration-1], F[j]);
			H1[j] = maxshort(H1[j], 0);
			E[j] = H1[j-1]+(j-1)*e_del;
		}

//		if(j==0){
//			for(int k=0;k<=tlen;++k)
//				printf("%d ", E[k]);
//			printf("\n");
//		}

		__syncthreads();
		for (int offset = 1; offset < n; offset *= 2)
		{
			if (j >= offset)
			E[j] = maxshort(E[j], E[j - offset]);
			__syncthreads();
		}

//		if(j==0){
//			for(int k=0;k<=tlen;++k)
//				printf("%d ", E[k]);
//			printf("\n");
//		}

		if (j > 0 && j <= tlen){
			E[j] = maxshort(E[j] - (j)*e_del, 0);
			H[j] = maxshort(H1[j], E[j] - o_del);
			Hp[j] = H[j];
			Fp[j] = F[j];
			maxInRow[j] = maxshort(maxInRow[j], H[j]);
		}

//		if(j==0){
//			for(int k=0;k<=tlen;++k)
//				printf("%d ", H[k]);
//			printf("\n");
//		}

		__syncthreads();
		for (int offset = 1; offset < n; offset *= 2)
		{
			if (j >= offset)
				maxInRow[j] =maxshort(maxInRow[j], maxInRow[j - offset]);
			__syncthreads();
		}


		if (j == 0)
			++iteration;

		__syncthreads();
	}
	if (j == 0) {
		*result = maxInRow[tlen];
		//printf("KERNEL RESULT: %d\n", maxInRow[tlen]);

	}

}

#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

void runTest(){
	char* target = "CAGCCTCGCTTAG";
	char* query = "AATGCCATTGCCGG";
	int ok = 5;
	int miss = -3;
	int gs = 8;
	int ge = 2;

	int H[15][14];
	int H1[15][14];
	int F[15][14];
	int E[15][14];

	for(int i = 0; i< 15; ++i)
		for(int j = 0; j<14;++j){
			H[i][j]=0;
			H1[i][j]=0;
			F[i][j]=0;
			E[i][j]=0;
		}

	for(int i = 1; i<15; ++i){
		for(int j = 1; j<14;++j){
			F[i][j] = max(F[i-1][j], H[i-1][j] - gs);
			F[i][j] -= ge;
			H1[i][j] = max(H[i-1][j-1] + (query[i-1] == target[j-1] ? ok : miss), F[i][j]);
			H1[i][j] = max(H1[i][j], 0);
			for(int k=1;k<j;++k){
				E[i][j] = max(H[i][j-k] - k*ge, E[i][j]);
			}
			H[i][j] = max(H1[i][j], E[i][j] - gs);
		}
	}
//	for(int i = 0; i< 14; ++i){
//		for(int j = 0; j<14;++j){
//			printf("%d ", H[i][j]);
//		}
//		printf("\n");
//	}
//	printf("\n");
}

int sw_kernel(int qlen, uint8_t *query, int tlen, uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int minsc, int endsc){
	checkCudaErrors(cudaSetDevice(0));
	uint8_t *query_d = NULL, *target_d = NULL;
	int *qp_d = NULL;
	int8_t *mat_d = NULL;
	int *qp = (int *)malloc(qlen * m * sizeof(int));
	int *result_d = NULL, *result_ind_d = NULL;
	int result = 0, result_ind = 0;
	checkCudaErrors(cudaMalloc((void **)&query_d, qlen*sizeof(uint8_t)));
	checkCudaErrors(cudaMalloc((void **)&mat_d, m*m*sizeof(int8_t)));
	checkCudaErrors(cudaMalloc((void **)&qp_d, qlen*m*sizeof(int)));
	checkCudaErrors(cudaMemcpy(query_d, query, qlen*sizeof(uint8_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(mat_d, mat, m*m*sizeof(int8_t), cudaMemcpyHostToDevice));
	// ################# query profile
	int threadsPerBlock = 256;
	int blocksPerGrid = (qlen + threadsPerBlock - 1) / threadsPerBlock;
	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	genqp_kernel<<<blocksPerGrid, threadsPerBlock>>>(qp_d, query_d, qlen, m, mat_d);
	checkCudaErrors(cudaGetLastError());
	// ################# end query profile
	if(qlen<MAX_SEQ_LEN && tlen<MAX_SEQ_LEN){
		//runTest();
		checkCudaErrors(cudaMalloc((void **)&target_d, tlen*sizeof(uint8_t)));
		checkCudaErrors(cudaMemcpy(target_d, target, tlen*sizeof(uint8_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **)&result_d, sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&result_ind_d, sizeof(int)));
		sw_kernel<<<1, MAX_SEQ_LEN>>>(qp_d, qlen, query_d, tlen, target_d, o_del, e_del, o_ins, e_ins, minsc, endsc, result_d, result_ind_d);

		checkCudaErrors(cudaMemcpy(&result, result_d, sizeof(int),
				cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&result_ind, result_ind_d, sizeof(int),
				cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(target_d));
		checkCudaErrors(cudaFree(result_d));
		checkCudaErrors(cudaFree(result_ind_d));
	}
	checkCudaErrors(cudaFree(query_d));
	checkCudaErrors(cudaMemcpy(qp, qp_d, m*qlen*sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(qp_d));
	checkCudaErrors(cudaFree(mat_d));
	checkCudaErrors(cudaDeviceReset());
	free(qp);
	return result;
}
