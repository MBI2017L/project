/*
 * kernel.cuh
 *
 *  Created on: Jun 11, 2017
 *      Author: ewa
 */

#ifndef KERNEL_CUH_
#define KERNEL_CUH_


#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "ksw.h"

#ifdef __cplusplus
extern "C" {
#endif
kswr_t sw_kernel(int qlen, uint8_t *query, int tlen, uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int minsc, int endsc);

#ifdef __cplusplus
}
#endif

#endif /* KERNEL_CUH_ */
