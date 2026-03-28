#include <stdio.h>
#include "fixedint.h"
#include "gpu_common.h"
#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

void __global__ profile_ops() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned char scalar[32] = {0};
    scalar[0] = 42; scalar[1] = 17; scalar[2] = 99;
    ge_p3 P;
    ge_scalarmult_base(&P, scalar);
    ge_precomp G_pre = base[0][0];
    unsigned char pub[32];
    ge_p1p1 R;
    int N = 10000;

    // 1. ge_madd + p1p1_to_p3 (point addition)
    clock_t t0 = clock();
    for (int i = 0; i < N; i++) {
        ge_madd(&R, &P, &G_pre);
        ge_p1p1_to_p3(&P, &R);
    }
    clock_t t1 = clock();

    // 2. ge_p3_tobytes (field inversion + serialize)
    clock_t t2 = clock();
    for (int i = 0; i < N; i++) {
        ge_p3_tobytes(pub, &P);
    }
    clock_t t3 = clock();

    // 3. fe_invert alone
    fe tz, ti;
    fe_copy(tz, P.Z);
    clock_t t4 = clock();
    for (int i = 0; i < N; i++) {
        fe_invert(ti, tz);
    }
    clock_t t5 = clock();

    // 4. fe_mul alone
    fe a, b, c;
    fe_copy(a, P.X); fe_copy(b, P.Y);
    clock_t t6 = clock();
    for (int i = 0; i < N; i++) {
        fe_mul(c, a, b);
    }
    clock_t t7 = clock();

    // 5. modular suffix check (6 chars)
    clock_t t8 = clock();
    for (int i = 0; i < N; i++) {
        unsigned long long int mod = 58ULL*58*58*58*58*58;
        unsigned long long int rem = 0;
        for (int j = 0; j < 32; j++)
            rem = (rem * 256ULL + (unsigned long long int)pub[j]) % mod;
        volatile int d = (int)(rem % 58ULL); // prevent optimization
    }
    clock_t t9 = clock();

    // 6. fe_tobytes alone
    clock_t t10 = clock();
    for (int i = 0; i < N; i++) {
        fe_tobytes(pub, P.Y);
    }
    clock_t t11 = clock();

    long long mul = (t7-t6)/N;
    printf("=== PROFILE (N=%d) ===\n", N);
    printf("fe_mul:          %6lld clocks (1.0x)\n", (long long)(t7-t6)/N);
    printf("ge_madd+to_p3:   %6lld clocks (%.1fx)\n", (long long)(t1-t0)/N, mul>0?(double)(t1-t0)/N/mul:0);
    printf("fe_invert:       %6lld clocks (%.0fx)\n", (long long)(t5-t4)/N, mul>0?(double)(t5-t4)/N/mul:0);
    printf("ge_p3_tobytes:   %6lld clocks (%.0fx)\n", (long long)(t3-t2)/N, mul>0?(double)(t3-t2)/N/mul:0);
    printf("fe_tobytes:      %6lld clocks (%.1fx)\n", (long long)(t11-t10)/N, mul>0?(double)(t11-t10)/N/mul:0);
    printf("mod_suffix_chk:  %6lld clocks (%.1fx)\n", (long long)(t9-t8)/N, mul>0?(double)(t9-t8)/N/mul:0);
    printf("\n");
    printf("Per-key cost breakdown (current batch-256):\n");
    long long add_cost = (t1-t0)/N;
    long long tobytes_cost = (t3-t2)/N;
    long long inv_cost = (t5-t4)/N;
    long long mod_cost = (t9-t8)/N;
    long long batch_per_key = add_cost + (inv_cost / 256) + (long long)((t11-t10)/N) + mod_cost;
    printf("  point_add:     %lld\n", add_cost);
    printf("  inversion/256: %lld\n", inv_cost/256);
    printf("  fe_tobytes:    %lld\n", (long long)(t11-t10)/N);
    printf("  mod_check:     %lld\n", mod_cost);
    printf("  TOTAL/key:     %lld clocks\n", batch_per_key);
    printf("  vs tobytes/key:%lld clocks (%.1fx faster with batch)\n", tobytes_cost, (double)tobytes_cost/batch_per_key);
}

int main() {
    printf("Profiling ed25519 GPU operations...\n");
    profile_ops<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 0;
}
