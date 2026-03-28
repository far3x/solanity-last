#include "fixedint.h"
#include "gpu_common.h"
#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"

__device__ __constant__ char B58A[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
bool __device__ b58e(char *b58, size_t *sz, uint8_t *d, size_t bs) {
    const uint8_t *bin=d;int carry;size_t i,j,high,zc=0,size;
    while(zc<bs&&!bin[zc])++zc;size=(bs-zc)*138/100+1;
    uint8_t buf[256];memset(buf,0,size);
    for(i=zc,high=size-1;i<bs;++i,high=j)
        for(carry=bin[i],j=size-1;(j>high)||carry;--j){carry+=256*buf[j];buf[j]=carry%58;carry/=58;if(!j)break;}
    for(j=0;j<size&&!buf[j];++j);
    if(*sz<=zc+size-j){*sz=zc+size-j+1;return false;}
    if(zc)memset(b58,'1',zc);
    for(i=zc;j<size;++i,++j)b58[i]=B58A[buf[j]];
    b58[i]=0;*sz=i+1;return true;
}

void __global__ debug_scan() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned char scalar[32];
    // random-ish seed
    for (int i = 0; i < 32; i++) scalar[i] = (i * 37 + 99) & 0xFF;

    ge_p3 P;
    ge_scalarmult_base(&P, scalar);
    ge_precomp G = base[0][0];

    int match3 = 0, match4 = 0, match5 = 0, match6 = 0;
    int total = 0;

    const char* target = "DUSTED";

    for (int k = 0; k < 1000000; k++) {
        unsigned char pub[32];
        ge_p3_tobytes(pub, &P);

        char addr[256] = {0};
        size_t s = 256;
        b58e(addr, &s, pub, 32);

        // check how many chars match
        int matched = 0;
        for (int j = 0; target[j] != 0; j++) {
            if (addr[j] == target[j]) matched++;
            else break;
        }

        if (matched >= 3) {
            match3++;
            if (matched >= 4) match4++;
            if (matched >= 5) match5++;
            if (matched >= 6) {
                match6++;
                printf("FULL MATCH: %s\n", addr);
            }
        }

        // print first few near-misses
        if (matched >= 3 && match3 <= 5) {
            printf("  %d-match: %s (matched=%d)\n", matched, addr, matched);
        }

        total++;

        ge_p1p1 R;
        ge_madd(&R, &P, &G);
        ge_p1p1_to_p3(&P, &R);

        if (k % 100000 == 99999) {
            printf("checked %dk: 3+match=%d 4+match=%d 5+match=%d 6match=%d\n",
                (k+1)/1000, match3, match4, match5, match6);
        }
    }

    printf("\n=== FINAL after %d keys ===\n", total);
    printf("3+ char match: %d (expected ~%d)\n", match3, total / (58*58*58));
    printf("4+ char match: %d (expected ~%d)\n", match4, total / (58*58*58*58));
    printf("5+ char match: %d (expected ~%d)\n", match5, total / (58*58*58*58*58));
    printf("6  char match: %d\n", match6);
}

int main() {
    printf("Debug prefix matching - scanning 1M keys on 1 thread\n");
    printf("Target: DUSTED\n\n");
    fflush(stdout);
    debug_scan<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("Done.\n");
    fflush(stdout);
    return 0;
}
