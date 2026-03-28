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

void __global__ dump_keys() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned char scalar[32];
    for (int i = 0; i < 32; i++) scalar[i] = (i * 7 + 13) & 0xFF;

    ge_p3 P;
    ge_scalarmult_base(&P, scalar);
    ge_precomp G = base[0][0];

    for (int k = 0; k < 20; k++) {
        unsigned char pub[32];
        ge_p3_tobytes(pub, &P);
        char addr[256] = {0};
        size_t s = 256;
        b58e(addr, &s, pub, 32);
        printf("key %2d: %s (len=%d byte0=%d)\n", k, addr, (int)(s-1), pub[0]);

        ge_p1p1 R;
        ge_madd(&R, &P, &G);
        ge_p1p1_to_p3(&P, &R);
    }
}

int main() {
    dump_keys<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
