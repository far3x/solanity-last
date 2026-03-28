/*
 * INCREMENTAL VANITY GENERATOR with BATCHED FIELD INVERSION
 *
 * Key optimizations:
 * 1. Incremental: P += G instead of full scalarmult (10 vs 2530 field muls)
 * 2. Batched inversion: Montgomery's trick — 1 inversion per BATCH_SIZE keys
 *    instead of 1 per key. Eliminates the ge_p3_tobytes bottleneck.
 * 3. Modular suffix check: O(32) mods instead of O(1408) for b58enc
 *
 * CSPRNG-safe: master seed from std::random_device (OS entropy)
 */

#include <random>
#include <chrono>
#include <iostream>
#include <ctime>
#include <stdio.h>

#include "fixedint.h"
#include "gpu_common.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

/* -- Constants ------------------------------------------------------------- */
#define BATCH_SIZE 32  // points to accumulate before batch inversion

__device__ __constant__ char B58_ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const char H_B58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
int b58_digit(char c) { for (int i = 0; i < 58; i++) if (H_B58[i] == c) return i; return -1; }

/* -- Device helpers -------------------------------------------------------- */

bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz) {
	const uint8_t *bin=data; int carry; size_t i,j,high,zcount=0,size;
	while(zcount<binsz&&!bin[zcount])++zcount;
	size=(binsz-zcount)*138/100+1;
	uint8_t buf[256]; memset(buf,0,size);
	for(i=zcount,high=size-1;i<binsz;++i,high=j)
		for(carry=bin[i],j=size-1;(j>high)||carry;--j){carry+=256*buf[j];buf[j]=carry%58;carry/=58;if(!j)break;}
	for(j=0;j<size&&!buf[j];++j);
	if(*b58sz<=zcount+size-j){*b58sz=zcount+size-j+1;return false;}
	if(zcount)memset(b58,'1',zcount);
	for(i=zcount;j<size;++i,++j) b58[i]=B58_ALPHABET[buf[j]];
	b58[i]='\0'; *b58sz=i+1; return true;
}

__device__ bool check_suffix_mod(uint8_t* pubkey, int* suffix_digits, int suffix_len) {
	unsigned long long int modulus = 1;
	for (int i = 0; i < suffix_len; i++) modulus *= 58;
	unsigned long long int rem = 0;
	for (int i = 0; i < 32; i++)
		rem = (rem * 256ULL + (unsigned long long int)pubkey[i]) % modulus;
	for (int i = suffix_len - 1; i >= 0; i--) {
		if ((int)(rem % 58ULL) != suffix_digits[i]) return false;
		rem /= 58ULL;
	}
	return true;
}

// Ultra-fast prefix pre-filter: check first 2 bytes against precomputed range.
// Rejects 99.97%+ of keys with just 2 byte comparisons.
// Only keys that pass go through full b58enc.
__device__ bool check_prefix_prefilter(uint8_t* pubkey, int byte0_min, int byte0_max) {
	return (pubkey[0] >= byte0_min && pubkey[0] <= byte0_max);
}

// Fast prefix check: uses prefilter + full b58enc only on candidates
__device__ bool check_prefix_fast(uint8_t* pubkey, int* prefix_digits, int prefix_len) {
	// Full base58 encode into buf[], but stop after we have enough leading digits
	// The b58enc algorithm produces digits in buf[] from MSB to LSB.
	// We can bail early once we've checked the prefix.

	uint8_t buf[64];
	size_t size = 45; // max b58 length for 32 bytes
	memset(buf, 0, size);

	// Standard b58 encode inner loop
	size_t zcount = 0;
	while (zcount < 32 && !pubkey[zcount]) ++zcount;

	size_t high;
	for (size_t i = zcount, high2 = size - 1; i < 32; ++i, high2 = high) {
		int carry = pubkey[i];
		size_t j;
		for (j = size - 1; ; --j) {
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (j == 0 || (j <= high2 && !carry)) break;
		}
		high = j;
	}

	// Find first non-zero digit
	size_t start = 0;
	while (start < size && buf[start] == 0) start++;

	// Check prefix digits
	for (int i = 0; i < prefix_len; i++) {
		int digit = (start + i < size) ? buf[start + i] : 0;
		if (prefix_digits[i] >= 0 && digit != prefix_digits[i]) return false;
	}
	return true;
}

// Fallback full b58enc prefix check
__device__ bool check_prefix_b58(uint8_t* pubkey, int prefix_len) {
	char key[256] = {0};
	size_t ks = 256;
	b58enc(key, &ks, pubkey, 32);
	for (int j = 0; j < prefix_len; j++) {
		if (prefixes[0][j] != '?' && prefixes[0][j] != key[j]) return false;
	}
	return true;
}

/* -- Batch tobytes: convert ge_p3 to bytes using Montgomery batch inversion - */
// Given N points in projective coords (X:Y:Z), compute affine (X/Z, Y/Z)
// using only 1 field inversion for all N points.
//
// Montgomery's trick:
//   products[0] = Z[0]
//   products[i] = products[i-1] * Z[i]   for i=1..N-1
//   inv = 1 / products[N-1]              (ONE inversion)
//   for i = N-1 down to 1:
//     inv_z[i] = inv * products[i-1]
//     inv = inv * Z[i]
//   inv_z[0] = inv
//
// Then X_affine[i] = X[i] * inv_z[i], Y_affine[i] = Y[i] * inv_z[i]

__device__ void batch_tobytes(ge_p3* points, uint8_t pubkeys[][32], int count) {
	if (count <= 0) return;

	fe products[BATCH_SIZE];
	fe inv_z[BATCH_SIZE];

	// Step 1: accumulate products of Z coordinates
	fe_copy(products[0], points[0].Z);
	for (int i = 1; i < count; i++) {
		fe_mul(products[i], products[i-1], points[i].Z);
	}

	// Step 2: single inversion
	fe inv;
	fe_invert(inv, products[count - 1]);

	// Step 3: compute individual inverses
	for (int i = count - 1; i > 0; i--) {
		fe_mul(inv_z[i], inv, products[i-1]);
		fe_mul(inv, inv, points[i].Z);
	}
	fe_copy(inv_z[0], inv);

	// Step 4: compute affine Y and encode to bytes
	// ed25519 tobytes: output = Y with sign bit from X
	for (int i = 0; i < count; i++) {
		fe recip, x, y;
		fe_copy(recip, inv_z[i]);
		fe_mul(x, points[i].X, recip);
		fe_mul(y, points[i].Y, recip);
		fe_tobytes(pubkeys[i], y);
		pubkeys[i][31] ^= fe_isnegative(x) << 7;
	}
}

/* -- Kernel ---------------------------------------------------------------- */
void __global__ vanity_batched(
	unsigned char* master_seed, unsigned long long int iteration,
	int* suffix_digits, int suffix_len, int prefix_len, bool is_suffix,
	int byte0_min, int byte0_max,
	int* keys_found, int* exec_count
) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	atomicAdd(exec_count, 1);

	// Derive starting scalar
	unsigned char scalar[32] = {0};
	{
		unsigned char input[48];
		for (int i = 0; i < 32; i++) input[i] = master_seed[i];
		unsigned long long int uid = (unsigned long long int)id;
		for (int i = 0; i < 8; i++) { input[32+i] = (unsigned char)(uid & 0xFF); uid >>= 8; }
		unsigned long long int uit = iteration;
		for (int i = 0; i < 8; i++) { input[40+i] = (unsigned char)(uit & 0xFF); uit >>= 8; }

		sha512_context md;
		md.curlen=0; md.length=0;
		md.state[0]=UINT64_C(0x6a09e667f3bcc908); md.state[1]=UINT64_C(0xbb67ae8584caa73b);
		md.state[2]=UINT64_C(0x3c6ef372fe94f82b); md.state[3]=UINT64_C(0xa54ff53a5f1d36f1);
		md.state[4]=UINT64_C(0x510e527fade682d1); md.state[5]=UINT64_C(0x9b05688c2b3e6c1f);
		md.state[6]=UINT64_C(0x1f83d9abfb41bd6b); md.state[7]=UINT64_C(0x5be0cd19137e2179);
		for (int i=0;i<48;i++) md.buf[i]=input[i];
		md.curlen=48; md.length+=md.curlen*UINT64_C(8);
		md.buf[md.curlen++]=0x80;
		while(md.curlen<120) md.buf[md.curlen++]=0;
		STORE64H(md.length, md.buf+120);
		uint64_t S[8],W[80],t0,t1;
		for(int i=0;i<8;i++) S[i]=md.state[i];
		for(int i=0;i<16;i++) LOAD64H(W[i],md.buf+(8*i));
		for(int i=16;i<80;i++) W[i]=Gamma1(W[i-2])+W[i-7]+Gamma0(W[i-15])+W[i-16];
		#define RND(a,b,c,d,e,f,g,h,i) t0=h+Sigma1(e)+Ch(e,f,g)+K[i]+W[i];t1=Sigma0(a)+Maj(a,b,c);d+=t0;h=t0+t1;
		for(int i=0;i<80;i+=8){RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);}
		#undef RND
		for(int i=0;i<8;i++) md.state[i]+=S[i];
		unsigned char h[64];
		for(int i=0;i<8;i++) STORE64H(md.state[i], h+(8*i));
		for(int i=0;i<32;i++) scalar[i]=h[i];
	}

	// Initial scalarmult (expensive, once)
	ge_p3 P;
	ge_scalarmult_base(&P, scalar);

	// Base point precomp for fast addition
	ge_precomp G_precomp = base[0][0];

	// Main loop: process in batches of BATCH_SIZE
	int total_attempts = ATTEMPTS_PER_EXECUTION;
	int batches = total_attempts / BATCH_SIZE;

	for (int b = 0; b < batches; b++) {
		// Step 1: generate BATCH_SIZE points via incremental addition
		ge_p3 batch_points[BATCH_SIZE];
		ge_p1p1 R;

		for (int i = 0; i < BATCH_SIZE; i++) {
			batch_points[i] = P;
			ge_madd(&R, &P, &G_precomp);
			ge_p1p1_to_p3(&P, &R);
		}

		// Step 2: batch convert to bytes (1 inversion for all BATCH_SIZE)
		uint8_t batch_pubkeys[BATCH_SIZE][32];
		batch_tobytes(batch_points, batch_pubkeys, BATCH_SIZE);

		// Step 3: check all BATCH_SIZE pubkeys
		for (int i = 0; i < BATCH_SIZE; i++) {
			bool matched = false;
			if (is_suffix && suffix_len > 0) {
				matched = check_suffix_mod(batch_pubkeys[i], suffix_digits, suffix_len);
			} else if (prefix_len > 0) {
				// FAST PREFIX: byte[0] pre-filter rejects 99.6%+ instantly
				if (batch_pubkeys[i][0] >= byte0_min && batch_pubkeys[i][0] <= byte0_max) {
					// Only ~0.4% reach here — do full b58enc
					matched = check_prefix_b58(batch_pubkeys[i], prefix_len);
				}
			}

			if (matched) {
				char key[256]={0}; size_t ks=256;
				b58enc(key, &ks, batch_pubkeys[i], 32);
				atomicAdd(keys_found, 1);
				printf("MATCH %s (thread=%d, batch=%d, idx=%d)\n", key, id, b, i);
				printf("[");
				for(int n=0;n<32;n++) printf("%d,",(unsigned char)scalar[n]);
				for(int n=0;n<32;n++){if(n+1==32)printf("%d",batch_pubkeys[i][n]);else printf("%d,",batch_pubkeys[i][n]);}
				printf("]\n");
			}
		}
	}
}

/* -- Host ------------------------------------------------------------------ */
void makeCsprngSeed(unsigned char* out) {
	std::random_device rd;
	for (int i = 0; i < 32; i++) out[i] = (unsigned char)(rd() & 0xFF);
}

std::string getTimeStr() {
	auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::string s(30, '\0');
	std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
	return s;
}

int main() {
	printf("BATCHED INCREMENTAL VANITY GENERATOR\n");
	fflush(stdout);

	// Parse pattern
	const char* test_patterns[] = VANITY_PATTERNS;
	const char* pattern = test_patterns[0];
	int full_len = 0; for (; pattern[full_len]; full_len++);

	int suffix_start = full_len;
	for (int j = full_len-1; j >= 0; j--) { if (pattern[j]=='?') break; suffix_start=j; }
	int suffix_len = full_len - suffix_start;

	int prefix_len = 0;
	for (int j = 0; j < full_len; j++) { if (pattern[j]=='?') break; prefix_len=j+1; }

	bool is_suffix = (suffix_len > 0 && prefix_len == 0);
	if (!is_suffix) suffix_len = 0;

	// Compute byte[0] pre-filter range for prefix mode
	int byte0_min = 0, byte0_max = 255;
	if (!is_suffix && prefix_len > 0) {
		// Compute the value range for this prefix in base58
		// For a 43-char b58 address, prefix value * 58^(43-prefix_len)
		// gives the number range. byte[0] = number >> 248.
		// Use double for approximate calculation (good enough for byte-level filter)
		double b58_val = 0;
		for (int i = 0; i < prefix_len; i++) {
			b58_val = b58_val * 58.0 + b58_digit(pattern[i]);
		}
		double pow58_rem = 1.0;
		for (int i = 0; i < 43 - prefix_len; i++) pow58_rem *= 58.0;
		double low = b58_val * pow58_rem;
		double high = (b58_val + 1.0) * pow58_rem;
		double pow2_248 = pow(2.0, 248);
		byte0_min = (int)(low / pow2_248);
		byte0_max = (int)(high / pow2_248);
		if (byte0_min < 0) byte0_min = 0;
		if (byte0_max > 255) byte0_max = 255;
		// Widen by 1 to handle rounding
		if (byte0_min > 0) byte0_min--;
		if (byte0_max < 255) byte0_max++;
		printf("PREFIX mode: \"%.*s\" (%d chars) [byte0 filter: %d-%d = %.1f%% pass]\n",
			prefix_len, pattern, prefix_len, byte0_min, byte0_max,
			(byte0_max - byte0_min + 1) / 256.0 * 100.0);
	} else if (is_suffix) {
		printf("SUFFIX mode: \"%s\" (%d chars) [modular check]\n", pattern+suffix_start, suffix_len);
	}
	printf("Batch size: %d (1 inversion per %d keys)\n", BATCH_SIZE, BATCH_SIZE);
	fflush(stdout);

	int host_digits[16]={0};
	if (is_suffix) for (int i=0;i<suffix_len&&i<16;i++) host_digits[i]=b58_digit(pattern[suffix_start+i]);

	int gpuCount=0; cudaGetDeviceCount(&gpuCount);
	unsigned char* dev_seeds[8]; int* dev_sd[8];

	for (int i=0;i<gpuCount;i++) {
		cudaSetDevice(i);
		cudaDeviceProp d; cudaGetDeviceProperties(&d,i);
		int blocks=4*d.multiProcessorCount;
		printf("GPU %d: %s — %d SMs, %d threads\n", i, d.name, d.multiProcessorCount, blocks*256);
		unsigned char seed[32]; makeCsprngSeed(seed);
		cudaMalloc((void**)&dev_seeds[i],32); cudaMemcpy(dev_seeds[i],seed,32,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_sd[i],16*sizeof(int)); cudaMemcpy(dev_sd[i],host_digits,16*sizeof(int),cudaMemcpyHostToDevice);
	}
	printf("Ready.\n\n"); fflush(stdout);

	unsigned long long int total=0; int found=0;
	for (int iter=0; iter<MAX_ITERATIONS; iter++) {
		auto start=std::chrono::high_resolution_clock::now();
		unsigned long long int ik=0;

		for (int g=0;g<gpuCount;g++) {
			cudaSetDevice(g);
			cudaDeviceProp d; cudaGetDeviceProperties(&d,g);
			int blocks=4*d.multiProcessorCount;
			int *dk,*de; cudaMalloc((void**)&dk,sizeof(int)); cudaMalloc((void**)&de,sizeof(int));
			int z=0; cudaMemcpy(dk,&z,sizeof(int),cudaMemcpyHostToDevice); cudaMemcpy(de,&z,sizeof(int),cudaMemcpyHostToDevice);
			vanity_batched<<<blocks,256>>>(dev_seeds[g],(unsigned long long)iter,dev_sd[g],suffix_len,prefix_len,is_suffix,byte0_min,byte0_max,dk,de);
			int kf=0; cudaDeviceSynchronize();
			cudaError_t err=cudaGetLastError();
			if(err!=cudaSuccess){printf("ERR: %s\n",cudaGetErrorString(err));fflush(stdout);exit(1);}
			cudaMemcpy(&kf,dk,sizeof(int),cudaMemcpyDeviceToHost);
			found+=kf;
			ik+=(unsigned long long)(blocks*256)*ATTEMPTS_PER_EXECUTION;
			cudaFree(dk); cudaFree(de);
		}
		total+=ik;
		auto finish=std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> el=finish-start;
		printf("%s Iter %d: %.0fM in %.1fs (%.0fM/s) — %.1fB total — %d found\n",
			getTimeStr().c_str(),iter+1,ik/1e6,el.count(),ik/el.count()/1e6,total/1e9,found);
		fflush(stdout);
		if(found>=STOP_AFTER_KEYS_FOUND){printf("Done!\n");fflush(stdout);exit(0);}
	}
	return 0;
}
