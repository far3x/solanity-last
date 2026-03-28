#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

#include "fixedint.h"
#include "gpu_common.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

/* -- Types ----------------------------------------------------------------- */
typedef struct {
	unsigned char* dev_seeds[8];
} config;

/* -- Prototypes ------------------------------------------------------------ */
void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_scan(unsigned char* master_seed, unsigned long long int iteration,
                            int* keys_found, int* gpu, int* execution_count);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);
bool __device__ b58enc_tail(char* out, int out_len, uint8_t* data, size_t binsz);

/* -- Entry Point ----------------------------------------------------------- */
int main(int argc, char const* argv[]) {
	config vanity;
	vanity_setup(vanity);
	vanity_run(vanity);
}

std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

void makeCsprngSeed(unsigned char* out) {
    std::random_device rd;
    for (int i = 0; i < 32; i++) {
        auto r = rd();
        out[i] = (unsigned char)(r & 0xFF);
    }
}

/* -- Setup ----------------------------------------------------------------- */
void vanity_setup(config &vanity) {
	printf("GPU: Initializing (CSPRNG-safe mode)\n");
	fflush(stdout);
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	for (int i = 0; i < gpuCount; ++i) {
		cudaSetDevice(i);
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, i);

		int blockSize = 256;
		int blocksPerSM = 4;
		int totalBlocks = blocksPerSM * device.multiProcessorCount;

		printf("GPU: %d (%s) -- %d SMs x %d blocks x %d threads = %d total\n",
			i, device.name, device.multiProcessorCount, blocksPerSM, blockSize, totalBlocks * blockSize);

		unsigned char host_seed[32];
		makeCsprngSeed(host_seed);
		printf("CSPRNG seed: ");
		for (int j = 0; j < 8; j++) printf("%02x", host_seed[j]);
		printf("...\n");
		fflush(stdout);

		cudaMalloc((void**)&vanity.dev_seeds[i], 32);
		cudaMemcpy(vanity.dev_seeds[i], host_seed, 32, cudaMemcpyHostToDevice);
	}
	printf("Ready.\n");
	fflush(stdout);
}

/* -- Run ------------------------------------------------------------------- */
void vanity_run(config &vanity) {
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	unsigned long long int executions_total = 0;
	unsigned long long int executions_this_iteration;
	int executions_this_gpu;
	int* dev_executions_this_gpu[100];
	int keys_found_total = 0;
	int keys_found_this_iteration;
	int* dev_keys_found[100];

	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		executions_this_iteration = 0;

		for (int g = 0; g < gpuCount; ++g) {
			cudaSetDevice(g);
			cudaDeviceProp device;
			cudaGetDeviceProperties(&device, g);
			int blockSize = 256;
			int totalBlocks = 4 * device.multiProcessorCount;

			int* dev_g;
			cudaMalloc((void**)&dev_g, sizeof(int));
			cudaMemcpy(dev_g, &g, sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&dev_keys_found[g], sizeof(int));
			cudaMalloc((void**)&dev_executions_this_gpu[g], sizeof(int));

			vanity_scan<<<totalBlocks, blockSize>>>(
				vanity.dev_seeds[g], (unsigned long long int)i,
				dev_keys_found[g], dev_g, dev_executions_this_gpu[g]
			);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("LAUNCH ERROR: %s\n", cudaGetErrorString(err));
				fflush(stdout);
				exit(1);
			}
		}

		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("KERNEL ERROR: %s\n", cudaGetErrorString(err));
			fflush(stdout);
			exit(1);
		}
		auto finish = std::chrono::high_resolution_clock::now();

		for (int g = 0; g < gpuCount; ++g) {
			cudaMemcpy(&keys_found_this_iteration, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost);
			keys_found_total += keys_found_this_iteration;
			cudaMemcpy(&executions_this_gpu, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost);
			executions_this_iteration += (unsigned long long int)executions_this_gpu * ATTEMPTS_PER_EXECUTION;
			executions_total += (unsigned long long int)executions_this_gpu * ATTEMPTS_PER_EXECUTION;
		}

		std::chrono::duration<double> elapsed = finish - start;
		printf("%s Iter %d: %llu keys in %.1fs (%0.1fM/s) - Total: %llu - Found: %d\n",
			getTimeStr().c_str(), i+1,
			executions_this_iteration, elapsed.count(),
			(executions_this_iteration / elapsed.count()) / 1000000.0,
			executions_total, keys_found_total
		);
		fflush(stdout);

		if (keys_found_total >= STOP_AFTER_KEYS_FOUND) {
			printf("Done!\n");
			fflush(stdout);
			exit(0);
		}
	}
	printf("Iterations complete.\n");
	fflush(stdout);
}

/* -- CUDA Kernel: optimized with fast suffix pre-filter -------------------- */

// Fast partial base58 of last N bytes — enough to check suffix
// Returns the last `out_len` base58 characters
bool __device__ b58enc_tail(char* out, int out_len, uint8_t* data, size_t binsz) {
	const char b58digits[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
	// full b58 encode but only keep the tail
	uint8_t buf[64];
	size_t size = (binsz) * 138 / 100 + 1;
	if (size > 64) size = 64;
	memset(buf, 0, size);

	for (size_t i = 0; i < binsz; ++i) {
		int carry = data[i];
		for (size_t j = size - 1; ; --j) {
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (j == 0) break;
		}
	}

	// find first non-zero
	size_t start = 0;
	while (start < size && buf[start] == 0) start++;

	size_t total_chars = size - start;
	if ((int)total_chars < out_len) return false;

	// copy last out_len chars
	for (int i = 0; i < out_len; i++) {
		out[i] = b58digits[buf[size - out_len + i]];
	}
	return true;
}

void __global__ vanity_scan(unsigned char* master_seed, unsigned long long int iteration,
                            int* keys_found, int* gpu, int* exec_count) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	atomicAdd(exec_count, 1);

	// Compute suffix to match and its length
	// We match the LAST characters of the address
	int suffix_len = 0;
	char suffix[16];
	{
		// Find the non-wildcard suffix from the first pattern
		int full_len = 0;
		for (; prefixes[0][full_len] != 0; full_len++);
		// scan backwards from end to find where wildcards stop
		int suffix_start = full_len;
		for (int j = full_len - 1; j >= 0; j--) {
			if (prefixes[0][j] == '?') break;
			suffix_start = j;
		}
		suffix_len = full_len - suffix_start;
		for (int j = 0; j < suffix_len && j < 16; j++) {
			suffix[j] = prefixes[0][suffix_start + j];
		}
	}

	// Local state
	ge_p3 A;
	unsigned char seed[32]     = {0};
	unsigned char publick[32]  = {0};
	unsigned char privatek[64] = {0};

	// Derive initial seed from CSPRNG master + thread id + iteration
	{
		unsigned char derive_input[48];
		for (int i = 0; i < 32; i++) derive_input[i] = master_seed[i];
		unsigned long long int uid = (unsigned long long int)id;
		for (int i = 0; i < 8; i++) { derive_input[32+i] = (unsigned char)(uid & 0xFF); uid >>= 8; }
		unsigned long long int uiter = iteration;
		for (int i = 0; i < 8; i++) { derive_input[40+i] = (unsigned char)(uiter & 0xFF); uiter >>= 8; }

		sha512_context dmd;
		dmd.curlen = 0; dmd.length = 0;
		dmd.state[0] = UINT64_C(0x6a09e667f3bcc908); dmd.state[1] = UINT64_C(0xbb67ae8584caa73b);
		dmd.state[2] = UINT64_C(0x3c6ef372fe94f82b); dmd.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
		dmd.state[4] = UINT64_C(0x510e527fade682d1); dmd.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
		dmd.state[6] = UINT64_C(0x1f83d9abfb41bd6b); dmd.state[7] = UINT64_C(0x5be0cd19137e2179);
		for (int i = 0; i < 48; i++) dmd.buf[i] = derive_input[i];
		dmd.curlen = 48;
		dmd.length += dmd.curlen * UINT64_C(8);
		dmd.buf[dmd.curlen++] = 0x80;
		while (dmd.curlen < 120) dmd.buf[dmd.curlen++] = 0;
		STORE64H(dmd.length, dmd.buf + 120);

		uint64_t S[8], W[80], t0, t1;
		for (int i = 0; i < 8; i++) S[i] = dmd.state[i];
		for (int i = 0; i < 16; i++) LOAD64H(W[i], dmd.buf + (8*i));
		for (int i = 16; i < 80; i++) W[i] = Gamma1(W[i-2]) + W[i-7] + Gamma0(W[i-15]) + W[i-16];
		#define RND(a,b,c,d,e,f,g,h,i) t0=h+Sigma1(e)+Ch(e,f,g)+K[i]+W[i]; t1=Sigma0(a)+Maj(a,b,c); d+=t0; h=t0+t1;
		for (int i = 0; i < 80; i += 8) {
			RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0); RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
			RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2); RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
			RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4); RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
			RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6); RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
		}
		#undef RND
		for (int i = 0; i < 8; i++) dmd.state[i] += S[i];
		unsigned char hash_out[64];
		for (int i = 0; i < 8; i++) STORE64H(dmd.state[i], hash_out + (8*i));
		for (int i = 0; i < 32; i++) seed[i] = hash_out[i];
	}

	sha512_context md;

	for (int attempts = 0; attempts < ATTEMPTS_PER_EXECUTION; ++attempts) {
		// SHA512(seed) -> privatek
		md.curlen = 0; md.length = 0;
		md.state[0] = UINT64_C(0x6a09e667f3bcc908); md.state[1] = UINT64_C(0xbb67ae8584caa73b);
		md.state[2] = UINT64_C(0x3c6ef372fe94f82b); md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
		md.state[4] = UINT64_C(0x510e527fade682d1); md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
		md.state[6] = UINT64_C(0x1f83d9abfb41bd6b); md.state[7] = UINT64_C(0x5be0cd19137e2179);
		for (size_t i = 0; i < 32; i++) md.buf[i] = seed[i];
		md.curlen = 32;
		md.length += md.curlen * UINT64_C(8);
		md.buf[md.curlen++] = 0x80;
		while (md.curlen < 120) md.buf[md.curlen++] = 0;
		STORE64H(md.length, md.buf + 120);

		uint64_t S[8], W[80], t0, t1;
		for (int i = 0; i < 8; i++) S[i] = md.state[i];
		for (int i = 0; i < 16; i++) LOAD64H(W[i], md.buf + (8*i));
		for (int i = 16; i < 80; i++) W[i] = Gamma1(W[i-2]) + W[i-7] + Gamma0(W[i-15]) + W[i-16];
		#define RND(a,b,c,d,e,f,g,h,i) t0=h+Sigma1(e)+Ch(e,f,g)+K[i]+W[i]; t1=Sigma0(a)+Maj(a,b,c); d+=t0; h=t0+t1;
		for (int i = 0; i < 80; i += 8) {
			RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0); RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
			RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2); RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
			RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4); RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
			RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6); RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
		}
		#undef RND
		for (int i = 0; i < 8; i++) md.state[i] += S[i];
		for (int i = 0; i < 8; i++) STORE64H(md.state[i], privatek + (8*i));

		privatek[0]  &= 248;
		privatek[31] &= 63;
		privatek[31] |= 64;

		ge_scalarmult_base(&A, privatek);
		ge_p3_tobytes(publick, &A);

		// FAST PATH: only encode the suffix portion to check
		char tail[16];
		if (suffix_len > 0 && b58enc_tail(tail, suffix_len, publick, 32)) {
			bool match = true;
			for (int j = 0; j < suffix_len; j++) {
				if (tail[j] != suffix[j]) { match = false; break; }
			}
			if (match) {
				// Full encode only on match for display
				char key[256] = {0};
				size_t keysize = 256;
				b58enc(key, &keysize, publick, 32);

				atomicAdd(keys_found, 1);
				printf("GPU %d MATCH %s\n[", *gpu, key);
				for (int n = 0; n < 32; n++) printf("%d,", (unsigned char)seed[n]);
				for (int n = 0; n < 32; n++) {
					if (n + 1 == 32) printf("%d", publick[n]);
					else printf("%d,", publick[n]);
				}
				printf("]\n");
			}
		}

		// Increment seed
		for (int i = 0; i < 32; ++i) {
			if (seed[i] == 255) { seed[i] = 0; }
			else { seed[i] += 1; break; }
		}
	}
}

/* -- Full Base58 Encoder (only used for display on match) ------------------ */
bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz) {
	const char b58digits[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
	const uint8_t *bin = data;
	int carry;
	size_t i, j, high, zcount = 0;
	while (zcount < binsz && !bin[zcount]) ++zcount;
	size_t size = (binsz - zcount) * 138 / 100 + 1;
	uint8_t buf[256];
	memset(buf, 0, size);
	for (i = zcount, high = size - 1; i < binsz; ++i, high = j) {
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j) {
			carry += 256 * buf[j]; buf[j] = carry % 58; carry /= 58;
			if (!j) break;
		}
	}
	for (j = 0; j < size && !buf[j]; ++j);
	if (*b58sz <= zcount + size - j) { *b58sz = zcount + size - j + 1; return false; }
	if (zcount) memset(b58, '1', zcount);
	for (i = zcount; j < size; ++i, ++j) b58[i] = b58digits[buf[j]];
	b58[i] = '\0';
	*b58sz = i + 1;
	return true;
}
