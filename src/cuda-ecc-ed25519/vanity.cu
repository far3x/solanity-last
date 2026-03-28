#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
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
	unsigned char* dev_seeds[8]; // CSPRNG seeds per GPU
} config;

/* -- Prototypes ------------------------------------------------------------ */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_scan(unsigned char* master_seed, unsigned long long int iteration, int* keys_found, int* gpu, int* execution_count);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

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

// Generate 32 bytes of cryptographically secure randomness from OS
void makeCsprngSeed(unsigned char* out) {
    std::random_device rd;
    for (int i = 0; i < 32; i++) {
        // random_device uses CryptGenRandom on Windows, /dev/urandom on Linux
        auto r = rd();
        out[i] = (unsigned char)(r & 0xFF);
    }
}

/* -- Setup ----------------------------------------------------------------- */

void vanity_setup(config &vanity) {
	printf("GPU: Initializing (CSPRNG-safe mode)\n");
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	for (int i = 0; i < gpuCount; ++i) {
		cudaSetDevice(i);

		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, i);

		// Hardcode for maximum occupancy — occupancy API fails on Blackwell JIT
		int blockSize = 256;
		int blocksPerSM = 4;
		int totalBlocks = blocksPerSM * device.multiProcessorCount;

		printf("GPU: %d (%s) -- %d SMs x %d blocks x %d threads = %d total threads\n",
			i, device.name, device.multiProcessorCount, blocksPerSM, blockSize, totalBlocks * blockSize);

		// Generate 32 bytes of CSPRNG seed on host
		unsigned char host_seed[32];
		makeCsprngSeed(host_seed);

		printf("CSPRNG seed (first 8 bytes): ");
		for (int j = 0; j < 8; j++) printf("%02x", host_seed[j]);
		printf("...\n");

		// Copy seed to device
		cudaMalloc((void**)&vanity.dev_seeds[i], 32);
		cudaMemcpy(vanity.dev_seeds[i], host_seed, 32, cudaMemcpyHostToDevice);
	}

	printf("END: Initializing\n");
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
				vanity.dev_seeds[g],
				(unsigned long long int)i,
				dev_keys_found[g],
				dev_g,
				dev_executions_this_gpu[g]
			);
		}

		cudaDeviceSynchronize();
		auto finish = std::chrono::high_resolution_clock::now();

		for (int g = 0; g < gpuCount; ++g) {
			cudaMemcpy(&keys_found_this_iteration, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost);
			keys_found_total += keys_found_this_iteration;

			cudaMemcpy(&executions_this_gpu, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost);
			executions_this_iteration += executions_this_gpu * ATTEMPTS_PER_EXECUTION;
			executions_total += executions_this_gpu * ATTEMPTS_PER_EXECUTION;
		}

		std::chrono::duration<double> elapsed = finish - start;
		printf("%s Iter %d: %llu attempts in %.2fs (%.0f/s) - Total: %llu - Found: %d\n",
			getTimeStr().c_str(), i+1,
			executions_this_iteration, elapsed.count(),
			executions_this_iteration / elapsed.count(),
			executions_total, keys_found_total
		);

		if (keys_found_total >= STOP_AFTER_KEYS_FOUND) {
			printf("Done!\n");
			exit(0);
		}
	}

	printf("Iterations complete.\n");
}

/* -- CUDA Kernel ----------------------------------------------------------- */

// Safe key derivation: SHA512(master_seed || thread_id || iteration || counter)
// This is deterministic given the inputs, but the master_seed is CSPRNG,
// so an attacker cannot reproduce keys without knowing the seed.
void __global__ vanity_scan(unsigned char* master_seed, unsigned long long int iteration, int* keys_found, int* gpu, int* exec_count) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	atomicAdd(exec_count, 1);

	// Build prefix match table
	int prefix_letter_counts[MAX_PATTERNS];
	for (unsigned int n = 0; n < sizeof(prefixes) / sizeof(prefixes[0]); ++n) {
		if (MAX_PATTERNS == n) return;
		int letter_count = 0;
		for (; prefixes[n][letter_count] != 0; letter_count++);
		prefix_letter_counts[n] = letter_count;
	}

	// Local state
	ge_p3 A;
	unsigned char seed[32]     = {0};
	unsigned char publick[32]  = {0};
	unsigned char privatek[64] = {0};
	char key[256]              = {0};

	// Derive initial seed: SHA512(master_seed || id || iteration || 0)
	// The first 32 bytes of the hash become our candidate ed25519 seed.
	// This is safe because master_seed is from CSPRNG.
	{
		// Build 48-byte input: master_seed(32) + id(8) + iteration(8)
		unsigned char derive_input[48];
		for (int i = 0; i < 32; i++) derive_input[i] = master_seed[i];

		// Encode thread id (8 bytes LE)
		unsigned long long int uid = (unsigned long long int)id;
		for (int i = 0; i < 8; i++) {
			derive_input[32 + i] = (unsigned char)(uid & 0xFF);
			uid >>= 8;
		}

		// Encode iteration (8 bytes LE)
		unsigned long long int uiter = iteration;
		for (int i = 0; i < 8; i++) {
			derive_input[40 + i] = (unsigned char)(uiter & 0xFF);
			uiter >>= 8;
		}

		// SHA512 the 48 bytes to get a 64-byte hash, take first 32 as seed
		sha512_context dmd;
		dmd.curlen = 0;
		dmd.length = 0;
		dmd.state[0] = UINT64_C(0x6a09e667f3bcc908);
		dmd.state[1] = UINT64_C(0xbb67ae8584caa73b);
		dmd.state[2] = UINT64_C(0x3c6ef372fe94f82b);
		dmd.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
		dmd.state[4] = UINT64_C(0x510e527fade682d1);
		dmd.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
		dmd.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
		dmd.state[7] = UINT64_C(0x5be0cd19137e2179);

		for (int i = 0; i < 48; i++) dmd.buf[i] = derive_input[i];
		dmd.curlen = 48;

		// sha512_final
		dmd.length += dmd.curlen * UINT64_C(8);
		dmd.buf[dmd.curlen++] = 0x80;
		while (dmd.curlen < 120) dmd.buf[dmd.curlen++] = 0;
		STORE64H(dmd.length, dmd.buf + 120);

		uint64_t S[8], W[80], t0, t1;
		for (int i = 0; i < 8; i++) S[i] = dmd.state[i];
		for (int i = 0; i < 16; i++) LOAD64H(W[i], dmd.buf + (8*i));
		for (int i = 16; i < 80; i++) W[i] = Gamma1(W[i-2]) + W[i-7] + Gamma0(W[i-15]) + W[i-16];

		#define RND(a,b,c,d,e,f,g,h,i) \
		t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
		t1 = Sigma0(a) + Maj(a, b, c); d += t0; h = t0 + t1;

		for (int i = 0; i < 80; i += 8) {
			RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);
			RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
			RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);
			RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
			RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);
			RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
			RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);
			RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
		}
		#undef RND

		for (int i = 0; i < 8; i++) dmd.state[i] += S[i];

		unsigned char hash_out[64];
		for (int i = 0; i < 8; i++) STORE64H(dmd.state[i], hash_out + (8*i));

		// Use first 32 bytes as our starting seed
		for (int i = 0; i < 32; i++) seed[i] = hash_out[i];
	}

	// Now scan: for each attempt, derive keypair from seed, check vanity, increment
	sha512_context md;

	for (int attempts = 0; attempts < ATTEMPTS_PER_EXECUTION; ++attempts) {
		// SHA512(seed) -> privatek
		md.curlen = 0;
		md.length = 0;
		md.state[0] = UINT64_C(0x6a09e667f3bcc908);
		md.state[1] = UINT64_C(0xbb67ae8584caa73b);
		md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
		md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
		md.state[4] = UINT64_C(0x510e527fade682d1);
		md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
		md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
		md.state[7] = UINT64_C(0x5be0cd19137e2179);

		const unsigned char *in = seed;
		for (size_t i = 0; i < 32; i++) md.buf[i + md.curlen] = in[i];
		md.curlen += 32;

		md.length += md.curlen * UINT64_C(8);
		md.buf[md.curlen++] = 0x80;
		while (md.curlen < 120) md.buf[md.curlen++] = 0;
		STORE64H(md.length, md.buf + 120);

		uint64_t S[8], W[80], t0, t1;
		for (int i = 0; i < 8; i++) S[i] = md.state[i];
		for (int i = 0; i < 16; i++) LOAD64H(W[i], md.buf + (8*i));
		for (int i = 16; i < 80; i++) W[i] = Gamma1(W[i-2]) + W[i-7] + Gamma0(W[i-15]) + W[i-16];

		#define RND(a,b,c,d,e,f,g,h,i) \
		t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
		t1 = Sigma0(a) + Maj(a, b, c); d += t0; h = t0 + t1;

		for (int i = 0; i < 80; i += 8) {
			RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);
			RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
			RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);
			RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
			RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);
			RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
			RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);
			RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
		}
		#undef RND

		for (int i = 0; i < 8; i++) md.state[i] += S[i];
		for (int i = 0; i < 8; i++) STORE64H(md.state[i], privatek + (8*i));

		// ed25519 clamping
		privatek[0]  &= 248;
		privatek[31] &= 63;
		privatek[31] |= 64;

		// Curve multiplication -> public key
		ge_scalarmult_base(&A, privatek);
		ge_p3_tobytes(publick, &A);

		// Base58 encode public key
		size_t keysize = 256;
		b58enc(key, &keysize, publick, 32);

		// Check all patterns
		for (int i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i) {
			for (int j = 0; j < prefix_letter_counts[i]; ++j) {
				if (!(prefixes[i][j] == '?') && !(prefixes[i][j] == key[j])) break;

				if (j == (prefix_letter_counts[i] - 1)) {
					atomicAdd(keys_found, 1);

					// Output: address + keypair as JSON array
					printf("GPU %d MATCH %s - ", *gpu, key);
					for (int n = 0; n < sizeof(seed); n++) printf("%02x", (unsigned char)seed[n]);
					printf("\n");

					printf("[");
					for (int n = 0; n < sizeof(seed); n++) printf("%d,", (unsigned char)seed[n]);
					for (int n = 0; n < sizeof(publick); n++) {
						if (n + 1 == sizeof(publick)) printf("%d", publick[n]);
						else printf("%d,", publick[n]);
					}
					printf("]\n");
					break;
				}
			}
		}

		// Increment seed (counter-based, safe because initial seed is CSPRNG-derived)
		for (int i = 0; i < 32; ++i) {
			if (seed[i] == 255) {
				seed[i] = 0;
			} else {
				seed[i] += 1;
				break;
			}
		}
	}
}

/* -- Base58 Encoder -------------------------------------------------------- */

bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz) {
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
	const uint8_t *bin = data;
	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;

	while (zcount < binsz && !bin[zcount]) ++zcount;
	size = (binsz - zcount) * 138 / 100 + 1;
	uint8_t buf[256];
	memset(buf, 0, size);

	for (i = zcount, high = size - 1; i < binsz; ++i, high = j) {
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j) {
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j) break;
		}
	}

	for (j = 0; j < size && !buf[j]; ++j);

	if (*b58sz <= zcount + size - j) {
		*b58sz = zcount + size - j + 1;
		return false;
	}

	if (zcount) memset(b58, '1', zcount);
	for (i = zcount; j < size; ++i, ++j) b58[i] = b58digits_ordered[buf[j]];
	b58[i] = '\0';
	*b58sz = i + 1;
	return true;
}
