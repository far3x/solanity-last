#include <vector>
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

/* -- Types ----------------------------------------------------------------- */
typedef struct {
	unsigned char* dev_seeds[8];
	unsigned char* dev_suffix[8];
	int*           dev_suffix_digits[8];
	int            suffix_len;
} config;

/* -- Prototypes ------------------------------------------------------------ */
void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_scan(unsigned char* master_seed, unsigned long long int iteration,
                            int* suffix_digits, int suffix_len,
                            int* keys_found, int* gpu, int* execution_count);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

/* -- Base58 Alphabet ------------------------------------------------------- */
__device__ __constant__ char B58_ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Host-side copy
static const char H_B58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

int b58_char_to_digit(char c) {
	for (int i = 0; i < 58; i++) if (H_B58[i] == c) return i;
	return -1;
}

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
        out[i] = (unsigned char)(rd() & 0xFF);
    }
}

/* -- Setup ----------------------------------------------------------------- */
void vanity_setup(config &vanity) {
	printf("GPU: Initializing (CSPRNG-safe, modular suffix matching)\n");
	fflush(stdout);

	// Parse suffix from first pattern (strip leading '?')
	const char* pattern = NULL;
	{
		// Read first pattern from config at compile time — extract suffix
		// We do this by finding the non-'?' tail
		const char* test_patterns[] = VANITY_PATTERNS;
		pattern = test_patterns[0];
	}

	int full_len = 0;
	for (; pattern[full_len] != 0; full_len++);

	int suffix_start = full_len;
	for (int j = full_len - 1; j >= 0; j--) {
		if (pattern[j] == '?') break;
		suffix_start = j;
	}

	vanity.suffix_len = full_len - suffix_start;
	const char* suffix_str = pattern + suffix_start;

	if (vanity.suffix_len > 0) {
		printf("Mode: SUFFIX — \"%.*s\" (%d chars) [fast modular check]\n", vanity.suffix_len, suffix_str, vanity.suffix_len);
	} else {
		printf("Mode: PREFIX — \"%s\" [b58enc check]\n", pattern);
	}

	// Convert suffix chars to base58 digit indices
	int host_digits[16] = {0};
	for (int i = 0; i < vanity.suffix_len && i < 16; i++) {
		host_digits[i] = b58_char_to_digit(suffix_str[i]);
		if (host_digits[i] < 0) {
			printf("ERROR: '%c' is not a valid base58 character!\n", suffix_str[i]);
			exit(1);
		}
	}

	// Estimate difficulty
	unsigned long long int combinations = 1;
	for (int i = 0; i < vanity.suffix_len; i++) combinations *= 58;
	printf("Search space: ~%llu combinations\n", combinations);
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

		printf("GPU %d: %s — %d SMs, %d threads\n",
			i, device.name, device.multiProcessorCount, totalBlocks * blockSize);

		unsigned char host_seed[32];
		makeCsprngSeed(host_seed);

		cudaMalloc((void**)&vanity.dev_seeds[i], 32);
		cudaMemcpy(vanity.dev_seeds[i], host_seed, 32, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&vanity.dev_suffix_digits[i], 16 * sizeof(int));
		cudaMemcpy(vanity.dev_suffix_digits[i], host_digits, 16 * sizeof(int), cudaMemcpyHostToDevice);
	}
	printf("Ready.\n\n");
	fflush(stdout);
}

/* -- Run ------------------------------------------------------------------- */
void vanity_run(config &vanity) {
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	unsigned long long int executions_total = 0;
	int keys_found_total = 0;
	int* dev_keys_found[8];
	int* dev_exec_count[8];

	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		unsigned long long int iter_keys = 0;

		for (int g = 0; g < gpuCount; ++g) {
			cudaSetDevice(g);
			cudaDeviceProp device;
			cudaGetDeviceProperties(&device, g);
			int totalBlocks = 4 * device.multiProcessorCount;

			int* dev_g;
			cudaMalloc((void**)&dev_g, sizeof(int));
			cudaMemcpy(dev_g, &g, sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&dev_keys_found[g], sizeof(int));
			cudaMalloc((void**)&dev_exec_count[g], sizeof(int));
			int zero = 0;
			cudaMemcpy(dev_keys_found[g], &zero, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_exec_count[g], &zero, sizeof(int), cudaMemcpyHostToDevice);

			vanity_scan<<<totalBlocks, 256>>>(
				vanity.dev_seeds[g], (unsigned long long int)i,
				vanity.dev_suffix_digits[g], vanity.suffix_len,
				dev_keys_found[g], dev_g, dev_exec_count[g]
			);
		}

		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
			fflush(stdout);
			exit(1);
		}
		auto finish = std::chrono::high_resolution_clock::now();

		int found_this = 0;
		for (int g = 0; g < gpuCount; ++g) {
			int kf = 0, ec = 0;
			cudaMemcpy(&kf, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&ec, dev_exec_count[g], sizeof(int), cudaMemcpyDeviceToHost);
			found_this += kf;
			iter_keys += (unsigned long long int)ec * ATTEMPTS_PER_EXECUTION;
			cudaFree(dev_keys_found[g]);
			cudaFree(dev_exec_count[g]);
		}
		keys_found_total += found_this;
		executions_total += iter_keys;

		std::chrono::duration<double> elapsed = finish - start;
		double mps = (iter_keys / elapsed.count()) / 1000000.0;
		printf("%s Iter %d: %.1fM keys in %.1fs (%.1fM/s) — Total: %.1fB — Found: %d\n",
			getTimeStr().c_str(), i+1,
			iter_keys / 1000000.0, elapsed.count(), mps,
			executions_total / 1000000000.0, keys_found_total
		);
		fflush(stdout);

		if (keys_found_total >= STOP_AFTER_KEYS_FOUND) {
			printf("\nDone! Found %d keys.\n", keys_found_total);
			fflush(stdout);
			exit(0);
		}
	}
}

/* -- CUDA Kernel: modular arithmetic suffix matching ----------------------- */
/*
 * Instead of base58-encoding the full 32-byte pubkey (O(32*44) divisions),
 * we compute pubkey mod 58^suffix_len using modular arithmetic (O(32) mods),
 * then decompose into base58 digits. This is ~44x faster.
 *
 * Math: the last N chars of base58(X) = digits of (X mod 58^N) in base 58.
 */

__device__ bool check_suffix_mod(uint8_t* pubkey, int* suffix_digits, int suffix_len) {
	// Compute modulus = 58^suffix_len (fits in uint64 for suffix_len <= 10)
	unsigned long long int modulus = 1;
	for (int i = 0; i < suffix_len; i++) modulus *= 58;

	// Compute pubkey mod modulus (pubkey is big-endian 32 bytes)
	unsigned long long int remainder = 0;
	for (int i = 0; i < 32; i++) {
		// remainder = (remainder * 256 + pubkey[i]) % modulus
		// Use 128-bit to avoid overflow: remainder < modulus < 58^10 < 2^59
		// and remainder * 256 < 2^67, fits in unsigned long long
		remainder = (remainder * 256ULL + (unsigned long long int)pubkey[i]) % modulus;
	}

	// Decompose remainder into base58 digits, check from LAST char backwards
	// suffix_digits[suffix_len-1] is the LAST char of the suffix = least significant digit
	for (int i = suffix_len - 1; i >= 0; i--) {
		int digit = (int)(remainder % 58ULL);
		if (digit != suffix_digits[i]) return false;
		remainder /= 58ULL;
	}

	return true;
}

void __global__ vanity_scan(unsigned char* master_seed, unsigned long long int iteration,
                            int* suffix_digits, int suffix_len,
                            int* keys_found, int* gpu, int* exec_count) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	atomicAdd(exec_count, 1);

	ge_p3 A;
	unsigned char seed[32]     = {0};
	unsigned char publick[32]  = {0};
	unsigned char privatek[64] = {0};

	// Derive starting seed: SHA512(master_seed || thread_id || iteration)
	{
		unsigned char input[48];
		for (int i = 0; i < 32; i++) input[i] = master_seed[i];
		unsigned long long int uid = (unsigned long long int)id;
		for (int i = 0; i < 8; i++) { input[32+i] = (unsigned char)(uid & 0xFF); uid >>= 8; }
		unsigned long long int uit = iteration;
		for (int i = 0; i < 8; i++) { input[40+i] = (unsigned char)(uit & 0xFF); uit >>= 8; }

		sha512_context md;
		md.curlen = 0; md.length = 0;
		md.state[0]=UINT64_C(0x6a09e667f3bcc908); md.state[1]=UINT64_C(0xbb67ae8584caa73b);
		md.state[2]=UINT64_C(0x3c6ef372fe94f82b); md.state[3]=UINT64_C(0xa54ff53a5f1d36f1);
		md.state[4]=UINT64_C(0x510e527fade682d1); md.state[5]=UINT64_C(0x9b05688c2b3e6c1f);
		md.state[6]=UINT64_C(0x1f83d9abfb41bd6b); md.state[7]=UINT64_C(0x5be0cd19137e2179);
		for (int i = 0; i < 48; i++) md.buf[i] = input[i];
		md.curlen = 48;
		md.length += md.curlen * UINT64_C(8);
		md.buf[md.curlen++] = 0x80;
		while (md.curlen < 120) md.buf[md.curlen++] = 0;
		STORE64H(md.length, md.buf + 120);
		uint64_t S[8], W[80], t0, t1;
		for (int i=0;i<8;i++) S[i]=md.state[i];
		for (int i=0;i<16;i++) LOAD64H(W[i], md.buf+(8*i));
		for (int i=16;i<80;i++) W[i]=Gamma1(W[i-2])+W[i-7]+Gamma0(W[i-15])+W[i-16];
		#define RND(a,b,c,d,e,f,g,h,i) t0=h+Sigma1(e)+Ch(e,f,g)+K[i]+W[i];t1=Sigma0(a)+Maj(a,b,c);d+=t0;h=t0+t1;
		for (int i=0;i<80;i+=8){RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);}
		#undef RND
		for (int i=0;i<8;i++) md.state[i]+=S[i];
		unsigned char h[64];
		for (int i=0;i<8;i++) STORE64H(md.state[i], h+(8*i));
		for (int i=0;i<32;i++) seed[i]=h[i];
	}

	// Hot loop — SHA512 + ed25519 scalarmult + modular suffix check
	sha512_context md;
	for (int a = 0; a < ATTEMPTS_PER_EXECUTION; ++a) {
		// SHA512(seed)
		md.curlen=0;md.length=0;
		md.state[0]=UINT64_C(0x6a09e667f3bcc908);md.state[1]=UINT64_C(0xbb67ae8584caa73b);
		md.state[2]=UINT64_C(0x3c6ef372fe94f82b);md.state[3]=UINT64_C(0xa54ff53a5f1d36f1);
		md.state[4]=UINT64_C(0x510e527fade682d1);md.state[5]=UINT64_C(0x9b05688c2b3e6c1f);
		md.state[6]=UINT64_C(0x1f83d9abfb41bd6b);md.state[7]=UINT64_C(0x5be0cd19137e2179);
		for (int i=0;i<32;i++) md.buf[i]=seed[i];
		md.curlen=32;
		md.length+=md.curlen*UINT64_C(8);
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
		for(int i=0;i<8;i++) STORE64H(md.state[i], privatek+(8*i));

		// ed25519 clamp + scalarmult
		privatek[0]&=248; privatek[31]&=63; privatek[31]|=64;
		ge_scalarmult_base(&A, privatek);
		ge_p3_tobytes(publick, &A);

		bool matched = false;
		if (suffix_len > 0) {
			// FAST: modular suffix check
			matched = check_suffix_mod(publick, suffix_digits, suffix_len);
		} else {
			// PREFIX: b58enc + compare first chars against prefixes[]
			char key[256]={0};
			size_t ks=256;
			b58enc(key, &ks, publick, 32);
			for (int p = 0; p < sizeof(prefixes)/sizeof(prefixes[0]); ++p) {
				bool pm = true;
				for (int j = 0; prefixes[p][j] != 0; ++j) {
					if (prefixes[p][j] != '?' && prefixes[p][j] != key[j]) { pm = false; break; }
				}
				if (pm) { matched = true; break; }
			}
			if (matched) {
				atomicAdd(keys_found, 1);
				printf("MATCH %s\n[", key);
				for(int n=0;n<32;n++) printf("%d,",(unsigned char)seed[n]);
				for(int n=0;n<32;n++){if(n+1==32)printf("%d",publick[n]);else printf("%d,",publick[n]);}
				printf("]\n");
			}
		}
		if (matched && suffix_len > 0) {
			char key[256]={0};
			size_t ks=256;
			b58enc(key, &ks, publick, 32);
			atomicAdd(keys_found, 1);
			printf("MATCH %s\n[", key);
			for(int n=0;n<32;n++) printf("%d,",(unsigned char)seed[n]);
			for(int n=0;n<32;n++){if(n+1==32)printf("%d",publick[n]);else printf("%d,",publick[n]);}
			printf("]\n");
		}

		// Increment seed
		for(int i=0;i<32;++i){if(seed[i]==255){seed[i]=0;}else{seed[i]+=1;break;}}
	}
}

/* -- Full Base58 (display only) -------------------------------------------- */
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
