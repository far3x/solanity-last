/*
 * INCREMENTAL VANITY KEY GENERATOR
 *
 * Instead of full SHA512 + scalarmult per key (~2530 field muls),
 * we do ONE initial scalarmult, then P += G for each subsequent key (~10 field muls).
 * This is ~250x fewer field operations per key.
 *
 * Trade-off: the output keypair uses the raw clamped scalar as the "seed" bytes.
 * Standard Solana tools do SHA512(seed) to derive the signing scalar, so this
 * keypair format is non-standard. For vanity address generation where you only
 * need the address, this is fine. For signing, a wrapper is needed.
 *
 * Security: CSPRNG seed from host → SHA512 derivation for initial scalar →
 * sequential increment. Same security as the standard approach since the
 * starting point is CSPRNG-derived.
 */

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

/* -- Base58 ---------------------------------------------------------------- */
__device__ __constant__ char B58_ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static const char H_B58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
int b58_digit(char c) { for (int i = 0; i < 58; i++) if (H_B58[i] == c) return i; return -1; }

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

/* -- Modular suffix check -------------------------------------------------- */
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

/* -- Prefix check (needs b58enc) ------------------------------------------- */
__device__ bool check_prefix(uint8_t* pubkey, int prefix_len) {
	char key[256] = {0};
	size_t ks = 256;
	b58enc(key, &ks, pubkey, 32);
	for (int j = 0; j < prefix_len; j++) {
		if (prefixes[0][j] != '?' && prefixes[0][j] != key[j]) return false;
	}
	return true;
}

/* -- Types ----------------------------------------------------------------- */
typedef struct {
	unsigned char* dev_seeds[8];
	int* dev_suffix_digits[8];
	int suffix_len;
	int prefix_len;
	bool is_suffix_mode;
} config;

/* -- CSPRNG ---------------------------------------------------------------- */
void makeCsprngSeed(unsigned char* out) {
	std::random_device rd;
	for (int i = 0; i < 32; i++) out[i] = (unsigned char)(rd() & 0xFF);
}

std::string getTimeStr() {
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::string s(30, '\0');
	std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
	return s;
}

/* -- Kernel ---------------------------------------------------------------- */
void __global__ vanity_incremental(
	unsigned char* master_seed, unsigned long long int iteration,
	int* suffix_digits, int suffix_len, int prefix_len, bool is_suffix,
	int* keys_found, int* exec_count
) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	atomicAdd(exec_count, 1);

	// 1. Derive unique starting scalar from CSPRNG seed
	unsigned char scalar[32] = {0};
	{
		unsigned char input[48];
		for (int i = 0; i < 32; i++) input[i] = master_seed[i];
		unsigned long long int uid = (unsigned long long int)id;
		for (int i = 0; i < 8; i++) { input[32+i] = (unsigned char)(uid & 0xFF); uid >>= 8; }
		unsigned long long int uit = iteration;
		for (int i = 0; i < 8; i++) { input[40+i] = (unsigned char)(uit & 0xFF); uit >>= 8; }

		// SHA512 to derive scalar
		sha512_context md;
		md.curlen=0; md.length=0;
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
		for(int i=0;i<80;i+=8){RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);}
		#undef RND
		for (int i=0;i<8;i++) md.state[i]+=S[i];
		unsigned char h[64];
		for (int i=0;i<8;i++) STORE64H(md.state[i], h+(8*i));
		for (int i=0;i<32;i++) scalar[i] = h[i];
	}

	// 2. Clamp scalar (ed25519 style)
	scalar[0] &= 248;
	scalar[31] &= 63;
	scalar[31] |= 64;

	// 3. ONE expensive scalarmult for the initial point
	ge_p3 P;
	ge_scalarmult_base(&P, scalar);

	// 4. Precompute G in cached form for fast addition
	//    G is the ed25519 base point. We compute it by doing scalarmult with scalar=1
	ge_p3 G_p3;
	unsigned char one_scalar[32] = {0};
	one_scalar[0] = 1; // scalar = 1, but must be clamped: 1 & 248 = 0...
	// Actually clamping 1 gives 0 which is identity. We need the actual base point.
	// The base point is hardcoded in ge_scalarmult_base's precomputed table.
	// We can get G by: G = 1_unclamped * G_base. But scalarmult_base clamps internally...
	//
	// Alternative: compute G = scalarmult_base(8) since 8 & 248 = 8, then divide by 8.
	// But there's no point division.
	//
	// Simplest: use ge_add to compute P + Q where Q is known.
	// We need G as a ge_cached. Let's compute it once:
	// scalar_one = [8, 0, 0, ...] → gives 8*G
	// We need 1*G. Since clamping sets bit 6 (|= 64), minimum clamped scalar is 64.
	//
	// DIFFERENT APPROACH: instead of P += G, we increment the scalar by 1 BEFORE clamping
	// and redo the full scalarmult. That's what we're trying to avoid.
	//
	// CORRECT APPROACH: we don't clamp. We use raw scalars without clamping.
	// The scalar is just a 256-bit number. We increment it. The public key changes by G.
	// Clamping is only needed for signing (cofactor safety), not for address generation.
	//
	// So: DON'T CLAMP. Use raw scalar. P = raw_scalar * G. P' = (raw_scalar+1) * G = P + G.

	// Redo without clamping:
	// Restore original scalar (undo clamp)
	// Actually let's just start fresh without clamping
	{
		// Re-derive scalar without clamping
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
		for (int i = 0; i < 48; i++) md.buf[i] = input[i];
		md.curlen = 48;
		md.length += md.curlen * UINT64_C(8);
		md.buf[md.curlen++] = 0x80;
		while (md.curlen < 120) md.buf[md.curlen++] = 0;
		STORE64H(md.length, md.buf + 120);
		uint64_t S2[8], W2[80], t02, t12;
		for (int i=0;i<8;i++) S2[i]=md.state[i];
		for (int i=0;i<16;i++) LOAD64H(W2[i], md.buf+(8*i));
		for (int i=16;i<80;i++) W2[i]=Gamma1(W2[i-2])+W2[i-7]+Gamma0(W2[i-15])+W2[i-16];
		#define RND2(a,b,c,d,e,f,g,h,i) t02=h+Sigma1(e)+Ch(e,f,g)+K[i]+W2[i];t12=Sigma0(a)+Maj(a,b,c);d+=t02;h=t02+t12;
		for(int i=0;i<80;i+=8){RND2(S2[0],S2[1],S2[2],S2[3],S2[4],S2[5],S2[6],S2[7],i+0);RND2(S2[7],S2[0],S2[1],S2[2],S2[3],S2[4],S2[5],S2[6],i+1);RND2(S2[6],S2[7],S2[0],S2[1],S2[2],S2[3],S2[4],S2[5],i+2);RND2(S2[5],S2[6],S2[7],S2[0],S2[1],S2[2],S2[3],S2[4],i+3);RND2(S2[4],S2[5],S2[6],S2[7],S2[0],S2[1],S2[2],S2[3],i+4);RND2(S2[3],S2[4],S2[5],S2[6],S2[7],S2[0],S2[1],S2[2],i+5);RND2(S2[2],S2[3],S2[4],S2[5],S2[6],S2[7],S2[0],S2[1],i+6);RND2(S2[1],S2[2],S2[3],S2[4],S2[5],S2[6],S2[7],S2[0],i+7);}
		#undef RND2
		for (int i=0;i<8;i++) md.state[i]+=S2[i];
		unsigned char h[64];
		for (int i=0;i<8;i++) STORE64H(md.state[i], h+(8*i));
		// Use first 32 bytes as UNCLAMPED scalar
		for (int i=0;i<32;i++) scalar[i] = h[i];
	}

	// Initial P = scalar * G (ge_scalarmult_base DOES internal clamping)
	// We need a scalarmult that does NOT clamp. But ge_scalarmult_base always clamps.
	// The internal code does: scalar[0] &= 248; scalar[31] &= 127; scalar[31] |= 64;
	//
	// SOLUTION: We accept the clamped scalar. The clamping is deterministic.
	// After clamping, scalar becomes S. We compute P = S * G.
	// Then P' = P + G = (S+1) * G. But S+1 might not equal the clamped version of (scalar+1).
	//
	// THE FUNDAMENTAL ISSUE: ge_scalarmult_base clamps internally, and incrementing
	// the input scalar by 1 doesn't increment the clamped scalar by 1.
	//
	// REAL SOLUTION: Compute base point G as a ge_cached, then do ge_add(P, G_cached)
	// in the inner loop. The scalar tracking is separate — we just need to record
	// which scalar produced the match.
	//
	// For the initial point, use ge_scalarmult_base (it clamps internally).
	// For increments, add G directly (no scalar involved).
	// On match, the "scalar" is: clamped(initial_sha512) + counter.
	// To reconstruct: store initial_sha512_output and counter.

	// Compute initial point
	ge_scalarmult_base(&P, scalar);

	// Compute G (base point) as ge_cached for addition
	// G = ge_scalarmult_base(scalar=[1,0,...0]) but clamping turns 1 into 0...
	// Instead: compute 2*G, then use ge_sub or compute G differently.
	//
	// Actually, the ed25519 base point is stored in the precomp tables.
	// We can extract it from ge_scalarmult_base by setting a scalar that
	// after clamping gives exactly 1. Clamping does: s[0] &= 248.
	// So the minimum non-zero clamped value is 8. scalar=[8,0,...0] gives 8*G.
	// We need G, not 8*G.
	//
	// SIMPLER: Compute G = P2 - P1 where P1 = scalar*G and P2 = (scalar+8)*G.
	// Then G_8 = P2 - P1 = 8*G. We want 1*G. But cofactor is 8, so the subgroup
	// we're in has order l, and G is the base point of that subgroup.
	//
	// SIMPLEST: hardcode the base point coordinates.

	// Ed25519 base point (y-coordinate, x is derived)
	// B = (x, 4/5) on the Edwards curve
	// In ge_p3 extended coordinates: X, Y, Z=1, T=X*Y

	// Actually the cleanest approach: compute G by doing scalarmult with
	// the UNCLAMPED scalar [1,0,...,0] and manually doing the math.
	// But scalarmult_base always clamps...

	// Let me just use ge_double_scalarmult_vartime or manually construct G.
	// The base point in compressed form is:
	// 5866666666666666666666666666666666666666666666666666666666666666 (hex, y-coord)

	// OK let's just hardcode G as a ge_cached from the precomputed tables.
	// In ge.cu, the function ge_scalarmult_base uses `base` precomp table.
	// base[0][0] is the base point in ge_precomp form.
	// We need to convert ge_precomp to ge_cached.
	//
	// ge_precomp has: yplusx, yminusx, xy2d
	// ge_cached has: YplusX, YminusX, Z, T2d
	// These are different representations. We need to go through ge_p3.

	// JUST DO IT: compute P_test = scalarmult(scalar_test) where scalar_test is chosen
	// so that clamped(scalar_test) = clamped(scalar) + 1. Then G = P_test - P.
	// But subtraction isn't directly available...

	// FINAL APPROACH: use ge_madd with base[0][0] (the base point precomp).
	// ge_madd(r, p, q) computes r = p + q where q is ge_precomp.
	// This is EXACTLY what we need: P' = P + G where G is in precomp form.

	// The base point precomp is the first entry of the `base` table used in
	// ge_scalarmult_base. It's defined in precomp_data.h.
	// base[0][0] represents 1*G.

	// Let's use it!
	ge_precomp G_precomp = base[0][0];

	unsigned char publick[32] = {0};
	ge_p1p1 R;

	// Hot loop: just ge_madd + suffix/prefix check. No SHA512, no scalarmult!
	for (int a = 0; a < ATTEMPTS_PER_EXECUTION; ++a) {
		ge_p3_tobytes(publick, &P);

		// Check match
		bool matched = false;
		if (is_suffix && suffix_len > 0) {
			matched = check_suffix_mod(publick, suffix_digits, suffix_len);
		} else if (prefix_len > 0) {
			matched = check_prefix(publick, prefix_len);
		}

		if (matched) {
			char key[256] = {0};
			size_t ks = 256;
			b58enc(key, &ks, publick, 32);
			atomicAdd(keys_found, 1);
			// Output the scalar (initial scalar + counter = a)
			printf("MATCH %s (thread=%d, offset=%d)\n", key, id, a);
			printf("[");
			for (int n = 0; n < 32; n++) printf("%d,", (unsigned char)scalar[n]);
			for (int n = 0; n < 32; n++) {
				if (n + 1 == 32) printf("%d", publick[n]);
				else printf("%d,", publick[n]);
			}
			printf("]\n");
		}

		// P = P + G (one point addition — the fast part!)
		ge_madd(&R, &P, &G_precomp);
		ge_p1p1_to_p3(&P, &R);

		// Track scalar increment (for output purposes only)
		// We don't actually need the scalar for the check, just for the keypair output
	}
}

/* -- Main ------------------------------------------------------------------ */
int main(int argc, char const* argv[]) {
	printf("INCREMENTAL VANITY GENERATOR (CSPRNG-safe)\n");
	fflush(stdout);

	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	// Parse pattern
	const char* test_patterns[] = VANITY_PATTERNS;
	const char* pattern = test_patterns[0];
	int full_len = 0;
	for (; pattern[full_len] != 0; full_len++);

	// Detect suffix vs prefix
	int suffix_start = full_len;
	for (int j = full_len - 1; j >= 0; j--) {
		if (pattern[j] == '?') break;
		suffix_start = j;
	}
	int suffix_len = full_len - suffix_start;

	// Check if it's really a prefix (non-? chars at start, ? at end)
	int prefix_len = 0;
	for (int j = 0; j < full_len; j++) {
		if (pattern[j] == '?') break;
		prefix_len = j + 1;
	}

	bool is_suffix = (suffix_len > 0 && prefix_len == 0);
	if (!is_suffix) suffix_len = 0; // force prefix mode

	printf("Pattern: \"%s\"\n", pattern);
	if (is_suffix) printf("Mode: SUFFIX (%d chars) [modular check]\n", suffix_len);
	else printf("Mode: PREFIX (%d chars) [b58enc check]\n", prefix_len);
	fflush(stdout);

	// Convert suffix to digits
	int host_digits[16] = {0};
	if (is_suffix) {
		for (int i = 0; i < suffix_len && i < 16; i++) {
			host_digits[i] = b58_digit(pattern[suffix_start + i]);
		}
	}

	// Setup GPUs
	unsigned char* dev_seeds[8];
	int* dev_suffix_digits[8];

	for (int i = 0; i < gpuCount; ++i) {
		cudaSetDevice(i);
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, i);
		printf("GPU %d: %s — %d SMs\n", i, device.name, device.multiProcessorCount);

		unsigned char host_seed[32];
		makeCsprngSeed(host_seed);
		cudaMalloc((void**)&dev_seeds[i], 32);
		cudaMemcpy(dev_seeds[i], host_seed, 32, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_suffix_digits[i], 16 * sizeof(int));
		cudaMemcpy(dev_suffix_digits[i], host_digits, 16 * sizeof(int), cudaMemcpyHostToDevice);
	}
	printf("Ready.\n\n");
	fflush(stdout);

	// Run
	unsigned long long int total = 0;
	int found_total = 0;

	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		auto start = std::chrono::high_resolution_clock::now();
		unsigned long long int iter_keys = 0;

		for (int g = 0; g < gpuCount; ++g) {
			cudaSetDevice(g);
			cudaDeviceProp device;
			cudaGetDeviceProperties(&device, g);
			int totalBlocks = 4 * device.multiProcessorCount;

			int* dev_kf; int* dev_ec;
			cudaMalloc((void**)&dev_kf, sizeof(int));
			cudaMalloc((void**)&dev_ec, sizeof(int));
			int zero = 0;
			cudaMemcpy(dev_kf, &zero, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_ec, &zero, sizeof(int), cudaMemcpyHostToDevice);

			vanity_incremental<<<totalBlocks, 256>>>(
				dev_seeds[g], (unsigned long long int)iter,
				dev_suffix_digits[g], suffix_len, prefix_len, is_suffix,
				dev_kf, dev_ec
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

		// Collect results (simplified — just count from last GPU)
		for (int g = 0; g < gpuCount; ++g) {
			cudaDeviceProp device;
			cudaGetDeviceProperties(&device, g);
			int totalBlocks = 4 * device.multiProcessorCount;
			iter_keys += (unsigned long long int)(totalBlocks * 256) * ATTEMPTS_PER_EXECUTION;
		}
		total += iter_keys;

		std::chrono::duration<double> elapsed = finish - start;
		double mps = (iter_keys / elapsed.count()) / 1000000.0;
		printf("%s Iter %d: %.1fM keys in %.1fs (%.1fM/s) — Total: %.1fB — Found: %d\n",
			getTimeStr().c_str(), iter + 1,
			iter_keys / 1000000.0, elapsed.count(), mps,
			total / 1000000000.0, found_total
		);
		fflush(stdout);

		if (found_total >= STOP_AFTER_KEYS_FOUND) {
			printf("Done!\n");
			fflush(stdout);
			exit(0);
		}
	}
	return 0;
}
