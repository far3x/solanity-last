#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 100000;
static int const STOP_AFTER_KEYS_FOUND = 1;

__device__ const int ATTEMPTS_PER_EXECUTION = 100000;
__device__ const int MAX_PATTERNS = 10;

// Use ? as wildcard. Solana addresses are 43-44 chars in base58.
// For suffix matching "pump", pad with ? to 44 chars total.
__device__ static char const *prefixes[] = {
	"????????????????????????????????????????pump",
};

#endif
