#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 100000;
static int const STOP_AFTER_KEYS_FOUND = 1;

__device__ const int ATTEMPTS_PER_EXECUTION = 2000;
__device__ const int MAX_PATTERNS = 10;

// Suffix matching: use ? for wildcards, suffix at the end
// The kernel auto-detects the non-? tail as the suffix to match
#define VANITY_PATTERNS { "????????????????????????????????????????rugged" }

__device__ static char const *prefixes[] = VANITY_PATTERNS;

#endif
