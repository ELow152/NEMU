/*
This is the backend for storage of anything that has to do with Brain.

*/

#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>

typedef struct Synapse {
    uint64_t post_id;
    float pre_x, pre_y, pre_z;
    float post_x, post_y, post_z;
    float size;
} Synapse;

typedef struct SynEvent {
    uint32_t voxel;   // index into voxel grid, UINT32_MAX if none
    float pre_x, pre_y, pre_z;
    float post_x, post_y, post_z;
    float size;
} SynEvent;

typedef struct Neuron {
    uint64_t root_id;
    uint32_t type_id;
    float x, y, z;
    uint32_t syn_in, syn_out;
    uint64_t n_neighbors;
    uint64_t *neighbors;
    /* no per-neuron synapse storage in this design */
    uint64_t n_out_synapses;    /* left 0 (we don't assign synapses to neurons) */
    uint64_t synapse_offset;    /* UINT64_MAX */
    Synapse *out_synapses;
} Neuron;

typedef struct Cluster {
    uint32_t id;
    uint64_t n_neurons;
    Neuron *neurons;
} Cluster;

/* ---------- Simple hash table for Neuron* keyed by root_id ---------- */
typedef struct Node {
    uint64_t key;
    Neuron *value;
    struct Node *next;
} Node;

typedef struct {
    Node **buckets;
} NeuronTable;

static inline uint64_t hash_u64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x % HASH_SIZE;
}

