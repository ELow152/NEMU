// serializer.h
#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <stdint.h>
#include <stddef.h>

typedef struct Synapse {
    uint64_t post_id;
    float pre_x, pre_y, pre_z;
    float post_x, post_y, post_z;
    float size;
} Synapse;

typedef struct Neuron {
    uint64_t root_id;
    float x, y, z;
    uint32_t type_id;
    uint32_t syn_in;
    uint32_t syn_out;

    uint64_t n_neighbors;
    uint64_t *neighbors;

    uint64_t n_out_synapses;
    Synapse *out_synapses;
} Neuron;

typedef struct Cluster {
    uint32_t id;
    uint32_t n_neurons;
    Neuron *neurons;
} Cluster;

void serialize_nemu(const char *path, Cluster *clusters, size_t n_clusters);

#endif