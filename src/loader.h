// loader.h
#ifndef LOADER_H
#define LOADER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-neuron serialized synapse record (same layout as in serialize.c) */
typedef struct Synapse {
    uint64_t post_id;
    float pre_x, pre_y, pre_z;
    float post_x, post_y, post_z;
    float size;
} Synapse;

/* Global synapse event format used by newer serializer (voxel + coords + size).
   Provided so callers can inspect or stream global events without attaching
   them to specific neurons in-memory. */
typedef struct SynEvent {
    uint32_t voxel;     /* UINT32_MAX if none */
    float pre_x, pre_y, pre_z;
    float post_x, post_y, post_z;
    float size;
} SynEvent;

typedef struct Neuron {
    uint64_t root_id;
    float x, y, z;
    uint32_t type_id;
    uint32_t syn_in, syn_out;

    uint64_t n_neighbors;
    uint64_t *neighbors;

    /* optional per-neuron out-synapse list (may be NULL if file used global events) */
    uint64_t n_out_synapses;
    Synapse *out_synapses;
} Neuron;

/* Brain returned by load_nemu().
   - neurons / n_neurons: always populated (if present in file)
   - syn_events / n_syn_events: populated only if the loader kept events in RAM
   - syn_event_on_disk: if non-zero, events were streamed to disk and syn_event_path
                        points to the temporary file containing SynEvent records.
   - syn_event_path is heap-allocated (strdup) when syn_event_on_disk != 0 and must
     be freed by free_brain().
*/
typedef struct Brain {
    Neuron *neurons;
    size_t n_neurons;

    /* optional global synapse events */
    SynEvent *syn_events;    /* NULL if events are not stored in RAM */
    uint64_t n_syn_events;   /* number of events read (may be truncated if file shorter) */

    /* if syn_events were spilled to disk */
    int syn_event_on_disk;   /* 0 => in-RAM or none, 1 => events stored in syn_event_path */
    char *syn_event_path;    /* path to temp file when syn_event_on_disk==1; NULL otherwise */
} Brain;

/* Load a serialized nemu file produced by serialize.c.
   The returned Brain must be freed with free_brain(). */
Brain load_nemu(const char *path);

/* Free resources held by Brain. If syn_event_on_disk==1 this will unlink the
   temporary syn_event file and free syn_event_path. */
void free_brain(Brain *b);

#ifdef __cplusplus
}
#endif

#endif /* LOADER_H */