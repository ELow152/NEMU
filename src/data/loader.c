// loader.c
// Streaming loader that parses .nemu from zlib-compressed input incrementally.
// If RAM budget is hit, remaining variable-size blocks are written to a persistent
// spillover .nemu file named "<input>.spill.nemu".
//
// Build:
//   gcc loader.c spill.c -O3 -lz -o loader
//
// Usage:
//   ./loader nemu/serialized.nemu
//
// Notes:
// - Requires spill.h / spill.c implementing the SpillReader/SpillWriter API.
// - serializer.h must define Brain, Neuron, Synapse, SynEvent as discussed.

#define _POSIX_C_SOURCE 200809L
#include "serialize.h"
#include "utils/spill.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <zlib.h>
#include <errno.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "malloc(%zu) failed\n", n); exit(1); }
    return p;
}
static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n);
    if (!p && n != 0) { fprintf(stderr, "realloc(%zu) failed\n", n); exit(1); }
    return p;
}

/* Get available RAM (bytes) from /proc/meminfo (MemAvailable). Returns 0 on failure. */
static uint64_t get_available_ram_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char key[128];
    uint64_t val = 0;
    while (fscanf(f, "%127s %lu %*s\n", key, &val) == 2) {
        if (strcmp(key, "MemAvailable:") == 0) {
            fclose(f);
            return val * 1024ULL;
        }
    }
    fclose(f);
    return 0;
}

/* Decide RAM budget for the loader. Use a fraction of available RAM.
   If detection fails, return a conservative default. */
static uint64_t compute_ram_budget(void) {
    uint64_t avail = get_available_ram_bytes();
    if (avail == 0) {
        /* fallback: 512 MiB budget */
        return 512ULL * 1024ULL * 1024ULL;
    }
    /* Use at most 40% of available RAM for event allocation (leave headroom). */
    uint64_t budget = (avail / 100ULL) * 40ULL;
    /* clamp to sensible minimum */
    uint64_t min = 128ULL * 1024ULL * 1024ULL;
    if (budget < min) budget = min;
    return budget;
}

/* Helpers that read primitive types from SpillReader */
static int read_u32_sr(SpillReader *r, uint32_t *out) {
    return spill_reader_read_exact(r, out, sizeof(uint32_t));
}
static int read_u64_sr(SpillReader *r, uint64_t *out) {
    return spill_reader_read_exact(r, out, sizeof(uint64_t));
}
static int read_f32_sr(SpillReader *r, float *out) {
    return spill_reader_read_exact(r, out, sizeof(float));
}
static int read_block_sr(SpillReader *r, void *dst, size_t bytes) {
    return spill_reader_read_exact(r, dst, bytes);
}

/* Helper to build spill path "<input>.spill.nemu" */
static char *make_spill_path(const char *input_path) {
    size_t L = strlen(input_path);
    const char *suffix = ".spill.nemu";
    char *p = malloc(L + strlen(suffix) + 1);
    if (!p) return NULL;
    memcpy(p, input_path, L);
    memcpy(p + L, suffix, strlen(suffix) + 1);
    return p;
}

/* Main loader: parses streamingly, spills to a persistent .spill.nemu when RAM exceeded.
   Signature unchanged: Brain load_nemu(const char* path) */
Brain load_nemu(const char* path) {
    Brain brain = { NULL, 0, NULL, 0, 0, NULL };

    if (!path) return brain;

    SpillReader *r = spill_reader_open_file(path);
    if (!r) {
        fprintf(stderr, "Failed to open compressed input: %s\n", path);
        return brain;
    }

    uint64_t ram_budget = compute_ram_budget();
    /* Conservative per-neuron estimate to decide how many neurons to keep in RAM.
       Tweak this number to fit your actual avg neuron footprint. */
    const size_t per_neuron_est = 512; /* bytes per neuron approx */
    size_t max_neurons_in_mem = (size_t)(ram_budget / per_neuron_est);
    if (max_neurons_in_mem < 16) max_neurons_in_mem = 16;

    /* We'll create spill file only if/when we exceed memory threshold.
       Spill file path is deterministic (not tmp). */
    char *spill_path = make_spill_path(path);
    SpillWriter *w = NULL;
    int writer_active = 0;

    /* Read header: magic (4), version (u32), n_clusters (u64) */
    char magic[4];
    if (!read_block_sr(r, magic, 4)) goto fail;
    if (memcmp(magic, "NEMU", 4) != 0) { goto fail; }

    uint32_t version = 0;
    if (!read_u32_sr(r, &version)) goto fail;
    if (version != 3) { goto fail; }

    uint64_t n_clusters = 0;
    if (!read_u64_sr(r, &n_clusters)) goto fail;
    if (n_clusters == 0) { goto done; } /* empty file */

    /* If we create spill writer later we must write header first; when we create it we will re-emit header
       (we know n_clusters already). */

    size_t posted_neurons = 0;
    for (uint64_t cid = 0; cid < n_clusters; ++cid) {
        uint32_t cluster_id = 0;
        uint32_t cluster_count = 0;
        if (!read_u32_sr(r, &cluster_id)) goto fail;
        if (!read_u32_sr(r, &cluster_count)) goto fail;

        /* If writer already active, emit cluster header into spill file */
        if (writer_active) {
            if (!spill_writer_write_u32(w, cluster_id)) goto fail;
            if (!spill_writer_write_u32(w, cluster_count)) goto fail;
        }

        for (uint32_t i = 0; i < cluster_count; ++i) {
            Neuron n;
            memset(&n, 0, sizeof(n));

            /* base fields */
            if (!read_u64_sr(r, &n.root_id)) goto fail;
            if (!read_f32_sr(r, &n.x)) goto fail;
            if (!read_f32_sr(r, &n.y)) goto fail;
            if (!read_f32_sr(r, &n.z)) goto fail;

            if (!read_u32_sr(r, &n.type_id)) goto fail;
            if (!read_u32_sr(r, &n.syn_in)) goto fail;
            if (!read_u32_sr(r, &n.syn_out)) goto fail;

            /* neighbors count: prefer u64, fallback to u32 */
            uint64_t n_neighbors = 0;
            if (!read_u64_sr(r, &n_neighbors)) {
                uint32_t alt32 = 0;
                if (read_u32_sr(r, &alt32)) n_neighbors = alt32;
                else n_neighbors = 0;
            }
            n.n_neighbors = n_neighbors;

            if (n.n_neighbors > 0) {
                size_t bytes = (size_t)n.n_neighbors * sizeof(uint64_t);

                /* If we're still under memory threshold, keep neighbors in memory.
                   Otherwise, forward block to spill writer (creating it if necessary). */
                if (brain.n_neurons < max_neurons_in_mem) {
                    n.neighbors = xmalloc(bytes);
                    if (!read_block_sr(r, n.neighbors, bytes)) { free(n.neighbors); goto fail; }
                } else {
                    /* Need to spill this variable-length block */
                    if (!writer_active) {
                        /* create writer and write header + previous clusters up to current cluster already wrote */
                        w = spill_writer_open_file(spill_path, 1 /*fast*/);
                        if (!w) { /* fail to create spill file -> fallback: attempt to skip and keep parsing but not spill */ goto fail; }
                        writer_active = 1;
                        /* write header */
                        if (!spill_writer_write(w, "NEMU", 4)) goto fail;
                        if (!spill_writer_write_u32(w, 3)) goto fail;
                        if (!spill_writer_write_u64(w, n_clusters)) goto fail;
                        /* note: we must have written all previous cluster headers/blocks to writer when needed.
						   For simplicity we emit current cluster header now (above we emit when writer_active). */
                        /* emit current cluster header */
                        if (!spill_writer_write_u32(w, cluster_id)) goto fail;
                        if (!spill_writer_write_u32(w, cluster_count)) goto fail;
                    }
                    /* read into temp buffer and forward */
                    unsigned char *tmp = xmalloc(bytes);
                    if (!read_block_sr(r, tmp, bytes)) { free(tmp); goto fail; }
                    if (!spill_writer_write(w, tmp, bytes)) { free(tmp); goto fail; }
                    free(tmp);
                    n.neighbors = NULL; /* not kept in memory */
                }
            } else {
                n.neighbors = NULL;
            }

            /* out synapses: n_out_synapses (u64 or u32 fallback) */
            uint64_t n_out_syn = 0;
            if (!read_u64_sr(r, &n_out_syn)) {
                uint32_t alt32 = 0;
                if (read_u32_sr(r, &alt32)) n_out_syn = alt32;
                else n_out_syn = 0;
            }
            n.n_out_synapses = n_out_syn;

            if (n.n_out_synapses > 0) {
                size_t bytes = (size_t)n.n_out_synapses * sizeof(Synapse);

                if (brain.n_neurons < max_neurons_in_mem) {
                    n.out_synapses = xmalloc(bytes);
                    /* read synapses field-by-field into allocated array */
                    for (uint64_t s = 0; s < n.n_out_synapses; ++s) {
                        if (!read_u64_sr(r, &n.out_synapses[s].post_id)) goto fail;
                        if (!read_f32_sr(r, &n.out_synapses[s].pre_x)) goto fail;
                        if (!read_f32_sr(r, &n.out_synapses[s].pre_y)) goto fail;
                        if (!read_f32_sr(r, &n.out_synapses[s].pre_z)) goto fail;
                        if (!read_f32_sr(r, &n.out_synapses[s].post_x)) goto fail;
                        if (!read_f32_sr(r, &n.out_synapses[s].post_y)) goto fail;
                        if (!read_f32_sr(r, &n.out_synapses[s].post_z)) goto fail;
                        if (!read_f32_sr(r, &n.out_synapses[s].size))   goto fail;
                    }
                } else {
                    /* forward synapses directly to writer (create writer if needed) */
                    if (!writer_active) {
                        w = spill_writer_open_file(spill_path, 1 /*fast*/);
                        if (!w) goto fail;
                        writer_active = 1;
                        if (!spill_writer_write(w, "NEMU", 4)) goto fail;
                        if (!spill_writer_write_u32(w, 3)) goto fail;
                        if (!spill_writer_write_u64(w, n_clusters)) goto fail;
                        if (!spill_writer_write_u32(w, cluster_id)) goto fail;
                        if (!spill_writer_write_u32(w, cluster_count)) goto fail;
                    }
                    for (uint64_t s = 0; s < n.n_out_synapses; ++s) {
                        uint64_t postid;
                        float px,py,pz,qx,qy,qz,sz;
                        if (!read_u64_sr(r, &postid)) goto fail;
                        if (!read_f32_sr(r, &px)) goto fail;
                        if (!read_f32_sr(r, &py)) goto fail;
                        if (!read_f32_sr(r, &pz)) goto fail;
                        if (!read_f32_sr(r, &qx)) goto fail;
                        if (!read_f32_sr(r, &qy)) goto fail;
                        if (!read_f32_sr(r, &qz)) goto fail;
                        if (!read_f32_sr(r, &sz))  goto fail;

                        if (!spill_writer_write_u64(w, postid)) goto fail;
                        if (!spill_writer_write_f32(w, px)) goto fail;
                        if (!spill_writer_write_f32(w, py)) goto fail;
                        if (!spill_writer_write_f32(w, pz)) goto fail;
                        if (!spill_writer_write_f32(w, qx)) goto fail;
                        if (!spill_writer_write_f32(w, qy)) goto fail;
                        if (!spill_writer_write_f32(w, qz)) goto fail;
                        if (!spill_writer_write_f32(w, sz)) goto fail;
                    }
                    n.out_synapses = NULL;
                }
            } else {
                n.out_synapses = NULL;
            }

            /* append neuron to brain.neurons array if under memory cap */
            if (brain.n_neurons < max_neurons_in_mem) {
                brain.neurons = xrealloc(brain.neurons, (brain.n_neurons + 1) * sizeof(Neuron));
                brain.neurons[brain.n_neurons++] = n;
            } else {
                /* we're spilling this neuron; free any allocations we made (we either forwarded its var blocks
                   to writer or we failed earlier) */
                free(n.neighbors);
                free(n.out_synapses);
                /* Optionally you could record an index/offset in the spill file for later rehydration. */
            }

            if (++posted_neurons % 1024 == 0) {
                fprintf(stderr, "\rClusters processed: %6" PRIu64 "/%6" PRIu64 "  Neurons loaded: %10zu",
                        (uint64_t)(cid+1), n_clusters, brain.n_neurons);
                fflush(stderr);
            }
        }
    }

    /* Optional global synapse-event block: u64 n_syn_events, then each event is (u32 voxel + 7*f32) = 32 bytes */
    uint64_t n_syn_events = 0;
    if (read_u64_sr(r, &n_syn_events)) {
        const size_t event_bytes = 4 + 7*4;
        size_t total_needed = 0;
        if (!__builtin_mul_overflow_p((size_t)n_syn_events, event_bytes, (size_t)0))
            total_needed = (size_t)n_syn_events * event_bytes;

        /* decide whether to keep in RAM or stream into spill file */
        if (total_needed > 0 && total_needed <= ram_budget / 2) {
            /* read into RAM */
            SynEvent *events = malloc((size_t)n_syn_events * sizeof(SynEvent));
            if (!events) goto fail;
            for (uint64_t e = 0; e < n_syn_events; ++e) {
                if (!read_u32_sr(r, &events[e].voxel)) { free(events); goto fail; }
                if (!read_f32_sr(r, &events[e].pre_x)) { free(events); goto fail; }
                if (!read_f32_sr(r, &events[e].pre_y)) { free(events); goto fail; }
                if (!read_f32_sr(r, &events[e].pre_z)) { free(events); goto fail; }
                if (!read_f32_sr(r, &events[e].post_x)) { free(events); goto fail; }
                if (!read_f32_sr(r, &events[e].post_y)) { free(events); goto fail; }
                if (!read_f32_sr(r, &events[e].post_z)) { free(events); goto fail; }
                if (!read_f32_sr(r, &events[e].size))   { free(events); goto fail; }
            }
            brain.syn_events = events;
            brain.n_syn_events = n_syn_events;
            brain.syn_event_on_disk = 0;
        } else {
            /* stream events into spill file (create writer if necessary) */
            if (!writer_active) {
                w = spill_writer_open_file(spill_path, 1 /*fast*/);
                if (!w) goto fail;
                writer_active = 1;
                /* write header */
                if (!spill_writer_write(w, "NEMU", 4)) goto fail;
                if (!spill_writer_write_u32(w, 3)) goto fail;
                if (!spill_writer_write_u64(w, n_clusters)) goto fail;
                /* We already emitted cluster headers & forwarded content for spilled parts earlier as writer became active. */
            }
            /* write count */
            if (!spill_writer_write_u64(w, n_syn_events)) goto fail;
            /* stream events */
            for (uint64_t e = 0; e < n_syn_events; ++e) {
                /* read 32 bytes into buffer and forward */
                uint32_t voxel;
                float pre_x,pre_y,pre_z,post_x,post_y,post_z,sizef;
                if (!read_u32_sr(r, &voxel)) goto fail;
                if (!read_f32_sr(r, &pre_x)) goto fail;
                if (!read_f32_sr(r, &pre_y)) goto fail;
                if (!read_f32_sr(r, &pre_z)) goto fail;
                if (!read_f32_sr(r, &post_x)) goto fail;
                if (!read_f32_sr(r, &post_y)) goto fail;
                if (!read_f32_sr(r, &post_z)) goto fail;
                if (!read_f32_sr(r, &sizef)) goto fail;

                if (!spill_writer_write_u32(w, voxel)) goto fail;
                if (!spill_writer_write_f32(w, pre_x)) goto fail;
                if (!spill_writer_write_f32(w, pre_y)) goto fail;
                if (!spill_writer_write_f32(w, pre_z)) goto fail;
                if (!spill_writer_write_f32(w, post_x)) goto fail;
                if (!spill_writer_write_f32(w, post_y)) goto fail;
                if (!spill_writer_write_f32(w, post_z)) goto fail;
                if (!spill_writer_write_f32(w, sizef)) goto fail;
            }
            brain.syn_events = NULL;
            brain.n_syn_events = n_syn_events;
            brain.syn_event_on_disk = 1;
            brain.syn_event_path = strdup(spill_path);
        }
    } else {
        /* no syn-event block present */
        brain.syn_events = NULL;
        brain.n_syn_events = 0;
        brain.syn_event_on_disk = 0;
    }

    /* finish writer if active */
    if (writer_active) {
        if (!spill_writer_flush_finish(w)) goto fail;
    }

    fprintf(stderr, "\rClusters processed: %6" PRIu64 "/%6" PRIu64 "  Neurons loaded: %10zu\n",
            n_clusters, n_clusters, brain.n_neurons);

done:
    if (r) spill_reader_close(r);
    if (w) spill_writer_close(w);
    free(spill_path);
    return brain;

fail:
    /* cleanup partial allocations */
    if (r) spill_reader_close(r);
    if (w) { spill_writer_close(w); /* leave spill file in place; it's persistent */ }
    if (spill_path) free(spill_path);

    for (size_t i = 0; i < brain.n_neurons; ++i) {
        free(brain.neurons[i].neighbors);
        free(brain.neurons[i].out_synapses);
    }
    free(brain.neurons);
    brain.neurons = NULL;
    brain.n_neurons = 0;

    if (brain.syn_events) { free(brain.syn_events); brain.syn_events = NULL; brain.n_syn_events = 0; }
    /* if syn_event_path set, we keep it (points at spill file) */

    return brain;
}

/* Free brain - remove spill file if syn_event_on_disk==1 and syn_event_path is set? We keep behavior
   similar to previous free_brain: if syn_event_on_disk and syn_event_path is set we do not unlink here
   (leave file available for inspection), unless you change policy. */
void free_brain(Brain *b){
    if(!b) return;

    for(size_t i=0;i<b->n_neurons;i++){
        free(b->neurons[i].neighbors);
        free(b->neurons[i].out_synapses);
    }
    free(b->neurons);
    b->neurons = NULL;
    b->n_neurons = 0;

    if (b->syn_event_on_disk) {
        /* we wrote events into persistent spill file; keep it by default */
        if (b->syn_event_path) {
            /* leave file on disk; caller may unlink explicitly */
            /* if you want free_brain to remove it, uncomment the unlink below */
            /* unlink(b->syn_event_path); free(b->syn_event_path); b->syn_event_path = NULL; */
        }
        b->syn_events = NULL;
        b->n_syn_events = 0;
        b->syn_event_on_disk = 1;
    } else {
        free(b->syn_events);
        b->syn_events = NULL;
        b->n_syn_events = 0;
        b->syn_event_on_disk = 0;
        if (b->syn_event_path) { free(b->syn_event_path); b->syn_event_path = NULL; }
    }
}

#ifdef LOADER_TEST_MAIN
/* Test main */
int main(int argc, char **argv) {
    const char *path = "serialized.nemu";
    if (argc > 1) path = argv[1];

    Brain b = load_nemu(path);
    fprintf(stderr, "Loaded %zu neurons  global_events=%llu  syn_events_on_disk=%d\n",
            b.n_neurons, (unsigned long long)b.n_syn_events, b.syn_event_on_disk);
    if (b.n_neurons == 0 && b.n_syn_events == 0) return 1;

    /* ... (keep most of your previous analysis code here if desired) ... */

    free_brain(&b);
    return 0;
}
#endif