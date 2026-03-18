// loader.c
// Robust, tolerant loader for the zlib-deflated format produced by serialize_nemu.
// Memory-aware: checks system RAM and spills large synapse-event blocks to disk.
//
// Build:
//   gcc loader.c -O3 -lz -o loader
//
// Usage:
//   ./loader nemu/serialized.nemu

#define _POSIX_C_SOURCE 200809L
#include "loader.h"

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
static Neuron* append_neuron(Neuron* array, size_t* count, Neuron n) {
    array = xrealloc(array, (*count + 1) * sizeof(Neuron));
    array[*count] = n;
    (*count)++;
    return array;
}

/* safe check for available bytes */
static int ensure_space(size_t offset, size_t total, size_t need) {
    if (need > total) return 0; /* avoid overflow */
    return offset + need <= total;
}

/* read helpers (assume little-endian host) */
static uint32_t read_u32_le(const unsigned char *buf, size_t *off, size_t len) {
    if (!ensure_space(*off, len, 4)) { fprintf(stderr, "read_u32_le: truncated\n"); exit(1); }
    uint32_t v; memcpy(&v, buf + *off, 4); *off += 4; return v;
}
static uint64_t read_u64_le(const unsigned char *buf, size_t *off, size_t len) {
    if (!ensure_space(*off, len, 8)) { fprintf(stderr, "read_u64_le: truncated\n"); exit(1); }
    uint64_t v; memcpy(&v, buf + *off, 8); *off += 8; return v;
}
static float read_f32_le(const unsigned char *buf, size_t *off, size_t len) {
    if (!ensure_space(*off, len, 4)) { fprintf(stderr, "read_f32_le: truncated\n"); exit(1); }
    float v; memcpy(&v, buf + *off, 4); *off += 4; return v;
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

/* tolerant parsing: prefer 64-bit counts, fall back to 32-bit,
   but do NOT abort on "absurd" values - attempt reinterpretation
   and if still doesn't fit, clamp to remaining buffer where sensible.
   Additionally: if the global syn-event block is too large for RAM budget,
   stream it to a temp file instead of allocating memory.
*/
static Brain parse_decompressed(const unsigned char *buf, size_t buf_len) {
    Brain brain = { NULL, 0, NULL, 0, 0, NULL };
    size_t off = 0;

    if (!ensure_space(off, buf_len, 4 + 4 + 8)) {
        fprintf(stderr, "buffer too small for header\n");
        return brain;
    }
    if (memcmp(buf + off, "NEMU", 4) != 0) {
        fprintf(stderr, "invalid magic\n");
        return brain;
    }
    off += 4;

    uint32_t version = read_u32_le(buf, &off, buf_len);
    if (version != 3) {
        fprintf(stderr, "unsupported version %u\n", version);
        return brain;
    }

    uint64_t n_clusters = read_u64_le(buf, &off, buf_len);
    if (n_clusters == 0) {
        fprintf(stderr, "zero clusters\n");
        return brain;
    }

    size_t printed_neurons = 0;
    for (uint64_t cid = 0; cid < n_clusters; cid++) {
        if (!ensure_space(off, buf_len, 4 + 4)) { fprintf(stderr, "truncated cluster header\n"); return brain; }
        uint32_t cluster_id = read_u32_le(buf, &off, buf_len);
        uint32_t cluster_count = read_u32_le(buf, &off, buf_len);

        /* debug if large but continue */
        if (cluster_count > 10000000U) {
            fprintf(stderr, "DEBUG: cluster %u has large size %u\n", cluster_id, cluster_count);
        }

        for (uint32_t i = 0; i < cluster_count; ++i) {
            Neuron n;
            memset(&n, 0, sizeof(Neuron));

            /* base fields */
            if (!ensure_space(off, buf_len, 8 + 4*3 + 4*3)) { fprintf(stderr, "truncated neuron base\n"); return brain; }
            n.root_id = read_u64_le(buf, &off, buf_len);
            n.x = read_f32_le(buf, &off, buf_len);
            n.y = read_f32_le(buf, &off, buf_len);
            n.z = read_f32_le(buf, &off, buf_len);

            n.type_id = read_u32_le(buf, &off, buf_len);
            n.syn_in  = read_u32_le(buf, &off, buf_len);
            n.syn_out = read_u32_le(buf, &off, buf_len);

            /* neighbors count: try 64-bit, fall back to 32-bit */
            size_t off_before = off;
            uint64_t n_neighbors = 0;
            if (ensure_space(off, buf_len, 8)) {
                n_neighbors = read_u64_le(buf, &off, buf_len);
            } else if (ensure_space(off, buf_len, 4)) {
                n_neighbors = read_u32_le(buf, &off, buf_len);
            } else {
                fprintf(stderr, "truncated neighbors count\n");
                return brain;
            }

            /* if count doesn't fit remaining buffer, try 32-bit reinterpret */
            size_t needed_neighbors_bytes = (size_t)n_neighbors * sizeof(uint64_t);
            if (!ensure_space(off, buf_len, needed_neighbors_bytes)) {
                /* rewind and try 32-bit */
                off = off_before;
                if (ensure_space(off, buf_len, 4)) {
                    uint32_t alt32 = read_u32_le(buf, &off, buf_len);
                    size_t alt_needed = (size_t)alt32 * sizeof(uint64_t);
                    if (ensure_space(off, buf_len, alt_needed)) {
                        n_neighbors = alt32;
                        fprintf(stderr, "warning: neighbors count parsed as 32-bit for neuron %llu -> %u\n",
                                (unsigned long long)n.root_id, (unsigned int)alt32);
                    } else {
                        /* clamp to what remains (safe fallback) */
                        size_t remain = buf_len - off;
                        uint64_t max_possible = remain / sizeof(uint64_t);
                        if (max_possible > 0) {
                            n_neighbors = max_possible;
                            fprintf(stderr, "DEBUG: neighbors count clamped to %llu for neuron %llu (insufficient bytes)\n",
                                    (unsigned long long)n_neighbors, (unsigned long long)n.root_id);
                        } else {
                            n_neighbors = 0;
                            fprintf(stderr, "DEBUG: neighbors count set to 0 for neuron %llu (no bytes remain)\n",
                                    (unsigned long long)n.root_id);
                        }
                    }
                } else {
                    /* no space to reinterpret; set to 0 */
                    n_neighbors = 0;
                    fprintf(stderr, "DEBUG: neighbors count set to 0 for neuron %llu (cannot reinterpret)\n",
                            (unsigned long long)n.root_id);
                }
            }

            n.n_neighbors = n_neighbors;
            if (n.n_neighbors > 0) {
                size_t bytes = (size_t)n.n_neighbors * sizeof(uint64_t);
                n.neighbors = xmalloc(bytes);
                memcpy(n.neighbors, buf + off, bytes);
                off += bytes;
            } else n.neighbors = NULL;

            /* out synapses: similar tolerant logic */
            off_before = off;
            uint64_t n_out_syn = 0;
            if (ensure_space(off, buf_len, 8)) {
                n_out_syn = read_u64_le(buf, &off, buf_len);
            } else if (ensure_space(off, buf_len, 4)) {
                n_out_syn = read_u32_le(buf, &off, buf_len);
            } else {
                fprintf(stderr, "truncated synapse count\n");
                return brain;
            }

            size_t needed_syn_bytes = (size_t)n_out_syn * sizeof(Synapse);
            if (!ensure_space(off, buf_len, needed_syn_bytes)) {
                off = off_before;
                if (ensure_space(off, buf_len, 4)) {
                    uint32_t alt32 = read_u32_le(buf, &off, buf_len);
                    size_t alt_needed = (size_t)alt32 * sizeof(Synapse);
                    if (ensure_space(off, buf_len, alt_needed)) {
                        n_out_syn = alt32;
                        fprintf(stderr, "warning: synapse count parsed as 32-bit for neuron %llu -> %u\n",
                                (unsigned long long)n.root_id, (unsigned int)alt32);
                    } else {
                        /* clamp to remaining whole synapse structs */
                        size_t remain = buf_len - off;
                        uint64_t max_possible = remain / sizeof(Synapse);
                        if (max_possible > 0) {
                            n_out_syn = max_possible;
                            fprintf(stderr, "DEBUG: synapse count clamped to %llu for neuron %llu (insufficient bytes)\n",
                                    (unsigned long long)n_out_syn, (unsigned long long)n.root_id);
                        } else {
                            n_out_syn = 0;
                            fprintf(stderr, "DEBUG: synapse count set to 0 for neuron %llu (no bytes remain)\n",
                                    (unsigned long long)n.root_id);
                        }
                    }
                } else {
                    n_out_syn = 0;
                    fprintf(stderr, "DEBUG: synapse count set to 0 for neuron %llu (cannot reinterpret)\n",
                            (unsigned long long)n.root_id);
                }
            }

            n.n_out_synapses = n_out_syn;
            if (n.n_out_synapses > 0) {
                n.out_synapses = xmalloc((size_t)n.n_out_synapses * sizeof(Synapse));
                for (uint64_t s = 0; s < n.n_out_synapses; ++s) {
                    Synapse *sp = &n.out_synapses[s];
                    /* if truncated in the middle, read helpers will abort */
                    sp->post_id = read_u64_le(buf, &off, buf_len);
                    sp->pre_x = read_f32_le(buf, &off, buf_len);
                    sp->pre_y = read_f32_le(buf, &off, buf_len);
                    sp->pre_z = read_f32_le(buf, &off, buf_len);
                    sp->post_x = read_f32_le(buf, &off, buf_len);
                    sp->post_y = read_f32_le(buf, &off, buf_len);
                    sp->post_z = read_f32_le(buf, &off, buf_len);
                    sp->size   = read_f32_le(buf, &off, buf_len);
                }
            } else {
                n.out_synapses = NULL;
            }

            brain.neurons = append_neuron(brain.neurons, &brain.n_neurons, n);

            /* realtime progress every 1024 neurons */
            if (++printed_neurons % 1024 == 0) {
                fprintf(stderr, "\rClusters processed: %6llu/%6llu  Neurons loaded: %10zu",
                        (unsigned long long)(cid+1), (unsigned long long)n_clusters, brain.n_neurons);
                fflush(stderr);
            }
        }
    }

    /* After clusters, try to read optional global synapse-event block:
       u64 n_syn_events, then n_syn_events * (u32 voxel + 7*f32) */
    brain.syn_events = NULL;
    brain.n_syn_events = 0;
    brain.syn_event_on_disk = 0;
    brain.syn_event_path = NULL;

    if (ensure_space(off, buf_len, 8)) {
        uint64_t n_syn_events = read_u64_le(buf, &off, buf_len);
        if (n_syn_events > 0) {
            const size_t event_bytes = 4 + 7 * 4; /* 32 bytes per event */
            size_t total_needed = 0;
            if (__builtin_mul_overflow_p((size_t)n_syn_events, event_bytes, (size_t)0)) {
                total_needed = 0;
            } else {
                total_needed = (size_t)n_syn_events * event_bytes;
            }

            uint64_t ram_budget = compute_ram_budget();

            /* Decide whether to allocate events in RAM or stream to disk.
               Heuristic: only allocate if total_needed <= ram_budget/2 and also reasonably sized. */
            int allocate_in_ram = 0;
            if (total_needed > 0 && total_needed <= ram_budget / 2) allocate_in_ram = 1;

            if (allocate_in_ram && total_needed && ensure_space(off, buf_len, total_needed)) {
                SynEvent *events = malloc((size_t)n_syn_events * sizeof(SynEvent));
                if (!events) {
                    allocate_in_ram = 0;
                    fprintf(stderr, "DEBUG: allocation of syn_events failed; will attempt streaming to disk\n");
                } else {
                    uint64_t ev_count = 0;
                    double total_ev_len = 0.0;
                    for (uint64_t e = 0; e < n_syn_events; ++e) {
                        uint32_t voxel = read_u32_le(buf, &off, buf_len);
                        float pre_x = read_f32_le(buf, &off, buf_len);
                        float pre_y = read_f32_le(buf, &off, buf_len);
                        float pre_z = read_f32_le(buf, &off, buf_len);
                        float post_x = read_f32_le(buf, &off, buf_len);
                        float post_y = read_f32_le(buf, &off, buf_len);
                        float post_z = read_f32_le(buf, &off, buf_len);
                        float sizef = read_f32_le(buf, &off, buf_len);

                        events[e].voxel = voxel;
                        events[e].pre_x = pre_x; events[e].pre_y = pre_y; events[e].pre_z = pre_z;
                        events[e].post_x = post_x; events[e].post_y = post_y; events[e].post_z = post_z;
                        events[e].size = sizef;

                        double dx = (double)pre_x - (double)post_x;
                        double dy = (double)pre_y - (double)post_y;
                        double dz = (double)pre_z - (double)post_z;
                        total_ev_len += sqrt(dx*dx + dy*dy + dz*dz);
                        ev_count++;
                    }
                    brain.syn_events = events;
                    brain.n_syn_events = n_syn_events;
                    brain.syn_event_on_disk = 0;
                    if (ev_count > 0) {
                        double avg_ev_len = total_ev_len / (double)ev_count;
                        fprintf(stderr, "Read synapse event block into RAM: events=%llu  avg_event_len=%.5f\n",
                                (unsigned long long)ev_count, avg_ev_len);
                    }
                }
            } else {
                /* Stream to a temp file in the current directory. */
                char tmp_template[] = "syn_events_XXXXXX.bin";
                int fd = mkstemps(tmp_template, 4); /* keep ".bin" suffix safe */
                if (fd < 0) {
                    /* mkstemp fallback (without suffix) */
                    char tmp2[] = "syn_events_XXXXXX";
                    int fd2 = mkstemp(tmp2);
                    if (fd2 < 0) {
                        fprintf(stderr, "DEBUG: cannot create temp file to stream syn events: %s\n", strerror(errno));
                        /* As a final fallback, attempt to read and discard events to advance offset */
                        uint64_t ev_count = 0;
                        double total_ev_len = 0.0;
                        size_t possible = buf_len - off;
                        uint64_t possible_count = possible / event_bytes;
                        uint64_t to_read = possible_count < n_syn_events ? possible_count : n_syn_events;
                        for (uint64_t e = 0; e < to_read; ++e) {
                            uint32_t voxel = read_u32_le(buf, &off, buf_len);
                            float pre_x = read_f32_le(buf, &off, buf_len);
                            float pre_y = read_f32_le(buf, &off, buf_len);
                            float pre_z = read_f32_le(buf, &off, buf_len);
                            float post_x = read_f32_le(buf, &off, buf_len);
                            float post_y = read_f32_le(buf, &off, buf_len);
                            float post_z = read_f32_le(buf, &off, buf_len);
                            float sizef = read_f32_le(buf, &off, buf_len);
                            double dx = (double)pre_x - (double)post_x;
                            double dy = (double)pre_y - (double)post_y;
                            double dz = (double)pre_z - (double)post_z;
                            total_ev_len += sqrt(dx*dx + dy*dy + dz*dz);
                            ev_count++;
                        }
                        brain.syn_events = NULL;
                        brain.n_syn_events = ev_count;
                        brain.syn_event_on_disk = 0;
                        fprintf(stderr, "DEBUG: syn_event streamed-and-discarded: events=%llu\n", (unsigned long long)ev_count);
                    } else {
                        /* we have fd2 and path tmp2 */
                        /* write possible events into fd2 */
                        uint64_t ev_count = 0;
                        double total_ev_len = 0.0;
                        size_t to_write = n_syn_events;
                        if (!ensure_space(off, buf_len, (size_t)n_syn_events * event_bytes)) {
                            size_t remain = buf_len - off;
                            to_write = remain / event_bytes;
                        }
                        for (size_t e = 0; e < to_write; ++e) {
                            uint32_t voxel = read_u32_le(buf, &off, buf_len);
                            float pre_x = read_f32_le(buf, &off, buf_len);
                            float pre_y = read_f32_le(buf, &off, buf_len);
                            float pre_z = read_f32_le(buf, &off, buf_len);
                            float post_x = read_f32_le(buf, &off, buf_len);
                            float post_y = read_f32_le(buf, &off, buf_len);
                            float post_z = read_f32_le(buf, &off, buf_len);
                            float sizef = read_f32_le(buf, &off, buf_len);
                            SynEvent se;
                            se.voxel = voxel;
                            se.pre_x = pre_x; se.pre_y = pre_y; se.pre_z = pre_z;
                            se.post_x = post_x; se.post_y = post_y; se.post_z = post_z;
                            se.size = sizef;
                            ssize_t wr = write(fd2, &se, sizeof(se));
                            if (wr != (ssize_t)sizeof(se)) {
                                /* write error -> stop */
                                fprintf(stderr, "DEBUG: write to temp syn_event file failed: %s\n", strerror(errno));
                                break;
                            }
                            ev_count++;
                            double dx = (double)pre_x - (double)post_x;
                            double dy = (double)pre_y - (double)post_y;
                            double dz = (double)pre_z - (double)post_z;
                            total_ev_len += sqrt(dx*dx + dy*dy + dz*dz);
                        }
                        close(fd2);
                        brain.syn_events = NULL;
                        brain.n_syn_events = ev_count;
                        brain.syn_event_on_disk = 1;
                        brain.syn_event_path = strdup(tmp2);
                        fprintf(stderr, "Streamed syn_event block to disk: path=%s events=%llu\n", brain.syn_event_path, (unsigned long long)ev_count);
                    }
                } else {
                    /* we have fd and tmp_template path */
                    uint64_t ev_count = 0;
                    double total_ev_len = 0.0;
                    size_t to_write = n_syn_events;
                    if (!ensure_space(off, buf_len, (size_t)n_syn_events * event_bytes)) {
                        size_t remain = buf_len - off;
                        to_write = remain / event_bytes;
                    }
                    for (size_t e = 0; e < to_write; ++e) {
                        uint32_t voxel = read_u32_le(buf, &off, buf_len);
                        float pre_x = read_f32_le(buf, &off, buf_len);
                        float pre_y = read_f32_le(buf, &off, buf_len);
                        float pre_z = read_f32_le(buf, &off, buf_len);
                        float post_x = read_f32_le(buf, &off, buf_len);
                        float post_y = read_f32_le(buf, &off, buf_len);
                        float post_z = read_f32_le(buf, &off, buf_len);
                        float sizef = read_f32_le(buf, &off, buf_len);
                        SynEvent se;
                        se.voxel = voxel;
                        se.pre_x = pre_x; se.pre_y = pre_y; se.pre_z = pre_z;
                        se.post_x = post_x; se.post_y = post_y; se.post_z = post_z;
                        se.size = sizef;
                        ssize_t wr = write(fd, &se, sizeof(se));
                        if (wr != (ssize_t)sizeof(se)) {
                            fprintf(stderr, "DEBUG: write to temp syn_event file failed: %s\n", strerror(errno));
                            break;
                        }
                        ev_count++;
                        double dx = (double)pre_x - (double)post_x;
                        double dy = (double)pre_y - (double)post_y;
                        double dz = (double)pre_z - (double)post_z;
                        total_ev_len += sqrt(dx*dx + dy*dy + dz*dz);
                    }
                    close(fd);
                    brain.syn_events = NULL;
                    brain.n_syn_events = ev_count;
                    brain.syn_event_on_disk = 1;
                    brain.syn_event_path = strdup(tmp_template);
                    fprintf(stderr, "Streamed syn_event block to disk: path=%s events=%llu\n", brain.syn_event_path, (unsigned long long)ev_count);
                }
            }
        } else {
            /* zero events */
            brain.syn_events = NULL;
            brain.n_syn_events = 0;
            brain.syn_event_on_disk = 0;
        }
    } else {
        /* no further block present */
        brain.syn_events = NULL;
        brain.n_syn_events = 0;
        brain.syn_event_on_disk = 0;
    }

    /* final newline for progress line */
    fprintf(stderr, "\rClusters processed: %6llu/%6llu  Neurons loaded: %10zu\n",
            (unsigned long long)n_clusters, (unsigned long long)n_clusters, brain.n_neurons);
    return brain;
}

Brain load_nemu(const char* path) {
    Brain brain = { NULL, 0, NULL, 0, 0, NULL };

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s: %s\n", path, strerror(errno)); return brain; }

    if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); fclose(f); return brain; }
    long fsize = ftell(f);
    if (fsize < 0) { perror("ftell"); fclose(f); return brain; }
    if (fseek(f, 0, SEEK_SET) != 0) { perror("fseek"); fclose(f); return brain; }

    unsigned char *comp = xmalloc((size_t)fsize);
    size_t r = fread(comp, 1, (size_t)fsize, f);
    fclose(f);
    if (r != (size_t)fsize) { fprintf(stderr, "short read: %zu != %ld\n", r, fsize); free(comp); return brain; }

    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    zs.next_in = comp;
    zs.avail_in = (uInt)fsize;

    if (inflateInit(&zs) != Z_OK) {
        fprintf(stderr, "inflateInit failed\n");
        free(comp);
        return brain;
    }

    size_t out_cap = 1 << 20;
    size_t out_len = 0;
    unsigned char *outbuf = xmalloc(out_cap);

    int inf_ret = Z_OK;
    while (inf_ret != Z_STREAM_END) {
        if (out_len + (1 << 20) > out_cap) {
            out_cap *= 2;
            outbuf = xrealloc(outbuf, out_cap);
        }
        zs.next_out = outbuf + out_len;
        zs.avail_out = (uInt)(out_cap - out_len);

        inf_ret = inflate(&zs, Z_NO_FLUSH);
        if (inf_ret != Z_OK && inf_ret != Z_STREAM_END && inf_ret != Z_BUF_ERROR) {
            fprintf(stderr, "inflate error: %d\n", inf_ret);
            inflateEnd(&zs);
            free(comp);
            free(outbuf);
            return brain;
        }

        size_t produced = (out_cap - out_len) - zs.avail_out;
        out_len += produced;

        if (inf_ret == Z_BUF_ERROR && produced == 0) {
            out_cap *= 2;
            outbuf = xrealloc(outbuf, out_cap);
            continue;
        }
        if (inf_ret == Z_STREAM_END) break;
    }

    inflateEnd(&zs);
    free(comp);

    Brain parsed = parse_decompressed(outbuf, out_len);
    free(outbuf);
    return parsed;
}

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
        if (b->syn_event_path) {
            /* remove temp file and free path */
            unlink(b->syn_event_path);
            free(b->syn_event_path);
            b->syn_event_path = NULL;
        }
        b->syn_events = NULL;
        b->n_syn_events = 0;
        b->syn_event_on_disk = 0;
    } else {
        free(b->syn_events);
        b->syn_events = NULL;
        b->n_syn_events = 0;
    }
}

/* Test main (compile with -DLOADER_MAIN to include) */
#ifdef LOADER_MAIN

/* simple open-addressing map from uint64 -> size_t (index) */
typedef struct {
    uint64_t key;
    size_t val;
    int used;
} U64ToSizeEntry;

typedef struct {
    U64ToSizeEntry *entries;
    size_t cap;
    size_t len;
} U64ToSizeMap;

static size_t next_pow2_size(size_t n) {
    size_t v = 1;
    while (v < n) v <<= 1;
    return v;
}

static U64ToSizeMap *u64tosize_create(size_t expect) {
    U64ToSizeMap *m = malloc(sizeof(U64ToSizeMap));
    if (!m) { perror("malloc"); exit(1); }
    m->cap = next_pow2_size(expect ? expect*2 : 16);
    m->len = 0;
    m->entries = calloc(m->cap, sizeof(U64ToSizeEntry));
    if (!m->entries) { perror("calloc"); exit(1); }
    return m;
}

static void u64tosize_free(U64ToSizeMap *m) {
    free(m->entries);
    free(m);
}

static void u64tosize_insert(U64ToSizeMap *m, uint64_t key, size_t val) {
    size_t mask = m->cap - 1;
    size_t i = (size_t)(key * 11400714819323198485ULL) & mask;
    while (m->entries[i].used) {
        if (m->entries[i].key == key) { m->entries[i].val = val; return; }
        i = (i + 1) & mask;
    }
    m->entries[i].used = 1;
    m->entries[i].key = key;
    m->entries[i].val = val;
    m->len++;
}

static int u64tosize_lookup(U64ToSizeMap *m, uint64_t key, size_t *out) {
    size_t mask = m->cap - 1;
    size_t i = (size_t)(key * 11400714819323198485ULL) & mask;
    while (m->entries[i].used) {
        if (m->entries[i].key == key) { *out = m->entries[i].val; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

/* adjacency map key: (uint64_t)src_type << 32 | dst_type */
typedef struct {
    uint64_t key;
    uint64_t count;
    int used;
} U64ToU64Entry;

typedef struct {
    U64ToU64Entry *entries;
    size_t cap;
    size_t len;
} U64ToU64Map;

static U64ToU64Map *u64tou64_create(size_t expect) {
    U64ToU64Map *m = malloc(sizeof(U64ToU64Map));
    if (!m) { perror("malloc"); exit(1); }
    m->cap = next_pow2_size(expect ? expect*2 : 64);
    m->len = 0;
    m->entries = calloc(m->cap, sizeof(U64ToU64Entry));
    if (!m->entries) { perror("calloc"); exit(1); }
    return m;
}

static void u64tou64_insert_add(U64ToU64Map *m, uint64_t key, uint64_t add) {
    size_t mask = m->cap - 1;
    size_t i = (size_t)(key * 11400714819323198485ULL) & mask;
    while (m->entries[i].used) {
        if (m->entries[i].key == key) { m->entries[i].count += add; return; }
        i = (i + 1) & mask;
    }
    m->entries[i].used = 1;
    m->entries[i].key = key;
    m->entries[i].count = add;
    m->len++;
}

static void u64tou64_free(U64ToU64Map *m) { free(m->entries); free(m); }

/* comparator for sorting adjacency entries by count desc */
static int cmp_adj_desc(const void *a, const void *b) {
    const U64ToU64Entry *x = a;
    const U64ToU64Entry *y = b;
    if (x->count < y->count) return 1;
    if (x->count > y->count) return -1;
    return 0;
}

/* brute-force nearest neuron finder (used for mapping global syn_events) */
static int find_nearest_bruteforce(Neuron *arr, size_t n, float x, float y, float z, size_t *out_idx) {
    if (n == 0) return 0;
    size_t best = 0;
    double best_d2 = DBL_MAX;
    for (size_t i = 0; i < n; ++i) {
        double dx = (double)arr[i].x - (double)x;
        double dy = (double)arr[i].y - (double)y;
        double dz = (double)arr[i].z - (double)z;
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    *out_idx = best;
    return 1;
}

int main(int argc, char **argv) {
    const char *path = "serialized.nemu";
    if (argc > 1) path = argv[1];

    Brain b = load_nemu(path);
    fprintf(stderr, "Loaded %zu neurons  global_events=%llu  syn_events_on_disk=%d\n",
            b.n_neurons, (unsigned long long)b.n_syn_events, b.syn_event_on_disk);
    if (b.n_neurons == 0 && b.n_syn_events == 0) return 1;

    /* Build root_id -> index map (still useful if per-neuron synapses reference post_id) */
    U64ToSizeMap *idmap = u64tosize_create(b.n_neurons > 0 ? b.n_neurons : 16);
    for (size_t i = 0; i < b.n_neurons; ++i) {
        u64tosize_insert(idmap, b.neurons[i].root_id, i);
    }

    /* Per-cluster (type_id) aggregates: use dynamic arrays keyed by type_id */
    uint32_t max_type = 0;
    for (size_t i = 0; i < b.n_neurons; ++i) if (b.neurons[i].type_id > max_type) max_type = b.neurons[i].type_id;
    size_t n_types = (size_t)max_type + 1;
    if (n_types == 0) n_types = 1;
    uint64_t *type_counts = calloc(n_types, sizeof(uint64_t));
    uint64_t *type_neighbors = calloc(n_types, sizeof(uint64_t));
    uint64_t *type_outsyn = calloc(n_types, sizeof(uint64_t));
    uint64_t *type_insyn = calloc(n_types, sizeof(uint64_t));

    uint64_t total_neighbors = 0;
    uint64_t total_outsyn = 0;
    uint64_t total_insyn = 0;
    uint64_t min_neighbors = UINT64_MAX, max_neighbors = 0;
    uint64_t min_outsyn = UINT64_MAX, max_outsyn = 0;
    uint64_t isolated = 0;

    double total_syn_length_per_neuron = 0.0;
    uint64_t total_syn_count_per_neuron = 0;
    uint64_t missing_post_refs = 0;

    /* adjacency map between types (will aggregate from both per-neuron lists and global events) */
    U64ToU64Map *adj = u64tou64_create(1024);

    /* Aggregate per-neuron synapses if present (optional) */
    for (size_t i = 0; i < b.n_neurons; ++i) {
        Neuron *n = &b.neurons[i];
        uint32_t t = n->type_id;
        if (t >= n_types) t = 0;
        type_counts[t]++;

        uint64_t neigh = n->n_neighbors;
        total_neighbors += neigh;
        type_neighbors[t] += neigh;
        if (neigh < min_neighbors) min_neighbors = neigh;
        if (neigh > max_neighbors) max_neighbors = neigh;

        uint64_t outs = n->n_out_synapses;
        total_outsyn += outs;
        type_outsyn[t] += outs;
        if (outs < min_outsyn) min_outsyn = outs;
        if (outs > max_outsyn) max_outsyn = outs;

        total_insyn += n->syn_in;
        type_insyn[t] += n->syn_in;

        if (neigh == 0 && outs == 0 && n->syn_in == 0) isolated++;

        /* iterate synapses for length and adjacency (only works if per-neuron synapses exist) */
        for (uint64_t s = 0; s < n->n_out_synapses; ++s) {
            Synapse *sp = &n->out_synapses[s];
            double dx = (double)sp->pre_x - (double)sp->post_x;
            double dy = (double)sp->pre_y - (double)sp->post_y;
            double dz = (double)sp->pre_z - (double)sp->post_z;
            double len = sqrt(dx*dx + dy*dy + dz*dz);
            total_syn_length_per_neuron += len;
            total_syn_count_per_neuron++;

            size_t post_idx;
            if (u64tosize_lookup(idmap, sp->post_id, &post_idx)) {
                uint32_t tdst = b.neurons[post_idx].type_id;
                uint64_t key = (((uint64_t)n->type_id) << 32) | ((uint64_t)tdst & 0xffffffffULL);
                u64tou64_insert_add(adj, key, 1);
            } else {
                missing_post_refs++;
            }
        }
    }

    /* If new-style global synapse events exist, map them to nearest neurons.
       If events are on-disk, stream file; otherwise iterate memory array. */
    double event_avg_len = 0.0;
    uint64_t event_mapped = 0;
    if (b.n_syn_events > 0) {
        double total_ev_len = 0.0;
        uint64_t mapped = 0;
        if (b.syn_event_on_disk && b.syn_event_path) {
            /* stream from disk file */
            int fd = open(b.syn_event_path, O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "WARNING: cannot open syn_event file %s: %s\n", b.syn_event_path, strerror(errno));
            } else {
                SynEvent se;
                uint64_t e_read = 0;
                while (1) {
                    ssize_t rr = read(fd, &se, sizeof(se));
                    if (rr == 0) break;
                    if (rr < 0) { fprintf(stderr, "ERROR: read syn_event file: %s\n", strerror(errno)); break; }
                    if (rr != (ssize_t)sizeof(se)) {
                        fprintf(stderr, "DEBUG: partial syn_event read, stopping\n");
                        break;
                    }
                    /* map pre/post to nearest neuron indices */
                    size_t pre_idx = 0, post_idx = 0;
                    int have_pre = find_nearest_bruteforce(b.neurons, b.n_neurons, se.pre_x, se.pre_y, se.pre_z, &pre_idx);
                    int have_post = find_nearest_bruteforce(b.neurons, b.n_neurons, se.post_x, se.post_y, se.post_z, &post_idx);
                    double dx = (double)se.pre_x - (double)se.post_x;
                    double dy = (double)se.pre_y - (double)se.post_y;
                    double dz = (double)se.pre_z - (double)se.post_z;
                    total_ev_len += sqrt(dx*dx + dy*dy + dz*dz);
                    if (have_pre && have_post) {
                        mapped++;
                        uint32_t tpre = b.neurons[pre_idx].type_id;
                        uint32_t tpost = b.neurons[post_idx].type_id;
                        if (tpre >= n_types) tpre = 0;
                        if (tpost >= n_types) tpost = 0;
                        uint64_t key = (((uint64_t)tpre) << 32) | ((uint64_t)tpost & 0xffffffffULL);
                        u64tou64_insert_add(adj, key, 1);
                    }
                    e_read++;
                }
                close(fd);
                event_mapped = mapped;
                if (b.n_syn_events) event_avg_len = total_ev_len / (double)b.n_syn_events;
            }
        } else if (b.syn_events) {
            for (uint64_t e = 0; e < b.n_syn_events; ++e) {
                SynEvent *ev = &b.syn_events[e];
                size_t pre_idx = 0, post_idx = 0;
                int have_pre = find_nearest_bruteforce(b.neurons, b.n_neurons, ev->pre_x, ev->pre_y, ev->pre_z, &pre_idx);
                int have_post = find_nearest_bruteforce(b.neurons, b.n_neurons, ev->post_x, ev->post_y, ev->post_z, &post_idx);
                double dx = (double)ev->pre_x - (double)ev->post_x;
                double dy = (double)ev->pre_y - (double)ev->post_y;
                double dz = (double)ev->pre_z - (double)ev->post_z;
                total_ev_len += sqrt(dx*dx + dy*dy + dz*dz);
                if (have_pre && have_post) {
                    mapped++;
                    uint32_t tpre = b.neurons[pre_idx].type_id;
                    uint32_t tpost = b.neurons[post_idx].type_id;
                    if (tpre >= n_types) tpre = 0;
                    if (tpost >= n_types) tpost = 0;
                    uint64_t key = (((uint64_t)tpre) << 32) | ((uint64_t)tpost & 0xffffffffULL);
                    u64tou64_insert_add(adj, key, 1);
                }
            }
            event_mapped = mapped;
            if (b.n_syn_events) event_avg_len = total_ev_len / (double)b.n_syn_events;
        }
    }

    double avg_neighbors = b.n_neurons ? (double)total_neighbors / (double)b.n_neurons : 0.0;
    double avg_outsyn = b.n_neurons ? (double)total_outsyn / (double)b.n_neurons : 0.0;
    double avg_insyn = b.n_neurons ? (double)total_insyn / (double)b.n_neurons : 0.0;
    double avg_syn_length_per_neuron = total_syn_count_per_neuron ? (double)total_syn_length_per_neuron / (double)total_syn_count_per_neuron : 0.0;

    fprintf(stdout, "Neurons: %zu\n", b.n_neurons);
    fprintf(stdout, "Total neighbors: %llu  avg: %.3f  min: %llu  max: %llu\n",
            (unsigned long long)total_neighbors, avg_neighbors,
            (unsigned long long)min_neighbors, (unsigned long long)max_neighbors);
    fprintf(stdout, "Total out-synapses (per-neuron lists): %llu  avg per neuron: %.3f  min: %llu  max: %llu\n",
            (unsigned long long)total_outsyn, avg_outsyn,
            (unsigned long long)min_outsyn, (unsigned long long)max_outsyn);
    fprintf(stdout, "Total in-synapses (from header counts): %llu  avg: %.3f\n",
            (unsigned long long)total_insyn, avg_insyn);
    fprintf(stdout, "Total synapses iterated (per-neuron): %llu  avg length: %.5f  missing post refs: %llu\n",
            (unsigned long long)total_syn_count_per_neuron, avg_syn_length_per_neuron, (unsigned long long)missing_post_refs);

    if (b.n_syn_events > 0) {
        fprintf(stdout, "Global synapse events present: %llu  avg event length: %.5f  mapped_events: %llu  on_disk: %d\n",
                (unsigned long long)b.n_syn_events, event_avg_len, (unsigned long long)event_mapped, b.syn_event_on_disk);
    } else {
        fprintf(stdout, "No global synapse events present in file.\n");
    }

    fprintf(stdout, "Isolated neurons (no neigh/no out/no in): %llu\n", (unsigned long long)isolated);

    /* per-cluster summary */
    fprintf(stdout, "\nPer-cluster summary (type_id, count, avg_neighbors, avg_outsyn, avg_insyn):\n");
    for (uint32_t tid = 0; tid < n_types; ++tid) {
        if (type_counts[tid] == 0) continue;
        double a_neigh = (double)type_neighbors[tid] / (double)type_counts[tid];
        double a_out = (double)type_outsyn[tid] / (double)type_counts[tid];
        double a_in = (double)type_insyn[tid] / (double)type_counts[tid];
        printf("  %5u : %8llu  neigh_avg: %7.3f  out_avg: %7.3f  in_avg: %7.3f\n",
               tid,
               (unsigned long long)type_counts[tid],
               a_neigh, a_out, a_in);
    }

    /* top adjacency pairs (derived from whichever synapse data was available) */
    size_t adj_count = adj->len;
    if (adj_count > 0) {
        U64ToU64Entry *arr = malloc(adj->cap * sizeof(U64ToU64Entry));
        size_t idx = 0;
        for (size_t i = 0; i < adj->cap; ++i) {
            if (adj->entries[i].used) arr[idx++] = adj->entries[i];
        }
        qsort(arr, idx, sizeof(U64ToU64Entry), cmp_adj_desc);
        fprintf(stdout, "\nTop cluster->cluster synapse connections (top 20):\n");
        size_t show = idx < 20 ? idx : 20;
        for (size_t i = 0; i < show; ++i) {
            uint32_t a = (uint32_t)(arr[i].key >> 32);
            uint32_t btype = (uint32_t)(arr[i].key & 0xffffffffULL);
            printf("  %5u -> %5u : %10llu\n", a, btype, (unsigned long long)arr[i].count);
        }
        free(arr);
    } else {
        fprintf(stdout, "No adjacency data found.\n");
    }

    /* Basic sanity checks: report neurons with extreme values */
    fprintf(stdout, "\nSanity checks: listing neurons with extreme values (up to 25 total)\n");
    size_t reported = 0;
    for (size_t i = 0; i < b.n_neurons && reported < 25; ++i) {
        Neuron *n = &b.neurons[i];
        if (n->n_neighbors == 0 || n->n_out_synapses == 0 || n->syn_in == 0) {
            printf("root_id:%llu type:%u neigh:%llu outsyn:%llu insyn:%u pos:(%.3f,%.3f,%.3f)\n",
                   (unsigned long long)n->root_id,
                   n->type_id,
                   (unsigned long long)n->n_neighbors,
                   (unsigned long long)n->n_out_synapses,
                   n->syn_in,
                   n->x, n->y, n->z);
            reported++;
        }
    }
    if (reported == 0) printf("  No trivial isolates found in first pass.\n");

    /* free resources */
    u64tosize_free(idmap);
    u64tou64_free(adj);
    free(type_counts); free(type_neighbors); free(type_outsyn); free(type_insyn);

    for (size_t i = 0; i < b.n_neurons; ++i) {
        free(b.neurons[i].neighbors);
        free(b.neurons[i].out_synapses);
    }
    free(b.neurons);
    b.neurons = NULL;

    /* if we streamed syn_events to disk and want to keep them for further processing,
       do not unlink here; free_brain will clean up. For this tool we free immediately. */
    if (b.syn_event_on_disk && b.syn_event_path) {
        /* keep file for inspection; free_brain will unlink if desired */
    } else {
        free(b.syn_events);
    }

    return 0;
}
#endif