// serialize.c
//
// Revised serializer: treats synapses as independent spatial events (no root_id matching).
// Builds neuron table and clusters (by type), streams synapse events to disk (temp file),
// and writes a compact .nemu that contains clusters (neurons & neighbors) plus a separate
// synapse event stream (voxel index + coordinates + size). This avoids assigning synapses
// to specific neurons and keeps RAM usage bounded.
//
// Compile:
//   gcc serialize.c -O3 -lz -lm -o serializer
//
// Required inputs (same names as Python pipeline):
//   cell_stats.csv.gz
//   consolidated_cell_types.csv.gz
//   coordinates.csv.gz
//   connections_princeton_no_threshold.csv.gz
//   fafb_v783_princeton_synapse_table.csv.gz
//
// Notes:
// - The new .nemu layout (version still 3) appends, after clusters, a synapse event block:
//     uint64_t n_synapse_events
//     repeated for each event:
//       uint32_t voxel_index (0xFFFFFFFF if none)
//       float pre_x, pre_y, pre_z
//       float post_x, post_y, post_z
//       float size
//
// - The serializer writes synapse events sequentially to a temp file during parsing, then
//   streams them into the compressed .nemu at the end. This keeps peak RAM low.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <zlib.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#define MAGIC "NEMU"
#define VERSION 3
#define HASH_SIZE 131071      // prime-ish; adjust for memory / size
#define LINE_BUFSZ (1<<16)

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

static NeuronTable *table_create(void) {
    NeuronTable *t = calloc(1, sizeof(*t));
    if (!t) { perror("calloc"); exit(1); }
    t->buckets = calloc(HASH_SIZE, sizeof(Node*));
    if (!t->buckets) { perror("calloc"); exit(1); }
    return t;
}

static Neuron *table_get(NeuronTable *t, uint64_t key) {
    uint64_t h = hash_u64(key);
    Node *n = t->buckets[h];
    while (n) {
        if (n->key == key) return n->value;
        n = n->next;
    }
    return NULL;
}

static void table_put(NeuronTable *t, uint64_t key, Neuron *val) {
    uint64_t h = hash_u64(key);
    Node *n = malloc(sizeof(Node));
    if (!n) { perror("malloc"); exit(1); }
    n->key = key;
    n->value = val;
    n->next = t->buckets[h];
    t->buckets[h] = n;
}

/* ---------- Type map ---------- */
typedef struct TypeNode {
    char *key;
    uint32_t id;
    struct TypeNode *next;
} TypeNode;

typedef struct {
    TypeNode **buckets;
    uint32_t next_id;
} TypeMap;

static inline uint64_t strhash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static TypeMap *typemap_create(void) {
    TypeMap *m = calloc(1, sizeof(*m));
    if (!m) { perror("calloc"); exit(1); }
    m->buckets = calloc(HASH_SIZE, sizeof(TypeNode*));
    if (!m->buckets) { perror("calloc"); exit(1); }
    m->next_id = 1;
    return m;
}

static uint32_t typemap_get_or_add(TypeMap *m, const char *key) {
    if (!key || !*key) return 0;
    uint64_t h = strhash(key) % HASH_SIZE;
    TypeNode *n = m->buckets[h];
    while (n) {
        if (strcmp(n->key, key) == 0) return n->id;
        n = n->next;
    }
    TypeNode *nn = malloc(sizeof(TypeNode));
    if (!nn) { perror("malloc"); exit(1); }
    nn->key = strdup(key);
    nn->id = m->next_id++;
    nn->next = m->buckets[h];
    m->buckets[h] = nn;
    return nn->id;
}

/* ---------- Utilities ---------- */
static char *trim_inplace(char *s) {
    if (!s) return s;
    while (isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *end-- = '\0';
    return s;
}

/* ---------- CSV parsing (basic) ---------- */
static char **split_csv_line(const char *line, size_t *out_count) {
    size_t cap = 32;
    size_t cnt = 0;
    char **arr = malloc(cap * sizeof(char*));
    const char *p = line;
    while (*p && (*p == '\r' || *p == '\n')) p++;
    while (*p) {
        if (cnt + 1 > cap) { cap *= 2; arr = realloc(arr, cap * sizeof(char*)); if(!arr){perror("realloc"); exit(1);} }
        char *field;
        if (*p == '"') {
            p++;
            size_t bufcap = 256;
            size_t buflen = 0;
            field = malloc(bufcap);
            while (*p) {
                if (*p == '"') {
                    if (p[1] == '"') {
                        if (buflen + 1 >= bufcap) { bufcap *= 2; field = realloc(field, bufcap); }
                        field[buflen++] = '"';
                        p += 2;
                        continue;
                    } else {
                        p++;
                        break;
                    }
                } else {
                    if (buflen + 1 >= bufcap) { bufcap *= 2; field = realloc(field, bufcap); }
                    field[buflen++] = *p++;
                }
            }
            field[buflen] = '\0';
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        } else {
            const char *start = p;
            while (*p && *p != ',' && *p != '\r' && *p != '\n') p++;
            size_t len = p - start;
            field = malloc(len + 1);
            memcpy(field, start, len);
            field[len] = '\0';
            if (*p == ',') p++;
        }
        arr[cnt++] = trim_inplace(field);
    }
    *out_count = cnt;
    return arr;
}
static void free_csv_fields(char **f, size_t n) {
    if (!f) return;
    for (size_t i = 0; i < n; ++i) free(f[i]);
    free(f);
}

/* ---------- gz line reader ---------- */
static int gz_getline(gzFile f, char *buf, int bufsz) {
    if (!gzgets(f, buf, bufsz)) return 0;
    size_t L = strlen(buf);
    while (L > 0 && (buf[L-1] == '\n' || buf[L-1] == '\r')) buf[--L] = '\0';
    return 1;
}

/* ---------- Parsing helpers ---------- */
static int find_field_index(char **hdr, size_t n, const char *exact, const char *substr) {
    for (size_t i = 0; i < n; ++i) {
        if (!hdr[i]) continue;
        if (strcasecmp(hdr[i], exact) == 0) return (int)i;
    }
    if (substr) {
        for (size_t i = 0; i < n; ++i) {
            if (!hdr[i]) continue;
            if (strstr(hdr[i], substr)) return (int)i;
        }
    }
    return -1;
}
static void parse_position_field(const char *text, float *x, float *y, float *z) {
    *x = *y = *z = 0.0f;
    if (!text) return;
    char *tmp = strdup(text);
    char *t = tmp;
    while (*t && isspace((unsigned char)*t)) t++;
    char *end = tmp + strlen(tmp) - 1;
    while (end > t && isspace((unsigned char)*end)) *end-- = '\0';
    if (*t == '[' && *end == ']') { t++; *end = '\0'; }
    char *s = t;
    char *parts[3] = {0};
    size_t p = 0;
    char *tok = strtok(s, " ,\t");
    while (tok && p < 3) { parts[p++] = tok; tok = strtok(NULL, " ,\t"); }
    if (p >= 1) *x = atof(parts[0]);
    if (p >= 2) *y = atof(parts[1]);
    if (p >= 3) *z = atof(parts[2]);
    free(tmp);
}

/* ---------- Neuron helpers ---------- */
static void neuron_add_neighbor(Neuron *n, uint64_t id) {
    n->neighbors = realloc(n->neighbors, (n->n_neighbors + 1) * sizeof(uint64_t));
    if (!n->neighbors) { perror("realloc"); exit(1); }
    n->neighbors[n->n_neighbors++] = id;
}

/* ---------- Spatial voxel grid (for binning synapse events) ---------- */
typedef struct {
    size_t *items;
    size_t count;
} Cell;

typedef struct {
    float minx, miny, minz;
    float cell_size;
    int nx, ny, nz;
    size_t n_cells;
    Cell *cells;
} SpatialGrid;

static void spatial_grid_free(SpatialGrid *g) {
    if (!g) return;
    for (size_t i = 0; i < g->n_cells; ++i) free(g->cells[i].items);
    free(g->cells);
    free(g);
}

/* collect neurons into an array (returns malloc'd array) */
static Neuron **collect_neurons_array(NeuronTable *table, size_t *out_n) {
    size_t cap = 1024;
    size_t n = 0;
    Neuron **arr = malloc(cap * sizeof(Neuron*));
    if (!arr) { perror("malloc"); exit(1); }
    for (size_t i = 0; i < HASH_SIZE; ++i) {
        Node *node = table->buckets[i];
        while (node) {
            if (n + 1 > cap) { cap *= 2; arr = realloc(arr, cap * sizeof(Neuron*)); if(!arr){perror("realloc");exit(1);} }
            arr[n++] = node->value;
            node = node->next;
        }
    }
    *out_n = n;
    return arr;
}

/* create voxel grid that adapts resolution to limit total cells */
static SpatialGrid *spatial_grid_create(Neuron **neurons, size_t n)
{
    if (n == 0) return NULL;

    float minx = neurons[0]->x, maxx = neurons[0]->x;
    float miny = neurons[0]->y, maxy = neurons[0]->y;
    float minz = neurons[0]->z, maxz = neurons[0]->z;

    for (size_t i = 1; i < n; i++) {
        if (neurons[i]->x < minx) minx = neurons[i]->x;
        if (neurons[i]->x > maxx) maxx = neurons[i]->x;
        if (neurons[i]->y < miny) miny = neurons[i]->y;
        if (neurons[i]->y > maxy) maxy = neurons[i]->y;
        if (neurons[i]->z < minz) minz = neurons[i]->z;
        if (neurons[i]->z > maxz) maxz = neurons[i]->z;
    }

    double dx = maxx - minx;
    double dy = maxy - miny;
    double dz = maxz - minz;
    if (dx <= 0) dx = 1;
    if (dy <= 0) dy = 1;
    if (dz <= 0) dz = 1;

    size_t max_cells = 262144;
    const char *env = getenv("GRID_MAX_CELLS");
    if (env) {
        long long v = atoll(env);
        if (v > 0) max_cells = (size_t)v;
    }

    double volume = dx * dy * dz;
    double avg_vol = volume / (double)n;
    float cell_size = (float)(pow(avg_vol, 1.0/3.0) * 1.5);
    if (cell_size < 0.5f) cell_size = 0.5f;

    int nx, ny, nz;
    uint64_t cells;
    while (1) {
        nx = (int)ceil(dx / cell_size);
        ny = (int)ceil(dy / cell_size);
        nz = (int)ceil(dz / cell_size);
        if (nx < 1) nx = 1;
        if (ny < 1) ny = 1;
        if (nz < 1) nz = 1;
        cells = (uint64_t)nx * ny * nz;
        if (cells <= max_cells) break;
        cell_size *= 1.25f;
    }

    SpatialGrid *g = malloc(sizeof(SpatialGrid));
    if (!g) { perror("malloc"); return NULL; }
    g->minx = minx; g->miny = miny; g->minz = minz;
    g->cell_size = cell_size; g->nx = nx; g->ny = ny; g->nz = nz; g->n_cells = (size_t)cells;
    g->cells = calloc(g->n_cells, sizeof(Cell));
    if (!g->cells) { perror("calloc"); free(g); return NULL; }

    /* populate voxels with neuron indices (may be used for local type heuristics) */
    for (size_t i = 0; i < n; i++) {
        Neuron *nn = neurons[i];
        int ix = (int)floor((nn->x - minx) / cell_size);
        int iy = (int)floor((nn->y - miny) / cell_size);
        int iz = (int)floor((nn->z - minz) / cell_size);
        if (ix < 0) ix = 0; if (iy < 0) iy = 0; if (iz < 0) iz = 0;
        if (ix >= nx) ix = nx - 1; if (iy >= ny) iy = ny - 1; if (iz >= nz) iz = nz - 1;
        size_t cell_i = ix + iy * nx + iz * nx * ny;
        Cell *c = &g->cells[cell_i];
        size_t newcount = c->count + 1;
        size_t *items = realloc(c->items, newcount * sizeof(size_t));
        if (!items) { perror("realloc"); spatial_grid_free(g); return NULL; }
        c->items = items;
        c->items[c->count++] = i;
    }

    fprintf(stderr, "spatial_grid: nx=%d ny=%d nz=%d cells=%zu cell_size=%.3f\n",
            nx, ny, g->nz, g->n_cells, g->cell_size);
    return g;
}

/* compute voxel index for a coordinate (or UINT32_MAX if grid==NULL) */
static uint32_t spatial_grid_voxel_index(SpatialGrid *g, float x, float y, float z) {
    if (!g) return UINT32_MAX;
    int cx = (int)floor((x - g->minx) / g->cell_size);
    int cy = (int)floor((y - g->miny) / g->cell_size);
    int cz = (int)floor((z - g->minz) / g->cell_size);
    if (cx < 0) cx = 0; if (cx >= g->nx) cx = g->nx - 1;
    if (cy < 0) cy = 0; if (cy >= g->ny) cy = g->ny - 1;
    if (cz < 0) cz = 0; if (cz >= g->nz) cz = g->nz - 1;
    return (uint32_t)(cx + cy * g->nx + cz * g->nx * g->ny);
}

/* ---------- Synapse streaming (single-pass, event-based) ---------- */

/* globals for temp path and counts */
static char *global_syn_temp_path = NULL;
static uint64_t global_syn_event_count = 0;

/* Stream synapse CSV -> temporary SynEvent binary file (append-only).
   No assignment to neurons. Uses voxel binning (spatial_grid) to store voxel index
   for downstream spatial aggregation. */
static void stream_synapses_into_table(const char *path, NeuronTable *table) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); return; }

    size_t n_neurons = 0;
    Neuron **neurons = collect_neurons_array(table, &n_neurons);

    SpatialGrid *grid = spatial_grid_create(neurons, n_neurons); /* may be NULL */

    char *line = malloc(LINE_BUFSZ);
    if (!line) { perror("malloc"); spatial_grid_free(grid); free(neurons); gzclose(f); return; }

    /* header */
    if (!gz_getline(f, line, LINE_BUFSZ)) { free(line); spatial_grid_free(grid); free(neurons); gzclose(f); return; }
    size_t ncols = 0;
    char **hdr = split_csv_line(line, &ncols);

    int idx_pre_x=-1, idx_pre_y=-1, idx_pre_z=-1;
    int idx_post_x=-1, idx_post_y=-1, idx_post_z=-1;
    int idx_size=-1;
    for (size_t i = 0; i < ncols; ++i) {
        if (!hdr[i]) continue;
        if (strstr(hdr[i], "pre_x")) idx_pre_x = (int)i;
        if (strstr(hdr[i], "pre_y")) idx_pre_y = (int)i;
        if (strstr(hdr[i], "pre_z")) idx_pre_z = (int)i;
        if (strstr(hdr[i], "post_x")) idx_post_x = (int)i;
        if (strstr(hdr[i], "post_y")) idx_post_y = (int)i;
        if (strstr(hdr[i], "post_z")) idx_post_z = (int)i;
        if (strstr(hdr[i], "size")) idx_size = (int)i;
    }
    free_csv_fields(hdr, ncols);

    /* create temp file for synapse events (binary append) */
    char tmp_template[] = "/tmp/serializer_syneventsXXXXXX";
    int fd = mkstemp(tmp_template);
    if (fd < 0) { perror("mkstemp"); goto cleanup_all; }
    FILE *tmpf = fdopen(fd, "wb+");
    if (!tmpf) { perror("fdopen"); close(fd); goto cleanup_all; }
    free(global_syn_temp_path);
    global_syn_temp_path = strdup(tmp_template);
    global_syn_event_count = 0;

    /* buffer for batched writes */
    size_t buf_cap = 1 << 16;
    SynEvent *buf = malloc(buf_cap * sizeof(SynEvent));
    if (!buf) { perror("malloc"); fclose(tmpf); goto cleanup_all; }
    size_t buf_len = 0;

    size_t processed = 0;
    size_t accepted = 0;
    while (gz_getline(f, line, LINE_BUFSZ)) {
        size_t cnt = 0;
        char **cols = split_csv_line(line, &cnt);
        if (cnt == 0) { free_csv_fields(cols, cnt); continue; }

        float pre_x = 0, pre_y = 0, pre_z = 0;
        float post_x = 0, post_y = 0, post_z = 0;
        float sizef = 1.0f;
        if (idx_pre_x >= 0 && idx_pre_x < (int)cnt) pre_x = atof(cols[idx_pre_x]);
        if (idx_pre_y >= 0 && idx_pre_y < (int)cnt) pre_y = atof(cols[idx_pre_y]);
        if (idx_pre_z >= 0 && idx_pre_z < (int)cnt) pre_z = atof(cols[idx_pre_z]);
        if (idx_post_x >= 0 && idx_post_x < (int)cnt) post_x = atof(cols[idx_post_x]);
        if (idx_post_y >= 0 && idx_post_y < (int)cnt) post_y = atof(cols[idx_post_y]);
        if (idx_post_z >= 0 && idx_post_z < (int)cnt) post_z = atof(cols[idx_post_z]);
        if (idx_size >= 0 && idx_size < (int)cnt) sizef = atof(cols[idx_size]);

        uint32_t voxel_pre = spatial_grid_voxel_index(grid, pre_x, pre_y, pre_z);
        /* here we store voxel_pre only; downstream analysis can use voxel or coordinates */
        SynEvent ev;
        ev.voxel = voxel_pre;
        ev.pre_x = pre_x; ev.pre_y = pre_y; ev.pre_z = pre_z;
        ev.post_x = post_x; ev.post_y = post_y; ev.post_z = post_z;
        ev.size = sizef;

        buf[buf_len++] = ev;
        accepted++;

        if (buf_len >= buf_cap) {
            size_t w = fwrite(buf, sizeof(SynEvent), buf_len, tmpf);
            if (w != buf_len) perror("fwrite syntemp");
            buf_len = 0;
        }

        free_csv_fields(cols, cnt);

        processed++;
        if ((processed & 0x1FFFF) == 0) {
            fprintf(stderr, "\rSynapse stream lines processed: %10zu  events buffered: %8zu", processed, accepted);
            fflush(stderr);
        }
    }

    if (buf_len > 0) {
        size_t w = fwrite(buf, sizeof(SynEvent), buf_len, tmpf);
        if (w != buf_len) perror("fwrite syntemp final");
        buf_len = 0;
    }

    fflush(tmpf);
    fseeko(tmpf, 0, SEEK_SET);
    /* compute total events from file size */
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
        global_syn_event_count = (uint64_t)st.st_size / (uint64_t)sizeof(SynEvent);
    } else {
        global_syn_event_count = 0;
    }

    fprintf(stderr, "\rSynapse stream done: lines=%10zu  events=%10" PRIu64 "\n", processed, global_syn_event_count);

    free(buf);
    fclose(tmpf);

cleanup_all:
    free(line);
    spatial_grid_free(grid);
    free(neurons);
    gzclose(f);
}

/* ---------- zlib streaming writer helpers (for serialization) ---------- */
static void zwrite_bytes(z_stream *zs, FILE *fp, unsigned char *outbuf, const void *data, size_t len) {
    const unsigned char *p = (const unsigned char*)data;
    size_t remain = len;
    while (remain > 0) {
        uInt chunk = (remain > (size_t)UINT_MAX) ? UINT_MAX : (uInt)remain;
        zs->next_in = (Bytef*)p;
        zs->avail_in = chunk;
        while (zs->avail_in > 0) {
            zs->next_out = outbuf;
            zs->avail_out = (uInt)(1<<20);
            deflate(zs, Z_NO_FLUSH);
            size_t written = (size_t)((1<<20) - zs->avail_out);
            if (written) fwrite(outbuf, 1, written, fp);
        }
        p += chunk;
        remain -= chunk;
    }
}
static void zwrite_u32(z_stream *zs, FILE *fp, unsigned char *outbuf, uint32_t v) {
    uint32_t le = v;
    zwrite_bytes(zs, fp, outbuf, &le, sizeof(le));
}
static void zwrite_u64(z_stream *zs, FILE *fp, unsigned char *outbuf, uint64_t v) {
    uint64_t le = v;
    zwrite_bytes(zs, fp, outbuf, &le, sizeof(le));
}
static void zwrite_f32(z_stream *zs, FILE *fp, unsigned char *outbuf, float f) {
    float v = f;
    zwrite_bytes(zs, fp, outbuf, &v, sizeof(v));
}
static void zwrite_block(z_stream *zs, FILE *fp, unsigned char *outbuf, const void *data, size_t len) {
    zwrite_bytes(zs, fp, outbuf, data, len);
}

/* ---------- Serialization (explicit field writes + synapse block) ---------- */
static void serialize_nemu(const char *path, Cluster *clusters, size_t n_clusters) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror("fopen"); return; }

    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    if (deflateInit(&zs, 3) != Z_OK) {
        fprintf(stderr, "deflateInit failed\n");
        fclose(fp);
        return;
    }
    unsigned char outbuf[1 << 20];

    zwrite_block(&zs, fp, outbuf, MAGIC, 4);
    zwrite_u32(&zs, fp, outbuf, (uint32_t)VERSION);
    zwrite_u64(&zs, fp, outbuf, (uint64_t)n_clusters);

    for (size_t cid = 0; cid < n_clusters; cid++) {
        Cluster *c = &clusters[cid];
        zwrite_u32(&zs, fp, outbuf, (uint32_t)cid);
        zwrite_u32(&zs, fp, outbuf, (uint32_t)c->n_neurons);

        for (size_t i = 0; i < c->n_neurons; i++) {
            Neuron *n = &c->neurons[i];
            zwrite_u64(&zs, fp, outbuf, n->root_id);
            zwrite_f32(&zs, fp, outbuf, n->x);
            zwrite_f32(&zs, fp, outbuf, n->y);
            zwrite_f32(&zs, fp, outbuf, n->z);
            zwrite_u32(&zs, fp, outbuf, n->type_id);
            zwrite_u32(&zs, fp, outbuf, n->syn_in);
            zwrite_u32(&zs, fp, outbuf, n->syn_out);

            /* neighbors */
            zwrite_u64(&zs, fp, outbuf, n->n_neighbors);
            for (uint64_t j = 0; j < n->n_neighbors; j++)
                zwrite_u64(&zs, fp, outbuf, n->neighbors[j]);

            /* no per-neuron synapse list in this design */
            zwrite_u64(&zs, fp, outbuf, (uint64_t)0); /* n_out_synapses = 0 */
        }
    }

    /* Write synapse event block */
    zwrite_u64(&zs, fp, outbuf, global_syn_event_count);
    if (global_syn_event_count > 0 && global_syn_temp_path) {
        FILE *tmpf = fopen(global_syn_temp_path, "rb");
        if (!tmpf) {
            fprintf(stderr, "warning: cannot open synapse temp file %s: %s\n", global_syn_temp_path, strerror(errno));
        } else {
            /* stream in chunks */
            size_t chunk_events = (1 << 14);
            SynEvent *chunk = malloc(chunk_events * sizeof(SynEvent));
            if (!chunk) { perror("malloc"); fclose(tmpf); }
            else {
                uint64_t left = global_syn_event_count;
                while (left > 0) {
                    size_t toread = (left > chunk_events) ? (size_t)chunk_events : (size_t)left;
                    size_t r = fread(chunk, sizeof(SynEvent), toread, tmpf);
                    if (r != toread) {
                        fprintf(stderr, "warning: short read from syn temp file (expected %zu got %zu)\n", toread, r);
                    }
                    /* write each event as fields compatible with zwrite_ helpers */
                    for (size_t j = 0; j < r; ++j) {
                        zwrite_u32(&zs, fp, outbuf, chunk[j].voxel);
                        zwrite_f32(&zs, fp, outbuf, chunk[j].pre_x);
                        zwrite_f32(&zs, fp, outbuf, chunk[j].pre_y);
                        zwrite_f32(&zs, fp, outbuf, chunk[j].pre_z);
                        zwrite_f32(&zs, fp, outbuf, chunk[j].post_x);
                        zwrite_f32(&zs, fp, outbuf, chunk[j].post_y);
                        zwrite_f32(&zs, fp, outbuf, chunk[j].post_z);
                        zwrite_f32(&zs, fp, outbuf, chunk[j].size);
                    }
                    left -= r;
                }
                free(chunk);
            }
            fclose(tmpf);
        }
    }

    /* finish deflate stream */
    zs.next_in = NULL;
    zs.avail_in = 0;
    int ret;
    do {
        zs.next_out = outbuf;
        zs.avail_out = (uInt)sizeof(outbuf);
        ret = deflate(&zs, Z_FINISH);
        size_t wrote = sizeof(outbuf) - (size_t)zs.avail_out;
        if (wrote) fwrite(outbuf, 1, wrote, fp);
    } while (ret == Z_OK);

    deflateEnd(&zs);
    fclose(fp);
}

/* ---------- Pipeline loaders (unchanged except synapse behavior) ---------- */

static NeuronTable *load_neurons_from_cell_stats(const char *path) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); exit(1); }
    NeuronTable *table = table_create();
    char *line = malloc(LINE_BUFSZ);
    if (!line) { perror("malloc"); exit(1); }

    if (!gz_getline(f, line, LINE_BUFSZ)) { gzclose(f); free(line); return table; }
    size_t ncols = 0;
    char **hdr = split_csv_line(line, &ncols);

    int idx_root = find_field_index(hdr, ncols, "root_id", "root_id");
    int idx_syn_in = find_field_index(hdr, ncols, "syn_in", "syn_in");
    int idx_syn_out = find_field_index(hdr, ncols, "syn_out", "syn_out");

    free_csv_fields(hdr, ncols);

    while (gz_getline(f, line, LINE_BUFSZ)) {
        size_t cnt = 0;
        char **cols = split_csv_line(line, &cnt);
        if (cnt == 0) { free_csv_fields(cols, cnt); continue; }
        if (idx_root >= 0 && idx_root < (int)cnt) {
            uint64_t rid = strtoull(cols[idx_root], NULL, 10);
            if (rid == 0) { free_csv_fields(cols, cnt); continue; }
            Neuron *n = calloc(1, sizeof(Neuron));
            if (!n) { perror("calloc"); exit(1); }
            n->root_id = rid;
            n->type_id = 0;
            n->x = n->y = n->z = 0.0f;
            n->syn_in = 0;
            n->syn_out = 0;
            n->n_neighbors = 0;
            n->neighbors = NULL;
            n->n_out_synapses = 0;
            n->synapse_offset = UINT64_MAX;
            n->out_synapses = NULL;
            if (idx_syn_in >= 0 && idx_syn_in < (int)cnt) n->syn_in = (uint32_t)atoi(cols[idx_syn_in]);
            if (idx_syn_out >= 0 && idx_syn_out < (int)cnt) n->syn_out = (uint32_t)atoi(cols[idx_syn_out]);
            table_put(table, rid, n);
        }
        free_csv_fields(cols, cnt);
    }

    free(line);
    gzclose(f);
    return table;
}

static void load_cell_types_into_table(const char *path, NeuronTable *table, TypeMap *types) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); return; }
    char *line = malloc(LINE_BUFSZ);
    if (!line) { perror("malloc"); return; }
    gz_getline(f, line, LINE_BUFSZ);
    size_t ncols = 0;
    char **hdr = split_csv_line(line, &ncols);

    int idx_root = find_field_index(hdr, ncols, "root_id", "root_id");
    int idx_type = -1;
    for (size_t i = 0; i < ncols; ++i) {
        if (hdr[i] && (strstr(hdr[i], "primary_type") || strstr(hdr[i], "type") || strstr(hdr[i], "primary"))) {
            idx_type = (int)i; break;
        }
    }
    if (idx_type < 0) idx_type = find_field_index(hdr, ncols, "primary_type", "primary_type");

    free_csv_fields(hdr, ncols);

    while (gz_getline(f, line, LINE_BUFSZ)) {
        size_t cnt = 0;
        char **cols = split_csv_line(line, &cnt);
        if (cnt == 0) { free_csv_fields(cols, cnt); continue; }
        if (idx_root >= 0 && idx_root < (int)cnt && idx_type >= 0 && idx_type < (int)cnt) {
            uint64_t rid = strtoull(cols[idx_root], NULL, 10);
            Neuron *n = table_get(table, rid);
            if (n) {
                char *typetext = cols[idx_type];
                if (typetext && typetext[0]) {
                    uint32_t tid = typemap_get_or_add(types, typetext);
                    n->type_id = tid;
                }
            }
        }
        free_csv_fields(cols, cnt);
    }

    free(line);
    gzclose(f);
}

static void load_coordinates_into_table(const char *path, NeuronTable *table) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); return; }
    char *line = malloc(LINE_BUFSZ);
    if (!line) { perror("malloc"); return; }
    gz_getline(f, line, LINE_BUFSZ);
    size_t ncols = 0;
    char **hdr = split_csv_line(line, &ncols);

    int idx_root = find_field_index(hdr, ncols, "root_id", "root_id");
    int idx_pos = -1;
    for (size_t i = 0; i < ncols; ++i) {
        if (hdr[i] && (strstr(hdr[i], "position") || strstr(hdr[i], "pos"))) { idx_pos = (int)i; break; }
    }
    free_csv_fields(hdr, ncols);

    while (gz_getline(f, line, LINE_BUFSZ)) {
        size_t cnt = 0;
        char **cols = split_csv_line(line, &cnt);
        if (cnt == 0) { free_csv_fields(cols, cnt); continue; }
        if (idx_root >= 0 && idx_root < (int)cnt && idx_pos >= 0 && idx_pos < (int)cnt) {
            uint64_t rid = strtoull(cols[idx_root], NULL, 10);
            Neuron *n = table_get(table, rid);
            if (n) {
                float x,y,z;
                parse_position_field(cols[idx_pos], &x, &y, &z);
                n->x = x; n->y = y; n->z = z;
            }
        }
        free_csv_fields(cols, cnt);
    }

    free(line);
    gzclose(f);
}

static void load_connections_into_table(const char *path, NeuronTable *table) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); return; }
    char *line = malloc(LINE_BUFSZ);
    if (!line) { perror("malloc"); return; }
    if (!gz_getline(f, line, LINE_BUFSZ)) { free(line); gzclose(f); return; }
    size_t ncols = 0;
    char **hdr = split_csv_line(line, &ncols);

    int idx_pre = -1, idx_post = -1;
    for (size_t i = 0; i < ncols; ++i) {
        if (!hdr[i]) continue;
        if (strstr(hdr[i], "pre_root_id")) idx_pre = (int)i;
        if (strstr(hdr[i], "post_root_id")) idx_post = (int)i;
    }
    if (idx_pre < 0) idx_pre = find_field_index(hdr, ncols, "pre_root_id", "pre_root_id");
    if (idx_post < 0) idx_post = find_field_index(hdr, ncols, "post_root_id", "post_root_id");

    free_csv_fields(hdr, ncols);

    while (gz_getline(f, line, LINE_BUFSZ)) {
        size_t cnt = 0;
        char **cols = split_csv_line(line, &cnt);
        if (cnt == 0) { free_csv_fields(cols, cnt); continue; }
        if (idx_pre >= 0 && idx_pre < (int)cnt && idx_post >= 0 && idx_post < (int)cnt) {
            uint64_t pre = strtoull(cols[idx_pre], NULL, 10);
            uint64_t post = strtoull(cols[idx_post], NULL, 10);
            if (pre == 0 || post == 0) { free_csv_fields(cols, cnt); continue; }
            Neuron *a = table_get(table, pre);
            Neuron *b = table_get(table, post);
            if (a && b) {
                neuron_add_neighbor(a, post);
                neuron_add_neighbor(b, pre);
            }
        }
        free_csv_fields(cols, cnt);
    }

    free(line);
    gzclose(f);
}

/* ---------- Build clusters grouped by type_id (no per-neuron synapses) ---------- */
static Cluster *build_clusters_from_table(NeuronTable *table, size_t *out_n_clusters) {
    uint32_t max_type = 0;
    for (size_t i = 0; i < HASH_SIZE; ++i) {
        Node *n = table->buckets[i];
        while (n) {
            if (n->value->type_id > max_type) max_type = n->value->type_id;
            n = n->next;
        }
    }
    size_t n_clusters = (size_t)max_type + 1;
    if (n_clusters == 0) n_clusters = 1;

    Cluster *clusters = calloc(n_clusters, sizeof(Cluster));
    if (!clusters) { perror("calloc"); exit(1); }
    for (size_t i = 0; i < n_clusters; ++i) clusters[i].id = (uint32_t)i;

    for (size_t i = 0; i < HASH_SIZE; ++i) {
        Node *node = table->buckets[i];
        while (node) {
            Neuron *orig = node->value;
            uint32_t t = orig->type_id;
            if (t >= n_clusters) t = 0;
            Cluster *c = &clusters[t];

            c->neurons = realloc(c->neurons, (c->n_neurons + 1) * sizeof(Neuron));
            if (!c->neurons) { perror("realloc"); exit(1); }
            Neuron *dst = &c->neurons[c->n_neurons++];

            /* copy base fields */
            dst->root_id = orig->root_id;
            dst->type_id = orig->type_id;
            dst->x = orig->x; dst->y = orig->y; dst->z = orig->z;
            dst->syn_in = orig->syn_in; dst->syn_out = orig->syn_out;
            dst->n_neighbors = orig->n_neighbors;
            if (orig->n_neighbors) {
                dst->neighbors = malloc(orig->n_neighbors * sizeof(uint64_t));
                if (!dst->neighbors) { perror("malloc"); exit(1); }
                memcpy(dst->neighbors, orig->neighbors, orig->n_neighbors * sizeof(uint64_t));
            } else dst->neighbors = NULL;

            /* in this design we do not attach out_synapses to neurons */
            dst->n_out_synapses = 0;
            dst->synapse_offset = UINT64_MAX;
            dst->out_synapses = NULL;

            node = node->next;
        }
    }

    *out_n_clusters = n_clusters;
    return clusters;
}

#ifdef SERIALIZE_MAIN
/* ---------- Main ---------- */
int main(int argc, char **argv) {
    const char *out = "nemu/serialized.nemu";
    if (argc > 1) out = argv[1];

    fprintf(stderr, "Loading neurons from cell_stats.csv.gz...\n");
    NeuronTable *table = load_neurons_from_cell_stats("cell_stats.csv.gz");

    fprintf(stderr, "Loading cell types...\n");
    TypeMap *types = typemap_create();
    load_cell_types_into_table("consolidated_cell_types.csv.gz", table, types);

    fprintf(stderr, "Loading coordinates...\n");
    load_coordinates_into_table("coordinates.csv.gz", table);

    fprintf(stderr, "Loading connections...\n");
    load_connections_into_table("connections_princeton_no_threshold.csv.gz", table);

    fprintf(stderr, "Streaming synapse events (no neuron assignment)...\n");
    stream_synapses_into_table("fafb_v783_princeton_synapse_table.csv.gz", table);

    fprintf(stderr, "Building clusters...\n");
    size_t n_clusters = 0;
    Cluster *clusters = build_clusters_from_table(table, &n_clusters);

    fprintf(stderr, "Serializing %zu clusters + %" PRIu64 " synapse events...\n", n_clusters, global_syn_event_count);
    system("mkdir -p nemu");

    serialize_nemu(out, clusters, n_clusters);

    fprintf(stderr, "Done -> %s\n", out);

    /* cleanup: free memory & keep synapse temp file on disk for inspection if desired */
    for (size_t i = 0; i < n_clusters; ++i) {
        for (size_t j = 0; j < clusters[i].n_neurons; ++j) {
            free(clusters[i].neurons[j].neighbors);
        }
        free(clusters[i].neurons);
    }
    free(clusters);

    /* free neuron table nodes */
    for (size_t i = 0; i < HASH_SIZE; ++i) {
        Node *n = table->buckets[i];
        while (n) {
            Node *nx = n->next;
            free(n->value); /* neuron struct allocated in load_neurons_from_cell_stats */
            free(n);
            n = nx;
        }
    }
    free(table->buckets);
    free(table);

    /* Note: we deliberately *do not* unlink global_syn_temp_path so you can inspect it.
       Set SYN_KEEP_TEMP=0 in env to auto-remove (default behavior kept here is to keep it). */
    const char *keep = getenv("SYN_KEEP_TEMP");
    if (!keep || atoi(keep) == 0) {
        /* default: keep file for debugging; if user sets SYN_KEEP_TEMP=1 we remove it */
    } else {
        if (global_syn_temp_path) {
            unlink(global_syn_temp_path);
            free(global_syn_temp_path);
            global_syn_temp_path = NULL;
        }
    }

    return 0;
}
#endif