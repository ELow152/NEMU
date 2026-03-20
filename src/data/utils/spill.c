/* spill.c - implementation for spilling
neuronal information onto disk in .nemu format.
Basically a tiny runtime serializer.
 */
#define _POSIX_C_SOURCE 200809L

#include "spill.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <limits.h>

#define COMP_IN_CAP (1<<20)   /* 1 MiB compressed input */
#define DECOMP_CAP  (1<<20)   /* 1 MiB decompressed sliding buffer */
#define OUTBUF_CAP  (1<<20)   /* 1 MiB output buffer for writer */

/* --- SpillReader --- */
struct SpillReader {
    FILE *fp;
    z_stream zs;
    unsigned char *inbuf;
    unsigned char *outbuf;
    size_t out_cap;
    size_t out_start;
    size_t out_end;
    int finished;
};

static int sr_inflate_more(SpillReader *r) {
    if (r->finished) return 0;

    /* Only read new input if zlib has consumed previous input */
    if (r->zs.avail_in == 0) {
        size_t rlen = fread(r->inbuf, 1, COMP_IN_CAP, r->fp);
        if (rlen == 0) {
            if (ferror(r->fp)) return 0;
            r->finished = 1;
            return 0;
        }
        r->zs.next_in = r->inbuf;
        r->zs.avail_in = (uInt)rlen;
    }

    /* ensure space in output buffer */
    if (r->out_end == r->out_cap) {
        if (r->out_start > 0) {
            size_t used = r->out_end - r->out_start;
            memmove(r->outbuf, r->outbuf + r->out_start, used);
            r->out_start = 0;
            r->out_end = used;
        } else {
            size_t ncap = r->out_cap * 2;
            unsigned char *n = realloc(r->outbuf, ncap);
            if (!n) return 0;
            r->outbuf = n;
            r->out_cap = ncap;
        }
    }

    r->zs.next_out = r->outbuf + r->out_end;
    r->zs.avail_out = (uInt)(r->out_cap - r->out_end);

    int ret = inflate(&r->zs, Z_NO_FLUSH);
    if (ret == Z_STREAM_END) r->finished = 1;
    if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR) return 0;

    size_t produced = (r->out_cap - r->out_end) - r->zs.avail_out;
    r->out_end += produced;

    return (produced > 0) || (!r->finished);
}

SpillReader *spill_reader_open_file(const char *path) {
    SpillReader *r = calloc(1, sizeof(*r));
    if (!r) return NULL;

    r->fp = fopen(path, "rb");
    if (!r->fp) { free(r); return NULL; }

    r->inbuf = malloc(COMP_IN_CAP);
    r->outbuf = malloc(DECOMP_CAP);
    if (!r->inbuf || !r->outbuf) {
        fclose(r->fp);
        free(r->inbuf);
        free(r->outbuf);
        free(r);
        return NULL;
    }

    r->out_cap = DECOMP_CAP;
    r->out_start = 0;
    r->out_end = 0;
    r->finished = 0;

    memset(&r->zs, 0, sizeof(r->zs));

    if (inflateInit(&r->zs) != Z_OK) {
        fclose(r->fp);
        free(r->inbuf);
        free(r->outbuf);
        free(r);
        return NULL;
    }

    return r;
}

static int sr_ensure(SpillReader *r, size_t need) {
    while ((r->out_end - r->out_start) < need) {
        if (!sr_inflate_more(r)) return 0;
    }
    return 1;
}

int spill_reader_read_exact(SpillReader *r, void *dst, size_t len) {
    if (len == 0) return 1;

    unsigned char *p = dst;

    while (len > 0) {
        size_t avail = r->out_end - r->out_start;

        if (avail == 0) {
            if (!sr_inflate_more(r)) return 0;
            avail = r->out_end - r->out_start;
            if (avail == 0) return 0;
        }

        size_t take = avail < len ? avail : len;
        memcpy(p, r->outbuf + r->out_start, take);

        r->out_start += take;
        p += take;
        len -= take;
    }

    return 1;
}

int spill_reader_peek_bytes(SpillReader *r, void *dst, size_t len) {
    if (!sr_ensure(r, len)) return 0;
    memcpy(dst, r->outbuf + r->out_start, len);
    return 1;
}

void spill_reader_skip(SpillReader *r, size_t len) {
    if (len <= (r->out_end - r->out_start)) {
        r->out_start += len;
        return;
    }

    unsigned char tmp[4096];
    size_t left = len;

    while (left) {
        size_t n = left > sizeof(tmp) ? sizeof(tmp) : left;
        if (!spill_reader_read_exact(r, tmp, n)) return;
        left -= n;
    }
}

void spill_reader_close(SpillReader *r) {
    if (!r) return;

    inflateEnd(&r->zs);
    if (r->fp) fclose(r->fp);
    free(r->inbuf);
    free(r->outbuf);
    free(r);
}

/* --- SpillWriter --- */
struct SpillWriter {
    FILE *fp;
    z_stream zs;
    unsigned char *outbuf;
    size_t out_cap;
    int finished;
};

SpillWriter *spill_writer_open_file(const char *path, int compression_level) {
    SpillWriter *w = calloc(1, sizeof(*w));
    if (!w) return NULL;

    w->fp = fopen(path, "wb");
    if (!w->fp) { free(w); return NULL; }

    w->outbuf = malloc(OUTBUF_CAP);
    if (!w->outbuf) {
        fclose(w->fp);
        free(w);
        return NULL;
    }

    w->out_cap = OUTBUF_CAP;
    w->finished = 0;

    memset(&w->zs, 0, sizeof(w->zs));

    if (deflateInit(&w->zs, compression_level) != Z_OK) {
        fclose(w->fp);
        free(w->outbuf);
        free(w);
        return NULL;
    }

    return w;
}

int spill_writer_write(SpillWriter *w, const void *src, size_t len) {
    if (!w || !src) return 0;

    const unsigned char *p = src;
    size_t remain = len;

    while (remain) {
        uInt chunk = (remain > (size_t)UINT_MAX) ? UINT_MAX : (uInt)remain;

        w->zs.next_in = (Bytef*)p;
        w->zs.avail_in = chunk;

        while (w->zs.avail_in > 0) {
            w->zs.next_out = w->outbuf;
            w->zs.avail_out = (uInt)w->out_cap;

            if (deflate(&w->zs, Z_NO_FLUSH) != Z_OK) return 0;

            size_t wrote = w->out_cap - w->zs.avail_out;
            if (wrote) {
                if (fwrite(w->outbuf, 1, wrote, w->fp) != wrote) return 0;
            }
        }

        p += chunk;
        remain -= chunk;
    }

    return 1;
}

int spill_writer_write_u32(SpillWriter *w, uint32_t v) {
    return spill_writer_write(w, &v, sizeof(v));
}

int spill_writer_write_u64(SpillWriter *w, uint64_t v) {
    return spill_writer_write(w, &v, sizeof(v));
}

int spill_writer_write_f32(SpillWriter *w, float f) {
    return spill_writer_write(w, &f, sizeof(f));
}

int spill_writer_flush_finish(SpillWriter *w) {
    if (!w) return 0;

    int ret;
    do {
        w->zs.next_out = w->outbuf;
        w->zs.avail_out = (uInt)w->out_cap;

        ret = deflate(&w->zs, Z_FINISH);

        size_t wrote = w->out_cap - w->zs.avail_out;
        if (wrote) {
            if (fwrite(w->outbuf, 1, wrote, w->fp) != wrote) return 0;
        }
    } while (ret == Z_OK);

    deflateEnd(&w->zs);
    w->finished = 1;

    return 1;
}

void spill_writer_close(SpillWriter *w) {
    if (!w) return;

    if (!w->finished) spill_writer_flush_finish(w);
    if (w->fp) fclose(w->fp);

    free(w->outbuf);
    free(w);
}