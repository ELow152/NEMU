#ifndef SPILL_H
#define SPILL_H

#include <stdio.h>
#include <stddef.h>
#include <zlib.h>

typedef struct SpillReader SpillReader;
typedef struct SpillWriter SpillWriter;

/* Reader API */
SpillReader *spill_reader_open_file(const char *path);
int spill_reader_read_exact(SpillReader *r, void *dst, size_t len);
void spill_reader_close(SpillReader *r);

/* Writer API */
SpillWriter *spill_writer_open_file(const char *path, int compression_level);
int spill_writer_write(SpillWriter *w, const void *data, size_t len);
int spill_writer_flush_finish(SpillWriter *w);
void spill_writer_close(SpillWriter *w);

#endif