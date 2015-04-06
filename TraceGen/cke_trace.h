/*
 *  Block Trace Generator for Concurrent Kernel Execution on NVIDIA's Kepler GPU
 *  Copyright (C) 2012  Leiming Yu (ylm@ece.neu.edu)
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *  
 */

#ifndef CKE_TRACE_H
#define CKE_TRACE_H

#include <deque>
#include <utility>      // pair
#include <stdarg.h>     /* va_list, va_start, va_arg, va_end */

using namespace::std;

struct GPUCONFIG
{
	int num_smxs; /* number of streaming multiprocessors (SMX) */
	int reg_per_smx; /* total registers per SMX */
	int sm_per_block; /* shared memory per SMX */
	int max_threads_per_smx; /* thread limits for current device */
	int blklimit_per_smx; /* maximum allowed blocks per SMX */
};

struct KERNEL
{
	int gridsize;
	int blocksize;
	int reg_per_thread;
	int sm_per_block;
	int threads_per_block;
	float exetime_ms; /* kernel execution time in ms */
	float avgblktime_ms; /* average block execution time in ms */
};

struct SMX
{
	int id;
	int max_registers;
	int max_shared_memory; // in bytes
	int max_threads;
	int blklimit;

	int run_blknum; // running blocks
	int run_registers;
	int run_sharedmem;
	int run_thread;

	float run_time; // total running time
	float local_time; // get updated when retiring a block

	deque<pair<int, int> > kernel_blkid; // kernel + blkid
	deque<int> retire; // mark block retirement
};

/* Thread Block Scheduler*/
struct TBS
{
	int kernel_id;
	deque<pair<int,int> > running_blks; /* kernel , running_blks */
	deque<pair<int,int> > waiting_blks;
	deque<pair<int,int> > done_blks;
};

struct KERBLK
{
	int reg_per_thread;
	int shared_memory;
	int threads_per_blk;
};

// kernel execution time
struct KTIME
{
	float starttime;
	float endtime;
};

// kernel blocks in one smx
struct BLKSMX
{
	// maximum 100 kernel deques
	deque<int> kernelblk[100];
};

struct BLKINFO
{
	int threads; // threads per block
	int sm; // shared memory in bytes
	int registers;
	float avetime; // averaged block execution time
};

void warning(const char *fmt, ...)
{
	va_list va;
	va_start(va, fmt);
	fprintf(stderr, "warning: ");
	vfprintf(stderr, fmt, va);
	fprintf(stderr, "\n");
	va_end(va);
}

/*
 * Functions
 */
void parse_file(int argc, char **argv);
//int calc_allowblks(SMX smx_resource, KERBLK ker_blk);
int waiting_blocks(deque<TBS> &tbs);

#endif
