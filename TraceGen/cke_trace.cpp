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

#include <iostream>
#include <sstream> /* istringstream*/
#include <fstream>
#include <string>
#include <vector>

#include <deque>
#include <limits>
#include <stdlib.h> /* exit */
#include <algorithm> /* min_element */
#include <string.h> /* memset */
#include <iomanip>      /* setw */
#include <utility>      // pair

#include "cke_trace.h"

using namespace std;

// max vector column number
#define MAXCOL 100000


int num_kernels;

GPUCONFIG gpuconfig; /* targeted GPU info */
deque <KERNEL> kernelInfo; /* detailed kernel info */
deque <SMX> smxs_status; /* smx resources utilization */

//deque <TBS> tbs;



int find_targetSMX(deque<SMX> &smxs)
{
	int smxnum = smxs.size();

	// find the most available smx by sorting the number of running blocks
	int target = 0;
	int small = smxs[0].run_blknum;

	for(int i = 1; i < smxnum; i++)
	{
		if(small > smxs[i].run_blknum)
		{
			target = i;
			small = smxs[i].run_blknum;
		}
	}

	return target;
}

int check_resources(deque<BLKINFO> &blkinfo, deque<SMX> &smxs_status, int smx_id,
		int kern_id)
{
	// current block info
	float blktime = blkinfo[kern_id].avetime;

	int add_sharedmem = blkinfo[kern_id].sm;
	int add_regs = blkinfo[kern_id].registers;
	int add_threadnum = blkinfo[kern_id].threads;

	// current smx status
	int blklimit = smxs_status[smx_id].blklimit;
	int running_blks = smxs_status[smx_id].run_blknum;
	int running_regs = smxs_status[smx_id].run_registers;
	int running_sm = smxs_status[smx_id].run_sharedmem;
	int running_threads = smxs_status[smx_id].run_thread;

	// update the resource
	int cur_sharedmem = running_sm + add_sharedmem;
	int cur_reg = running_regs + add_regs;
	int cur_threads = running_threads + add_threadnum;
	int cur_blknum = running_blks + 1;

	// limitations
	int max_sm = smxs_status[smx_id].max_shared_memory;
	int max_reg = smxs_status[smx_id].max_registers;
	int max_threads = smxs_status[smx_id].max_threads;
	int max_blocks = smxs_status[smx_id].blklimit;

	int avail = 0;

	// make judgment
	if( cur_sharedmem < max_sm && cur_reg < max_reg && cur_threads < max_threads
			&& cur_blknum < max_blocks)
	{
		avail = 1;
	}

	return avail;
}

int main(int argc, char *argv[])
{
	//------------------------------------------------------------------------//
	//	Read input
	//------------------------------------------------------------------------//
	parse_file(argc, argv);

	//------------------------------------------------------------------------//
	//	Initialize smx for the gpua:	smxs_status
	//------------------------------------------------------------------------//
	for (int i = 0; i < gpuconfig.num_smxs; i++)
	{
		SMX cur_smx;

		cur_smx.id = i;
		cur_smx.max_threads = gpuconfig.max_threads_per_smx;
		cur_smx.max_registers = gpuconfig.reg_per_smx;
		cur_smx.max_shared_memory = gpuconfig.sm_per_block;
		cur_smx.blklimit = gpuconfig.blklimit_per_smx;

		cur_smx.run_time = 0.f; // total running time on the smx
		cur_smx.local_time = 0.f; // update running time

		cur_smx.run_blknum = 0; // running blocks on the smx
		cur_smx.run_registers = 0;
		cur_smx.run_sharedmem = 0;
		cur_smx.run_thread = 0;

		smxs_status.push_back(cur_smx);
	}

	//------------------------------------------------------------------------//
	//	Initialize Thread Block Scheduler (TBS)
	//------------------------------------------------------------------------//
	vector<vector<int> > tbs;

	for(int k = 0; k < kernelInfo.size(); k++)// go through each registered kernels
	{
		vector<int> kerblk;

		for(int i = 0; i < kernelInfo[k].gridsize; i++)	// at current kernel, register the block id
		{
			kerblk.push_back(i);
		}
		tbs.push_back(kerblk);
	}

	if(tbs.empty())
	{
		warning("No kernel has been scheduled. Please check the input file.");
		exit(EXIT_FAILURE);
	}

	//------------------- allocate block info for each kernel -----------------//
	deque<BLKINFO> blkinfo;

	for(int k = 0; k < kernelInfo.size(); k++)
	{
		BLKINFO cur_blk_info;

		cur_blk_info.avetime = 1.f; // assume 1 ms
		cur_blk_info.sm = kernelInfo[k].sm_per_block;
		cur_blk_info.registers = kernelInfo[k].reg_per_thread;
		cur_blk_info.threads = kernelInfo[k].blocksize;

		blkinfo.push_back(cur_blk_info);
	}



	//------------------------------------------------------------------------//
	// Schedule blocks on the GPU
	//------------------------------------------------------------------------//

	float avgblktime = 1.f; // assume the averaged block execution time

	for(int k = 0; k < tbs.size(); k++) // go through each kernel
	{
		int waitblknum = tbs[k].size(); // waiting blocks in current kernel

		for(int b = 0; b < waitblknum; b++) // assign each block to most available smx in RR fashion
		{
			// FIXME: may need other info to sort
			int smx_id = find_targetSMX(smxs_status); // find the most free smx by sorting the running blocks

			pair<int, int> tmp;
			tmp = make_pair(k, b);

			// check whether enough resources available on the smx
			int avail = check_resources(blkinfo, smxs_status, smx_id, k);

			if (avail)// allocate the block on this smx
			{
				// update the smx resources
				smxs_status[smx_id].run_blknum += 1;
				smxs_status[smx_id].run_registers += blkinfo[k].registers;
				smxs_status[smx_id].run_sharedmem += blkinfo[k].sm;
				smxs_status[smx_id].run_thread += blkinfo[k].threads;

				// update the smx timing
				smxs_status[smx_id].run_time += blkinfo[k].avetime;
				smxs_status[smx_id].local_time += blkinfo[k].avetime;

				// record the kernel and block id
				smxs_status[smx_id].kernel_blkid.push_back(tmp);

				smxs_status[smx_id].retire.push_back(0);
			}
			else
			{	// when the target smx is fully occupied
				int smx_blks = smxs_status[smx_id].kernel_blkid.size();

				// consider retiring a running block in the smx
				// and allocate a new block
				for(int id = 0; id < smx_blks; id++)
				{
					int retire = smxs_status[smx_id].retire[id]; // retire or not

					if (!retire)
					{
						// ---------------   retire current blk ------------- //
						smxs_status[smx_id].retire[id] = 1;

						// update status/resources
						smxs_status[smx_id].run_blknum -= 1;
						smxs_status[smx_id].run_registers -= blkinfo[k].registers;
						smxs_status[smx_id].run_sharedmem -= blkinfo[k].sm;
						smxs_status[smx_id].run_thread -= blkinfo[k].threads;

						// update timing
						smxs_status[smx_id].local_time -= blkinfo[k].avetime;
						smxs_status[smx_id].run_time += blkinfo[k].avetime;

						//---------- check other smxs for retiring -----------//
						for(int sid = 0; sid < gpuconfig.num_smxs; sid++)
						{
							if(sid != smx_id)
							{
								// list of blocks in this smx
								int sid_smxblks = smxs_status[sid].kernel_blkid.size();

								// FIXME: consider multiple kernel execution
								// use time to determine whether retire or not
								// since only one kernel is executed, retiring is a sure thing
								for(int lid = 0; lid < sid_smxblks; lid++)
								{
									// find the first non-tired block
									int retire_sid = smxs_status[sid].retire[lid];

									if(!retire_sid)
									{
										// mark it retired
										smxs_status[sid].retire[lid] = 1;

										// update status/resources
										smxs_status[sid].run_blknum -= 1;
										smxs_status[sid].run_registers -= blkinfo[k].registers;
										smxs_status[sid].run_sharedmem -= blkinfo[k].sm;
										smxs_status[sid].run_thread -= blkinfo[k].threads;

										// update timing
										smxs_status[sid].local_time -= blkinfo[k].avetime;
									}
									break; // go to the next smx
								}

							}
						} // end of check other smx

						// register this block, in the case when one block is retired
						smxs_status[smx_id].kernel_blkid.push_back(tmp);
						smxs_status[smx_id].retire.push_back(0);

						break;
					} // stop searching the current smx
				}
			} // end of if-else
		}
	}

	//------------------------------------------------------------------------//
	// end of scheduling
	//------------------------------------------------------------------------//

	// print trace
	cout << endl;

	for(int i = 0; i < gpuconfig.num_smxs; i++)
	{
		cout << "smx[" << i <<"]: ";
		int blknum = smxs_status[i].kernel_blkid.size();
		for(int j = 0; j < blknum; j++)
		{
			cout << smxs_status[i].kernel_blkid[j].second << " ";
		}
		cout << endl;
	}

}




/*
 * read input file
 */
void parse_file(int argc, char **argv)
{
	if (argc !=2)
	{
		cout << "Usage: " << argv[0] << " <filename>\n";
		exit(EXIT_FAILURE);
	}
	else
	{
		ifstream infile(argv[1]);
		int linenumber = 0;
		string readline;
		if(infile.is_open())
		{
			while (getline(infile, readline))
			{
				linenumber++;
				/*
				 * GPU Info
				 */
				if (linenumber == 3)
				{
					istringstream gpuinfo(readline);
					gpuinfo >> gpuconfig.num_smxs
						>> gpuconfig.reg_per_smx
						>> gpuconfig.sm_per_block
						>> gpuconfig.max_threads_per_smx
						>> gpuconfig.blklimit_per_smx;
				}

				if (linenumber == 6)
				{
					istringstream kernum(readline);
					kernum >> num_kernels;
				}

				// read kernel info
				KERNEL ker;
				if (linenumber >= 10)
				{
					// check empty line
					if(!readline.empty())
					{
						istringstream kerinfo(readline);
						kerinfo >> ker.gridsize >> ker.blocksize
								>> ker.reg_per_thread >> ker.sm_per_block
								>> ker.exetime_ms;
						kernelInfo.push_back(ker);

					}
				}
			}
		}
		else
		{
			cout << "Could not open file!\n";
			exit(EXIT_FAILURE);
		}

		if(num_kernels != kernelInfo.size())
		{
			warning("The number of CKE kernels doesn't match."
					"Please check the input file.");
			exit(EXIT_FAILURE);
		}

		infile.close();
	}
}


int waiting_blocks(deque<TBS> &tbs)
{
	int waitblk = 0;
	for(int k = 0; k < tbs.size(); k++)
	{
		waitblk += tbs[k].waiting_blks.size();
	}

	return waitblk;
}
