/*
 * Copyright (C) 2020, Xilinx Inc - All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#define EN_SORT_PROFILE 0

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

// driver includes
#include "ert.h"

// host_src includes
#include "xclhal2.h"
#include "xclbin.h"

// lowlevel common include
#include "utils.h"

#define DSA64 1

#if defined(DSA64)
#include "xsort_hw_64.h"      
#else
#include "xsort_hw.h"
#endif

class PPHandle {
public:

xclDeviceHandle handle;
uint64_t cu_base_addr;
unsigned cu_index;
unsigned boHandle2;
uint64_t bo2devAddr;
char *bo2;
unsigned execHandle;
void *execData;
};

extern PPHandle* pphandle;

const int NUM_CLASS = 91;
const int CLASS_SIZE = 1917;
const int TOPK = 80;
const int TOPK_SORT = TOPK;

static int runHWSort(PPHandle * &pphandle,
int8_t *&sort_score, short *&sort_index, short *&sort_size, int8_t *&loc_data)
{	

   xclDeviceHandle handle = pphandle->handle;
   uint64_t cu_base_addr = pphandle->cu_base_addr;
   unsigned boHandle2 = pphandle->boHandle2;
   uint64_t bo2devAddr = pphandle->bo2devAddr;
   char *bo2 = pphandle->bo2;
   unsigned execHandle = pphandle->execHandle;
   void *execData =  pphandle->execData;
   
   try {

        auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd*>(execData);

        // Clear the command in case it was recycled
        size_t regmap_size = XSORT_CONTROL_ADDR_SCALAR_DATA/4 + 1; // regmap
        std::memset(ecmd2,0,(sizeof *ecmd2) + regmap_size);

        // Program the command packet header
        ecmd2->state = ERT_CMD_STATE_NEW;
        ecmd2->opcode = ERT_START_CU;
        ecmd2->count = 1 + regmap_size;  // cu_mask + regmap

        // Program the CU mask. One CU at index 0
		ecmd2->cu_mask = 0x2;

        // Program the register map
        ecmd2->data[XSORT_CONTROL_ADDR_AP_CTRL] = 0x0; // ap_start

#if TGT_DEVICE==u280
	//std::cout << "  u280 device--" << std::endl;
	uint64_t bo1devAddr_conf = 0x100000000ull + 0xa84bf0ull;
	uint64_t bo1devAddr_box = 0x100000000ull + 0xaaf560ull;
#else
	//std::cout << " u50 device--" << std::endl;
	uint64_t bo1devAddr_conf = 0x170000000ull + 0xa84bf0ull;
	uint64_t bo1devAddr_box = 0x170000000ull + 0xaaf560ull;
#endif

        ecmd2->data[XSORT_CONTROL_ADDR_inConf_DATA/4] = bo1devAddr_conf & 0xFFFFFFFF; // input
        ecmd2->data[XSORT_CONTROL_ADDR_inConf_DATA/4 + 1] = (bo1devAddr_conf >> 32) & 0xFFFFFFFF; // input
        ecmd2->data[XSORT_CONTROL_ADDR_inBox_DATA/4] = bo1devAddr_box & 0xFFFFFFFF; // input
        ecmd2->data[XSORT_CONTROL_ADDR_inBox_DATA/4 + 1] = (bo1devAddr_box >> 32) & 0xFFFFFFFF; // input
        ecmd2->data[XSORT_CONTROL_ADDR_OUTPUT_SCORE_DATA/4] = bo2devAddr & 0xFFFFFFFF; // output score
        ecmd2->data[XSORT_CONTROL_ADDR_OUTPUT_SCORE_DATA/4 + 1] = (bo2devAddr >> 32) & 0xFFFFFFFF; // output score

	ecmd2->data[XSORT_CONTROL_ADDR_SCALAR_DATA/4] = CLASS_SIZE;


#if EN_SORT_PROFILE
        auto start_k2= std::chrono::system_clock::now();
#endif

 	int ret;
  	if ((ret = xclExecBuf(handle, execHandle)) != 0) {
    		std::cout << "Unable to trigger SORT, error:" << ret << std::endl;
    		return ret;
  	}
  	do {
    		ret = xclExecWait(handle, 1000);
    		if (ret == 0) {
      			std::cout << "SORT Task Time out, state =" << ecmd2->state << "cu_mask = " << ecmd2->cu_mask << std::endl;

    		} else if (ecmd2->state == ERT_CMD_STATE_COMPLETED) {
      			
      			break;
    		}
  	} while (1);


#if EN_SORT_PROFILE
        auto end_k2= std::chrono::system_clock::now();
        auto difft_k2= end_k2-start_k2;
        auto value_k2 = std::chrono::duration_cast<std::chrono::microseconds>(difft_k2);
        std::cout << "Kernel Execution " << value_k2.count() << " us\n";

        auto start_k3= std::chrono::system_clock::now();
#endif

	const int classNum_SizeBuff_align = ((NUM_CLASS + 8 - 1)/8)*8;//out_port_width/16=8
	const int BoxBuff_align = ((4*CLASS_SIZE + 16 - 1)/16)*16;//out_port_width/16=8
	long int outSize_bytes = NUM_CLASS*TOPK*sizeof(char) + (NUM_CLASS*TOPK)*sizeof(short) + (classNum_SizeBuff_align)*sizeof(short) + BoxBuff_align*sizeof(char);

        //Get the output;
        if(xclSyncBO(handle, boHandle2, XCL_BO_SYNC_BO_FROM_DEVICE, outSize_bytes, 0)) {
            return 1;
        }

#if EN_SORT_PROFILE
        auto end_k3= std::chrono::system_clock::now();
        auto difft_k3= end_k3-start_k3;
        auto value_k3 = std::chrono::duration_cast<std::chrono::microseconds>(difft_k3);
        std::cout << "synbo Out " << value_k3.count() << " us\n";
#endif

        sort_score = (int8_t *)bo2;
	sort_index = (short *)(bo2+(TOPK*NUM_CLASS));;
	sort_size = (short *)(bo2+(3*TOPK*NUM_CLASS));
	loc_data = (int8_t *)(bo2+(3*TOPK*NUM_CLASS)+classNum_SizeBuff_align*2);

    }
 

    catch (std::exception const& e)
    {
        std::cout << "Exception: " << e.what() << "\n";
        std::cout << "FAILED TEST\n";
        return 1;
    }

//    xclCloseContext(handle, xclbinId, cu_index);

    return 0;

}

static int hw_sort_init(
  PPHandle * &pphandle,
  std::string bitstreamFile)
{
  PPHandle *my_handle = new PPHandle;
  pphandle = my_handle = (PPHandle *)my_handle;

  unsigned index = 0;
  std::string halLogfile;
  unsigned cu_index = 1;
  //cout <<"sort xclbin"<< bitstreamFile << endl;

  xclDeviceHandle handle;
  uint64_t cu_base_addr = 0;
  uuid_t xclbinId;
  int first_mem = -1;
  bool ret_initXRT=0;
  bool ret_firstmem=0;
  bool ret_runkernel=0;
  bool ret_checkDevMem=0;


  if (initXRT(bitstreamFile.c_str(), index, halLogfile.c_str(), handle, cu_index, cu_base_addr, first_mem, xclbinId))
      ret_initXRT=1;

  if(xclOpenContext(handle, xclbinId, cu_index, true))
    throw std::runtime_error("Cannot create context");

  // Allocate the device memory
  const int classNum_SizeBuff_align = ((NUM_CLASS + 8 - 1)/8)*8;//out_port_width/16=8
  const int BoxBuff_align = ((4*CLASS_SIZE + 16 - 1)/16)*16;//out_port_width/8=16
  long int outSize_bytes = NUM_CLASS*TOPK*sizeof(char) + (NUM_CLASS*TOPK)*sizeof(short) + (classNum_SizeBuff_align)*sizeof(short) + BoxBuff_align*sizeof(char);
  unsigned boHandle2 = xclAllocBO(handle, outSize_bytes, XCL_BO_DEVICE_RAM, 9);   // output score

  // Create the mapping to the host memory
  char *bo2 = (char*)xclMapBO(handle, boHandle2, false);

  // Get & check the device memory address
  xclBOProperties p;
  uint64_t bo2devAddr = !xclGetBOProperties(handle, boHandle2, &p) ? p.paddr : -1;
  
  if( (bo2devAddr == (uint64_t)(-1)) ){
      ret_checkDevMem=1;
  }
  
  //thread_local static 
  unsigned execHandle = 0;
  //thread_local static 
  void *execData = nullptr;
    
  if(execHandle == 0) execHandle = xclAllocBO(handle, 4096, xclBOKind(0), (1<<31));
  if(execData == nullptr) execData = xclMapBO(handle, execHandle, true);

  my_handle->handle = handle;
  my_handle->cu_base_addr = cu_base_addr;
  my_handle->cu_index = cu_index;
  my_handle->boHandle2 = boHandle2;
  my_handle->bo2devAddr = bo2devAddr;
  my_handle->bo2 = bo2;
  my_handle->execHandle = execHandle;
  my_handle->execData = execData;

  return 0;
}

