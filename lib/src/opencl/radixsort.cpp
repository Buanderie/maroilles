
// STL
#include <iostream>

// INTERNAL
#include "opencl/radixsort.hpp"

using namespace std;
using namespace mrl::opencl;

mrl::opencl::RadixSort::RadixSort( mrl::opencl::Device* device, unsigned int maxElements )
    :_device(device), mMaxElements(maxElements)
{
    init();
}

mrl::opencl::RadixSort::~RadixSort()
{
    destroy();
}

void mrl::opencl::RadixSort::init()
{
    mNumBlocks = ((mMaxElements % (CTA_SIZE * 4)) == 0) ?
                (mMaxElements / (CTA_SIZE * 4)) : (mMaxElements / (CTA_SIZE * 4) + 1);
    mNumBlocks2 = ((mMaxElements % (CTA_SIZE * 2)) == 0) ?
                (mMaxElements / (CTA_SIZE * 2)) : (mMaxElements / (CTA_SIZE * 2) + 1);

    cout << "mNumBlocks=" << mNumBlocks << endl;
    cout << "mNumBlocks2=" << mNumBlocks2 << endl;

    _dTempKeys = new DeviceBuffer( _device, sizeof(unsigned int) * mMaxElements );
    _dCounters = new DeviceBuffer( _device, WARP_SIZE * mNumBlocks * sizeof(unsigned int) );
    _dCounterSum = new DeviceBuffer( _device, WARP_SIZE * mNumBlocks * sizeof(unsigned int) );
    _dBlockOffsets = new DeviceBuffer( _device, WARP_SIZE * mNumBlocks * sizeof(unsigned int) );

    _oclProgram = new Program( _device, "/home/said/temp/oclRadixSort/RadixSort.cl" );
    _ckRadixSortBlocksKeysOnly = _oclProgram->getKernel( "radixSortBlocksKeysOnly" );

}
/*
void mrl::opencl::RadixSort::radixSortBlocksKeysOnlyOCL(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
    unsigned int totalBlocks = numElements/4/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};
    cl_int ciErrNum;
    ciErrNum  = clSetKernelArg(ckRadixSortBlocksKeysOnly, 0, sizeof(cl_mem), (void*)&d_keys);
  ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 1, sizeof(cl_mem), (void*)&d_tempKeys);
    ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 2, sizeof(unsigned int), (void*)&nbits);
    ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 3, sizeof(unsigned int), (void*)&startbit);
  ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 4, sizeof(unsigned int), (void*)&numElements);
  ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 5, sizeof(unsigned int), (void*)&totalBlocks);
    ciErrNum |= clSetKernelArg(ckRadixSortBlocksKeysOnly, 6, 4*CTA_SIZE*sizeof(unsigned int), NULL);
  ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckRadixSortBlocksKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}
*/

void mrl::opencl::RadixSort::destroy()
{

}

void RadixSort::radixSortBlocksKeysOnlyOCL(DeviceBuffer *d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
    unsigned int totalBlocks = numElements/4/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};
    cl_int ciErrNum;
    _ckRadixSortBlocksKeysOnly->setArgument< cl_mem >( (cl_mem)d_keys->getBufferPtr() );
    _ckRadixSortBlocksKeysOnly->setArgument< cl_mem >( (cl_mem)_dTempKeys->getBufferPtr() );
    _ckRadixSortBlocksKeysOnly->setArgument< unsigned int >( nbits );
    _ckRadixSortBlocksKeysOnly->setArgument< unsigned int >( startbit );
    _ckRadixSortBlocksKeysOnly->setArgument< unsigned int >( numElements );
    _ckRadixSortBlocksKeysOnly->setArgument< unsigned int >( totalBlocks );
    _ckRadixSortBlocksKeysOnly->setLocalArgument( 4*CTA_SIZE*sizeof(unsigned int) );
    _device->enqueueKernel( _ckRadixSortBlocksKeysOnly, localWorkSize[0], globalWorkSize[0] );
}


