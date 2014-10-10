#ifndef __RADIXSORT_HPP__
#define __RADIXSORT_HPP__

#include "opencl.hpp"

namespace mrl
{
    namespace opencl
    {
        class RadixSort
        {
        public:
            RadixSort(mrl::opencl::Device *device, unsigned int maxElements);
            virtual ~RadixSort();

        private:
            void init();
            void destroy();

            static const int CTA_SIZE = 128; // Number of threads per block
            static const unsigned int WARP_SIZE = 32;
            static const unsigned int bitStep = 4;

            mrl::opencl::Device* _device;
            mrl::opencl::DeviceBuffer* _dTempKeys;
            mrl::opencl::DeviceBuffer* _dCounters;
            mrl::opencl::DeviceBuffer* _dCounterSum;
            mrl::opencl::DeviceBuffer* _dBlockOffsets;

            mrl::opencl::Program*      _oclProgram;
            mrl::opencl::Kernel*       _ckRadixSortBlocksKeysOnly;

            // Kernel wrappers
            void radixSortBlocksKeysOnlyOCL( mrl::opencl::DeviceBuffer* d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements);
            //

            unsigned int  mNumElements;     // Number of elements of temp storage allocated
            unsigned int *mTempValues;      // Intermediate storage for values
            unsigned int  mNumBlocks;
            unsigned int  mNumBlocks2;
            unsigned int  mMaxElements;
        };
    }
}
#endif // RADIXSORT_HPP
