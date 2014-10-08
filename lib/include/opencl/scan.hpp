#ifndef __SCAN_HPP__
#define __SCAN_HPP__

#include "opencl.hpp"

#define MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 1024
#define MAX_LOCAL_GROUP_SIZE 256

namespace mrl
{
    namespace opencl
    {
        class Scan
        {
        public:
            Scan( mrl::opencl::Device* device, unsigned int numElements, const std::string& kernelSrcFile );

            virtual ~Scan();

        private:
            static const int WORKGROUP_SIZE = 32;
            static const unsigned int   MAX_BATCH_ELEMENTS = 64 * 1048576;
            static const unsigned int MIN_SHORT_ARRAY_SIZE = 4;
            static const unsigned int MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
            static const unsigned int MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
            static const unsigned int MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

            mrl::opencl::Device*    _device;
            mrl::opencl::Program*   _program;

            unsigned int _numElements;
            std::string _kernelSrcFile;
            mrl::opencl::Kernel*    _ckScanExclusiveLocal1;
            mrl::opencl::Kernel*    _ckScanExclusiveLocal2;
            mrl::opencl::Kernel*    _ckUniformUpdate;

            void init();
            void destroy();

            // KERNEL WRAPPERS
            void scanExclusiveLocal1(
                        DeviceBuffer* d_Dst,
                        DeviceBuffer* d_Src,
                        unsigned int n,
                        unsigned int size);

        };
    }
}
#endif // __SCAN_HPP__
