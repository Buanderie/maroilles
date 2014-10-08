
// STL
#include <iostream>

// INTERNAL
#include "opencl/scan.hpp"

using namespace std;

mrl::opencl::Scan::Scan(mrl::opencl::Device* device , unsigned int numElements, const std::string &kernelSrcFile)
    :_device(device), _numElements(numElements), _kernelSrcFile(kernelSrcFile)
{
    init();
}

mrl::opencl::Scan::~Scan()
{
    destroy();
}

void mrl::opencl::Scan::init()
{
    _program = new mrl::opencl::Program( _device, _kernelSrcFile );
    _ckScanExclusiveLocal1 = _program->getKernel( "scanExclusiveLocal1" );
    _ckScanExclusiveLocal2 = _program->getKernel( "scanExclusiveLocal2" );
    _ckUniformUpdate = _program->getKernel( "uniformUpdate" );

    mrl::opencl::DeviceBuffer* src = new mrl::opencl::DeviceBuffer(_device, 20000);
    mrl::opencl::DeviceBuffer* dst = new mrl::opencl::DeviceBuffer(_device, 20000);

    scanExclusiveLocal1(
        dst,
        src,
        1,
        4 * WORKGROUP_SIZE
    );

    cout << "done" << endl;
}

void mrl::opencl::Scan::destroy()
{
    if( _program )
    {
        delete _program;
    }
}

void mrl::opencl::Scan::scanExclusiveLocal1(mrl::opencl::DeviceBuffer *d_Dst, mrl::opencl::DeviceBuffer *d_Src, unsigned int n, unsigned int size)
{
    size_t localWorkSize, globalWorkSize;

    _ckScanExclusiveLocal1->setArgument< cl_mem >( (cl_mem)d_Dst->getBufferPtr() );
    _ckScanExclusiveLocal1->setArgument< cl_mem >( (cl_mem)d_Src->getBufferPtr() );
    _ckScanExclusiveLocal1->setLocalArgument( 2 * WORKGROUP_SIZE * sizeof(unsigned int) );
    _ckScanExclusiveLocal1->setArgument< unsigned int >( size );

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = (n * size) / 4;
    cout << "localWorkSize = " << localWorkSize << endl;
    cout << "globalWorkSize = " << globalWorkSize << endl;

    _device->enqueueKernel( _ckScanExclusiveLocal1, localWorkSize, globalWorkSize );

}
