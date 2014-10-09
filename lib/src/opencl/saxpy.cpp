

// INTERNAL
#include "opencl/saxpy.hpp"

using namespace mrl::opencl;

mrl::opencl::Saxpy::Saxpy()
{
    init();
}

mrl::opencl::Saxpy::~Saxpy()
{
    destroy();
}

void mrl::opencl::Saxpy::compute( mrl::opencl::Device* device, mrl::opencl::DeviceBuffer *dX, mrl::opencl::DeviceBuffer *dY, float alpha, const unsigned N )
{
    Program* prg = new Program( device, "/home/said/polbak.cl" );
    Kernel* k = prg->getKernel( "saxpy" );
    k->setArgument< float >( alpha );
    k->setArgument< cl_mem >( (cl_mem)dX->getBufferPtr() );
    k->setArgument< cl_mem >( (cl_mem)dY->getBufferPtr() );
    device->enqueueKernel( k, 16, N );
}

void mrl::opencl::Saxpy::init()
{

}

void mrl::opencl::Saxpy::destroy()
{

}




