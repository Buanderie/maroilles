
// STL
#include <iostream>

// MAROILLES
#include <maroilles.hpp>

using namespace std;
using namespace mrl::opencl;

int main( int argc, char** argv )
{
    OpenCL * cl = new OpenCL();
    std::vector< mrl::opencl::Platform* > platforms = cl->getPlatforms();
    for( int k = 0; k < platforms.size(); ++k )
    {
        std::vector< mrl::opencl::Device* > devices = platforms[k]->getDevices();
        for( int i = 0; i < devices.size(); ++i )
        {
            if( devices[i]->getType() & CL_DEVICE_TYPE_GPU ){
                Program* prg = new Program( devices[i], "/home/said/polbak.cl" );
                Kernel* ker = prg->getKernel( "saxpy" );
                Device* dev = devices[i];
                dev->enqueueKernel( ker );
            }
        }
    }
    delete cl;

    return 0;
}
