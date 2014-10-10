
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

                //Scan* scan = new Scan( dev, 15000, "/home/said/Scan_b.cl" );
                //cout << "lolol " << endl;
                RadixSort* rsort = new RadixSort( dev, 50000 );
                cout << "lolol " << endl;

                const int N = 10000000;
                HostBuffer * h0 = new HostBuffer( N * sizeof(float) );
                float* hostPtr = (float*)h0->getBufferPtr();
                for( int j = 0; j < N; ++j )
                {
                    hostPtr[j] = (float)j;
                }
                DeviceBuffer * d0 = new DeviceBuffer( dev, h0->getBufferSize() );
                DeviceBuffer * d1 = new DeviceBuffer( dev, h0->getBufferSize() );
                std::string pol;

                for(;;)
                {
                    dev->enqueueHostToDevice( h0, d0 );
                    ker->setArgument< float >( 1.0f );
                    ker->setArgument< cl_mem >( (cl_mem)d0->getBufferPtr() );
                    ker->setArgument< cl_mem >( (cl_mem)d1->getBufferPtr() );
                    dev->enqueueKernel( ker, 128, N );
                    dev->enqueueDeviceToHost( d1, h0 );
                    ker->clearArguments();
                }
                /*
                for( int j = 0; j < N; ++j )
                    cout << "h0[" << j << "]=" << hostPtr[j] << endl;
                */
            }
        }
    }
    delete cl;

    return 0;
}
