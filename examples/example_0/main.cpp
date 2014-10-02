
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
            if( devices[i]->getType() & CL_DEVICE_TYPE_GPU )
                cout << "youhou GPU" << endl;
        }
    }

    return 0;
}
