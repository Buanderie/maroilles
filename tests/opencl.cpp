
#include <cmath>

#include <iostream>

#include <gtest/gtest.h>

#include <maroilles.hpp>

using namespace std;
using namespace mrl::opencl;

TEST(OpenCL, FindCPUDevice) {
    OpenCL * cl = new OpenCL();
    Device * device = NULL;
    std::vector< mrl::opencl::Platform* > platforms = cl->getPlatforms();
    for( int k = 0; k < platforms.size(); ++k )
    {
        std::vector< mrl::opencl::Device* > devices = platforms[k]->getDevices();
        for( int i = 0; i < devices.size(); ++i )
        {
            if( devices[i]->getType() & CL_DEVICE_TYPE_CPU ){
                device = devices[i];
                break;
            }
        }
    }
    ASSERT_TRUE( device != NULL );
    delete cl;
}

TEST(OpenCL, FindGPUDevice) {
    OpenCL * cl = new OpenCL();
    Device * device = NULL;
    std::vector< mrl::opencl::Platform* > platforms = cl->getPlatforms();
    for( int k = 0; k < platforms.size(); ++k )
    {
        std::vector< mrl::opencl::Device* > devices = platforms[k]->getDevices();
        for( int i = 0; i < devices.size(); ++i )
        {
            if( devices[i]->getType() & CL_DEVICE_TYPE_GPU ){
                device = devices[i];
                break;
            }
        }
    }
    ASSERT_TRUE( device != NULL );
    delete cl;
}

TEST(OpenCL, SaxpyCPU)
{
    // Find device
    OpenCL * cl = new OpenCL();
    Device * device = NULL;
    std::vector< mrl::opencl::Platform* > platforms = cl->getPlatforms();
    for( int k = 0; k < platforms.size(); ++k )
    {
        std::vector< mrl::opencl::Device* > devices = platforms[k]->getDevices();
        for( int i = 0; i < devices.size(); ++i )
        {
            if( devices[i]->getType() & CL_DEVICE_TYPE_CPU ){
                device = devices[i];
                break;
            }
        }
    }
    //

    const int N = 5000000;
    const float alpha = 1.5f;
    const float eps = 0.00001;

    HostBuffer* hX = new HostBuffer( sizeof( float ) * N );
    HostBuffer* hYRef = new HostBuffer( sizeof( float ) * N );
    HostBuffer* hY = new HostBuffer( sizeof( float ) * N );

    DeviceBuffer* dX = new DeviceBuffer( device, sizeof( float ) * N );
    DeviceBuffer* dY = new DeviceBuffer( device, sizeof( float ) * N );

    // Fill host buffers with data
    float* hxPtr = (float*)(hX->getBufferPtr());
    float* hyPtr = (float*)(hY->getBufferPtr());
    for( int k = 0; k < N; ++k )
    {
        hxPtr[ k ] = (float)(k);
        hyPtr[ k ] = (float)rand() / (float)rand();
    }

    // Push them to the device
    device->enqueueHostToDevice( hX, dX );
    device->enqueueHostToDevice( hY, dY );

    // Compute
    Saxpy* saxpy = new Saxpy();
    saxpy->compute( device, dX, dY, alpha, N );

    // Compute reference solution
    for( int k = 0; k < N; ++k )
    {
        float* hyRefPtr = (float*)(hYRef->getBufferPtr());
        hyRefPtr[k] = hyPtr[k] + hxPtr[k] * alpha;
    }

    // Push the result back to the host
    device->enqueueDeviceToHost( dY, hY );

    // Compare shit with reference implementation
    for( int k = 0; k < N; ++k )
    {
        float* hyRefPtr = (float*)(hYRef->getBufferPtr());
        ASSERT_LT( fabs(hyRefPtr[k] - hyPtr[k]), eps );
    }
}

TEST(OpenCL, SaxpyGPU)
{
    // Find device
    OpenCL * cl = new OpenCL();
    Device * device = NULL;
    std::vector< mrl::opencl::Platform* > platforms = cl->getPlatforms();
    for( int k = 0; k < platforms.size(); ++k )
    {
        std::vector< mrl::opencl::Device* > devices = platforms[k]->getDevices();
        for( int i = 0; i < devices.size(); ++i )
        {
            if( devices[i]->getType() & CL_DEVICE_TYPE_GPU ){
                device = devices[i];
                break;
            }
        }
    }
    //

    const int N = 100000;
    const float alpha = 1.5f;
    const float eps = 0.00001;

    HostBuffer* hX = new HostBuffer( sizeof( float ) * N );
    HostBuffer* hYRef = new HostBuffer( sizeof( float ) * N );
    HostBuffer* hY = new HostBuffer( sizeof( float ) * N );

    DeviceBuffer* dX = new DeviceBuffer( device, sizeof( float ) * N );
    DeviceBuffer* dY = new DeviceBuffer( device, sizeof( float ) * N );

    // Fill host buffers with data
    float* hxPtr = (float*)(hX->getBufferPtr());
    float* hyPtr = (float*)(hY->getBufferPtr());
    for( int k = 0; k < N; ++k )
    {
        hxPtr[ k ] = (float)(k);
        hyPtr[ k ] = (float)rand() / (float)rand();
    }

    // Push them to the device
    device->enqueueHostToDevice( hX, dX );
    device->enqueueHostToDevice( hY, dY );

    // Compute
    Saxpy* saxpy = new Saxpy();
    saxpy->compute( device, dX, dY, alpha, N );

    // Compute reference solution
    for( int k = 0; k < N; ++k )
    {
        float* hyRefPtr = (float*)(hYRef->getBufferPtr());
        hyRefPtr[k] = hyPtr[k] + hxPtr[k] * alpha;
    }

    // Push the result back to the host
    device->enqueueDeviceToHost( dY, hY );

    // Compare shit with reference implementation
    for( int k = 0; k < N; ++k )
    {
        float* hyRefPtr = (float*)(hYRef->getBufferPtr());
        ASSERT_LT( fabs(hyRefPtr[k] - hyPtr[k]), eps );
    }
}
