
// STL
#include <iostream>

// INTERNAL
#include "opencl.hpp"

using namespace std;

mrl::opencl::OpenCL::OpenCL()
{
    init();
}

mrl::opencl::OpenCL::~OpenCL()
{
    destroy();
}

void mrl::opencl::OpenCL::init()
{
    // Enumerate platforms
    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);

    // get platform IDs
    cl_platform_id platformIDs[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platformIDs, NULL);

    for (size_t i = 0; i < numPlatforms; i++)
    {
        _platforms.push_back( new mrl::opencl::Platform(platformIDs[i]) );
    }
}

void mrl::opencl::OpenCL::destroy()
{
    for( size_t k = 0; k < _platforms.size(); ++k )
        delete _platforms[k];
    _platforms.clear();
}


mrl::opencl::Platform::Platform(cl_platform_id platformId)
    :_platformId(platformId)
{
    init();
}

mrl::opencl::Platform::~Platform()
{
    destroy();
}

void mrl::opencl::Platform::init()
{
    cl_uint numDevices;
    cl_int status = clGetDeviceIDs(_platformId,
                            CL_DEVICE_TYPE_ALL,
                            0,
                            NULL,
                            &numDevices);

    // get device IDs for a platform
    cl_device_id deviceIDs[numDevices];
    status = clGetDeviceIDs(_platformId,
                            CL_DEVICE_TYPE_ALL,
                            numDevices,
                            deviceIDs,
                            NULL);

    for( size_t k = 0; k < numDevices; ++k )
    {
        _devices.push_back( new mrl::opencl::Device(_platformId, deviceIDs[k]) );
    }
}

void mrl::opencl::Platform::destroy()
{
    for( size_t k = 0; k < _devices.size(); ++k )
        delete _devices[k];
    _devices.clear();
}


mrl::opencl::Device::Device(cl_platform_id platformId, cl_device_id deviceId)
    :_deviceId(deviceId), _platformId(platformId)
{
    init();
}

mrl::opencl::Device::~Device()
{
    destroy();
}

void mrl::opencl::Device::init()
{
    //cl_device_type deviceType;
    cl_int status = clGetDeviceInfo(_deviceId,
                             CL_DEVICE_TYPE,
                             sizeof(cl_device_type),
                             &_deviceType,
                             NULL);

    ////////////////////////////////////////
    // OpenCL context
    cl_context_properties props[] = { CL_CONTEXT_PLATFORM,
                                      (cl_context_properties) _platformId,
                                      0 };

    _context = clCreateContext(          props,
                                         1,
                                         &_deviceId,
                                         NULL,
                                         NULL,
                                         &status);

    ////////////////////////////////////////
    // OpenCL command queue
    _queue = clCreateCommandQueue(                _context,
                                                  _deviceId,
                                                  0,
                                                  &status);

    //cout << "init" << endl;
}

void mrl::opencl::Device::destroy()
{
    clReleaseCommandQueue(_queue);
    clReleaseContext(_context);
}
