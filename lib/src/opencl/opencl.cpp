
// C
#include <cstdlib>

// STL
#include <iostream>
#include <fstream>
#include <iomanip>

// INTERNAL
#include "opencl/opencl.hpp"

using namespace std;

void exitOnFail(cl_int status, const char* message)
{
    if (CL_SUCCESS != status)
    {
        printf("error: %s\n", message);
        exit(-1);
    }
}

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
    cl_platform_id * platformIDs = new cl_platform_id[numPlatforms];
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
    cl_device_id * deviceIDs = new cl_device_id[numDevices];
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

void mrl::opencl::Device::enqueueKernel(mrl::opencl::Kernel *kernel, const size_t localWorkSize, const size_t globalWorkSize )
{
    cl_int ret;

    // wait event synchronization handle used by OpenCL API
    cl_event event;

    ////////////////////////////////////////
    // OpenCL enqueue kernel and wait

    // N work-items in groups of 4
    const size_t groupsize = 16;
    const size_t global[] = { globalWorkSize }, local[] = { localWorkSize };

    //for(;;)
    {
    // enqueue kernel
    ret = clEnqueueNDRangeKernel(   _queue,
                                    kernel->getId(),
                                    sizeof(global)/sizeof(size_t),
                                    NULL,
                                    global,
                                    local,
                                    0,
                                    NULL,
                                    &event);
    //cout << "ret=" << ret << endl;
    exitOnFail(ret, "enqueue kernel");


    // wait for kernel, this forces execution
    ret = clWaitForEvents(1, &event);

    exitOnFail(ret, "wait for enqueue kernel");
    clReleaseEvent(event);
    }
}

void mrl::opencl::Device::enqueueHostToDevice(mrl::opencl::HostBuffer *host, mrl::opencl::DeviceBuffer *device)
{
    // wait event synchronization handle used by OpenCL API
    cl_event event;
    cl_int ret;
    // data transfer for array Y
    /*
    ret = clEnqueueWriteBuffer(_queue,
                                  memY,
                                  CL_FALSE,
                                  0,
                                  N * sizeof(float),
                                  cpuY,
                                  0,
                                  NULL,
                                  &event);
    */
    ret = clEnqueueWriteBuffer(   _queue,
                                  (cl_mem)device->getBufferPtr(),
                                  CL_FALSE,
                                  0,
                                  host->getBufferSize(),
                                  host->getBufferPtr(),
                                  0,
                                  NULL,
                                  &event);

    exitOnFail(ret, "write Y to device");

    ret = clWaitForEvents(1, &event);

    exitOnFail(ret, "wait for write Y to device");

    clReleaseEvent(event);
}

void mrl::opencl::Device::enqueueDeviceToHost(mrl::opencl::DeviceBuffer *device, mrl::opencl::HostBuffer *host)
{
    cl_event event;
    cl_int ret;
    ret = clEnqueueReadBuffer(   _queue,
                                 (cl_mem)device->getBufferPtr(),
                                 CL_FALSE,
                                 0,
                                 host->getBufferSize(),
                                 host->getBufferPtr(),
                                 0,
                                 NULL,
                                 &event);
    exitOnFail(ret, "read Y to device");
    ret = clWaitForEvents(1, &event);
    clReleaseEvent(event);
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


mrl::opencl::Program::Program(mrl::opencl::Device* device, const std::string &programSourcePath)
    :_device(device), _programSourcePath(programSourcePath), _sourceStr(0x0)
{
    init();
}

mrl::opencl::Program::~Program()
{
    destroy();
}

mrl::opencl::Kernel *mrl::opencl::Program::getKernel(const string &kernelName)
{
    cl_int ret;
    /* Create Kernel */
    cl_kernel kernel = NULL;
    kernel = clCreateKernel(_program, kernelName.c_str(), &ret);
    if( !kernel )
    {
        cout << "Couldn't find OpenCL Kernel \"" << kernelName << "\" in program..." << endl;
        return NULL;
    }
    else
    {
        return new mrl::opencl::Kernel( kernel );
    }
}

void mrl::opencl::Program::init()
{
    const cl_int MAX_SOURCE_SIZE (0x100000);
    cl_int ret;
    if( _programSourcePath.size() )
    {
        /* Load kernel code */
        FILE* fp = fopen(_programSourcePath.c_str(), "r");
        if (!fp) {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(1);
        }
        _sourceStr = (char*)malloc(MAX_SOURCE_SIZE);
        _sourceSize = fread(_sourceStr, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        /* Create Program Object */
        _program = clCreateProgramWithSource(_device->getContext(), 1, (const char **)&_sourceStr, (const size_t *)&_sourceSize, &ret);
        exitOnFail(ret, "Create Program");

        /* Compile kernel */
        cl_device_id deviceId = _device->getId();
        ret = clBuildProgram(_program, 1, &(deviceId), NULL, NULL, NULL);
        if (ret != CL_SUCCESS) {
            printCompilerError();
        }
    }
}

void mrl::opencl::Program::destroy()
{
    if( _sourceStr )
    {
        delete _sourceStr;
    }
}

void mrl::opencl::Program::printCompilerError()
{
    cl_build_status status;

    // check build error and build status first
    cl_device_id deviceId = _device->getId();
    clGetProgramBuildInfo(_program, deviceId, CL_PROGRAM_BUILD_STATUS,
                          sizeof(cl_build_status), &status, NULL);

    size_t logSize;
    char *programLog;

    // check build log
    clGetProgramBuildInfo(_program, deviceId,
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    programLog = (char*) calloc (logSize+1, sizeof(char));
    clGetProgramBuildInfo(_program, deviceId,
                          CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
    printf("OpenCL build failed: status=%d, programLog:\n\n%s", status, programLog);
    free(programLog);
    //
}


mrl::opencl::Kernel::Kernel(const cl_kernel &kernelId)
    :_kernelId(kernelId)
{
    init();
}

mrl::opencl::Kernel::~Kernel()
{
    destroy();
}

void mrl::opencl::Kernel::clearArguments()
{
    _argCount = 0;
}

void mrl::opencl::Kernel::init()
{
    _argCount = 0;
}

void mrl::opencl::Kernel::destroy()
{

}
