#ifndef __OPENCL_HPP__
#define __OPENCL_HPP__

// C
#include <cstdlib>

// STL
#include <vector>
#include <string>

// OPENCL
#include <CL/cl.h>

namespace mrl
{
    namespace opencl
    {
        class MemoryBuffer
        {
        public:
            MemoryBuffer(){}
            virtual ~MemoryBuffer(){}

            virtual void * getBufferPtr()=0;
            virtual size_t getBufferSize()=0;
        };

        class HostBuffer;
        class DeviceBuffer;
        class Kernel;
        class Device
        {
        public:
            Device(cl_platform_id platformId, cl_device_id deviceId);
            virtual ~Device();
            cl_device_type getType(){ return _deviceType; }
            cl_context getContext(){ return _context; }
            cl_device_id getId(){ return _deviceId; }
            void enqueueKernel( Kernel* kernel );
            /* TODO */
            void enqueueHostToDevice( HostBuffer* host, DeviceBuffer* device );
            void enqueueDeviceToHost( DeviceBuffer* device, HostBuffer* host );


        private:
            cl_platform_id _platformId;
            cl_device_id _deviceId;
            cl_device_type _deviceType;
            cl_context _context;
            cl_command_queue _queue;

            void init();
            void destroy();
        };

        class HostBuffer : public MemoryBuffer
        {
        public:
            HostBuffer( const size_t bufferSize )
                :_bufferSize( bufferSize ), _hostBuffer(0)
            {
                init();
            }

            virtual ~HostBuffer()
            {
                destroy();
            }

            virtual void * getBufferPtr()
            {
                return _hostBuffer;
            }

            virtual size_t getBufferSize()
            {
                return _bufferSize;
            }

        private:
            void*   _hostBuffer;
            size_t  _bufferSize;

            void init(){ _hostBuffer = malloc( _bufferSize ); }
            void destroy(){ if( _hostBuffer ){ free( _hostBuffer ); } }
        };

        class DeviceBuffer : public MemoryBuffer
        {
        public:
            DeviceBuffer( Device* device, const size_t bufferSize )
                :_bufferSize( bufferSize ), _deviceBuffer(0), _device( device )
            {
                init();
            }

            virtual ~DeviceBuffer()
            {
                destroy();
            }

            virtual void * getBufferPtr()
            {
                return (void*)(_deviceBuffer);
            }

            virtual size_t getBufferSize()
            {
                return _bufferSize;
            }

        private:
            cl_mem  _deviceBuffer;
            size_t  _bufferSize;
            Device* _device;

            void init(){
                cl_int ret;
                _deviceBuffer = clCreateBuffer( _device->getContext(), CL_MEM_READ_WRITE, _bufferSize, 0, &ret);
            }

            void destroy(){
                clReleaseMemObject( _deviceBuffer );
            }
        };

        class Platform
        {
        public:
            Platform(cl_platform_id platformId);
            virtual ~Platform();
            std::vector< mrl::opencl::Device* >& getDevices(){ return _devices; }

        private:
            cl_platform_id _platformId;
            std::vector< Device* > _devices;

            void init();
            void destroy();

        };

        class Program
        {
        public:
            Program(Device *device, const std::string& programSourcePath );
            virtual ~Program();

            Kernel* getKernel( const std::string& kernelName );

        private:
            Device* _device;
            std::string _programSourcePath;
            char* _sourceStr;
            size_t _sourceSize;

            cl_program _program;

            void init();
            void destroy();
            void printCompilerError();
        };

        class Kernel
        {
        public:
            Kernel( const cl_kernel& kernelId );
            virtual ~Kernel();
            cl_kernel getId(){ return _kernelId; }
            template< class T > void setArgument( const T& argValue ){ clSetKernelArg( _kernelId, _argCount++, sizeof(T), &argValue); }

        private:
            cl_kernel _kernelId;
            unsigned int _argCount;

            void init();
            void destroy();
        };

        class OpenCL
        {
        public:
            OpenCL();
            virtual ~OpenCL();
            std::vector< mrl::opencl::Platform* >& getPlatforms(){ return _platforms; }

        private:
            std::vector< opencl::Platform* > _platforms;

            void init();
            void destroy();

        };
    }
}
#endif // OPENCL_HPP
