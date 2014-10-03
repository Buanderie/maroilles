#ifndef __OPENCL_HPP__
#define __OPENCL_HPP__

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
            MemoryBuffer( const size_t bufferSize ){}
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
            void enqueueHostToDevice( );
            void enqueueDeviceToHost( );


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
            {

            }

            virtual ~HostBuffer()
            {

            }

        private:
        };

        class DeviceBuffer : public MemoryBuffer
        {
        public:

        private:
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
