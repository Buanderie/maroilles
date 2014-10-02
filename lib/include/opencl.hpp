#ifndef __OPENCL_HPP__
#define __OPENCL_HPP__

// STL
#include <vector>

// OPENCL
#include <CL/cl.h>

namespace mrl
{
    namespace opencl
    {
        class Device
        {
        public:
            Device(cl_platform_id platformId, cl_device_id deviceId);
            virtual ~Device();
            cl_device_type getType(){ return _deviceType; }

        private:
            cl_platform_id _platformId;
            cl_device_id _deviceId;
            cl_device_type _deviceType;
            cl_context _context;
            cl_command_queue _queue;

            void init();
            void destroy();
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
