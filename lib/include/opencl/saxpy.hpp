#ifndef __SAXPY_HPP__
#define __SAXPY_HPP__

#include "opencl/opencl.hpp"

namespace mrl
{
    namespace opencl
    {
        class Saxpy
        {
        public:
            Saxpy();
            virtual ~Saxpy();
            void compute(Device *device, mrl::opencl::DeviceBuffer* dX, mrl::opencl::DeviceBuffer* dY, float alpha,  const unsigned N  );

        private:

            void init();
            void destroy();

        };
    }
}

#endif // __SAXPY_HPP__
