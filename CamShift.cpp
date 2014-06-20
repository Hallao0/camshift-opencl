#include "CamShift.hpp"

CamShift::CamShift(void)
{
	try	{
		
		std::string programStr = common::get_file_content(std::string(KERNELS_SOURCE_FILE));	
		std::cout << programStr;
		this->sources.push_back(std::pair<char*, size_t>(const_cast<char*>(programStr.c_str()), programStr.length()));

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);	

		if(platforms.empty())
		{
			throw std::runtime_error("No OpenCL platform");
		}

		this->platform = platforms[0];

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		if(devices.empty())
			throw std::runtime_error("No devices");

		this->context = cl::Context(devices);

		cl::Program program(context, sources);
		program.build(devices);

		this->testKernel = cl::Kernel(program, "test");
		this->queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

	} catch(...)
	{
		std::rethrow_exception(std::current_exception());
	}
}

CamShift::~CamShift(void)
{

}

void CamShift::process(uchar * image, int w, int h)
{
	try {
		int width = w;
		int height = h;

		int size = 3*w*h*sizeof(uchar);

		cl::Buffer in(context, CL_MEM_READ_ONLY, size);
		cl::Buffer out(context, CL_MEM_WRITE_ONLY, size);

		queue.enqueueWriteBuffer(in, CL_TRUE, 0, size, image);

		cl::NDRange global(width*3, height);
		cl::NDRange local = cl::NullRange;
		cl::NDRange offset = cl::NullRange;

		this->testKernel.setArg(0, sizeof(cl_uchar *), &in);
		this->testKernel.setArg(1, sizeof(cl_uchar *), &out);
		this->testKernel.setArg(2, w*3);

		queue.enqueueNDRangeKernel(this->testKernel, offset, global, local);
		queue.finish();

		queue.enqueueReadBuffer(out, CL_TRUE, 0, size, image);

	} catch(...)
	{
		std::rethrow_exception(std::current_exception());
	}
}