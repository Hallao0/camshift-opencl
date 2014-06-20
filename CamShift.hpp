#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define KERNELS_SOURCE_FILE "camshift_kernels.cl"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <CL\cl.hpp>
#include <string>
#include <stdexcept>
#include <exception>
#include <iostream>
#include <fstream>

#include "common.hpp"

class CamShift
{
	/** OpenCL: Start */		
	cl::Platform platform;

	cl::Context context;
	cl::Program::Sources sources;

	cl::Kernel testKernel;
	cl::CommandQueue queue;
	/** OpenCL: End*/

public:
	CamShift(void);
	~CamShift(void);
	void process(uchar * image, int w, int h);

private:
	void clean();
};

