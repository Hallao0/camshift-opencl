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

#define TRACK_RECT_W 256
#define TRACK_RECT_H 192

class CamShift
{
	/** OpenCL: Start */		
	cl::Platform platform;

	cl::Context context;
	cl::Program::Sources sources;
	cl::Program program;

	cl::Kernel testKernel;
	cl::Kernel RGBAtoRGKernel;
	cl::CommandQueue queue;
	/** OpenCL: End*/
	
	/** Tract rect */
	cv::Rect trackRect;
	CvScalar trackRectColor;

	/** True if object is being tracked; false otherwise*/
	bool tracking;

public:
	CamShift(void);
	~CamShift(void);

	void drawTrackRect(cv::Mat& mat);	
	void startTracking(cv::Mat& mat);
	void process(cv::Mat& mat);

private:
	cv::Point meanShift(cv::Mat& mat);
	void clean();
};

