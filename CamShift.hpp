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
#include <array>

#include "common.hpp"

#define TRACK_RECT_W 128
#define TRACK_RECT_H 128

#define HISTOGRAM_LEVELS 256

#define __CS_DEBUG_OFF__

class CamShift
{
	/** OpenCL: Start */		
	cl::Platform platform;

	cl::Context context;
	cl::Program::Sources sources;
	cl::Program program;

	cl::Kernel testKernel;

	cl::Kernel kernelRGBA2RG_HIST_IDX_4;
	cl::Kernel kernelHistRG;
	cl::Kernel kernelRGBA2HistScore;

	cl::CommandQueue queue;
	/** OpenCL: End*/
	
	/** Tract rect */
	cv::Rect trackRect;
	CvScalar trackRectColor;

	/** True if object is being tracked; false otherwise*/
	bool tracking;
	cl_uint trackedObjHist[256];
	
public:
	CamShift(void);
	~CamShift(void);

	void drawTrackRect(cv::Mat& mat);	
	void startTracking(cv::Mat& mat);
	void process(cv::Mat& mat);

private:
	void getTrackedObjHist(cv::Mat& mat);
	cv::Point meanShift(cv::Mat& mat);
	void clean();
};

