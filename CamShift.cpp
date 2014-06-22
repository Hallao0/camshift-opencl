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

		this->program = cl::Program(context, sources);
		program.build(devices);

		this->testKernel = cl::Kernel(program, "test");
		this->RGBAtoRGKernel = cl::Kernel(program, "RGBAtoRxG16_4");
		this->queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		this->trackRect = cv::Rect(0, 0, TRACK_RECT_W, TRACK_RECT_H);
		this->trackRectColor = cvScalar(200, 0, 0);

		this->tracking = false;

	} catch(cl::Error &e)
	{		         	
		if(e.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
			std::cout << " \n\t\t\tBUILD LOG\n";
			std::cout << " ************************************************\n";
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
			std::cout << " ************************************************\n";
		}
		std::rethrow_exception(std::current_exception());
	} catch(...)
	{		         	
		std::rethrow_exception(std::current_exception());
	}
}

CamShift::~CamShift(void)
{

}

void CamShift::drawTrackRect(cv::Mat& mat)
{
	int width = mat.cols;
	int height = mat.rows;

	// Je¿eli nie œledzimy obiektu, to rysujemy prostok¹t na œrodku.
	if(!tracking)
	{
		this->trackRect.x = width/2 - this->trackRect.width/2;
		this->trackRect.y = height/2 - this->trackRect.height/2;
	}	
	cv::rectangle(mat, trackRect, trackRectColor, 2);
}

void CamShift::startTracking(cv::Mat& mat)
{
	this->tracking = true;
	getTrackedObjHist(mat);
}

void CamShift::getTrackedObjHist(cv::Mat& mat)
{
	int w = mat.cols;
	int h = mat.rows;

	// ZMIANA BGR -> RGBA
	uchar* dataRGBA = new uchar[mat.total()*4];
	cv::Mat matRGBA(mat.size(), CV_8UC4, dataRGBA);
	cv::cvtColor(mat, matRGBA, CV_BGR2RGBA, 4);

	// ZMIANA RGBA -> RG
	int size_rgba_bytes = w*h*sizeof(cl_uchar4);
	int size_rg_bytes = w*h*sizeof(cl_uchar);

	cl::Buffer in(context, CL_MEM_READ_ONLY, size_rgba_bytes);
	cl::Buffer out(context, CL_MEM_WRITE_ONLY, size_rg_bytes);

	queue.enqueueWriteBuffer(in, CL_TRUE, 0, size_rgba_bytes, matRGBA.data);

	// Szerokoœæ zmniejszamy o 4 bo kernel analizuje po 4 jednoczeœnie
	// Liczba pikseli musi byæ podzielna przez 4, no ale ka¿da szanuj¹ca siê
	// rozdzielczoœæ spe³nia ten warunek.
	const int w4 = w / 4;
	cl::NDRange global(w4, h);
	cl::NDRange local = cl::NullRange;
	cl::NDRange offset = cl::NDRange(0, 0);

	this->RGBAtoRGKernel.setArg(0, sizeof(cl_uint4 *), &in);
	this->RGBAtoRGKernel.setArg(1, sizeof(cl_uchar4 *), &out);

	queue.enqueueNDRangeKernel(this->RGBAtoRGKernel, offset, global, local);
	queue.finish();
	
#ifndef __CS_DEBUG_OFF__
	uchar* dataRG = new uchar[size_rg_bytes];
	queue.enqueueReadBuffer(out, CL_TRUE, 0, size_rg_bytes, dataRG);
	for(int i = 0; i < 50; i++)
	{
		std::cout << i << "\t Hist idx (RxG): " << int(dataRG[i]) << "\n";
		std::cout << i << "RGBA: \t R:" << int(dataRGBA[i*4]) << "\t G:" << int(dataRGBA[(i*4)+1])
			<< "\t B:" << int(dataRGBA[(i*4)+2]) << "\t A:" << int(dataRGBA[(i*4)+3]) << "\n";
		std::cout << i << "BGR: \t R:" << int(mat.data[(i*3)+2]) << "\t G:" << int(mat.data[(i*3)+1])
			<< "\t B:" << int(mat.data[(i*3)]) << "\n";
	}
#endif
}

void CamShift::process(cv::Mat& mat)
{
	try {

		int w = mat.cols;
		int h = mat.rows;
		uchar * image = mat.data;

		int size = 3*w*h*sizeof(uchar);

		cl::Buffer in(context, CL_MEM_READ_ONLY, size);
		cl::Buffer out(context, CL_MEM_WRITE_ONLY, size);

		queue.enqueueWriteBuffer(in, CL_TRUE, 0, size, image);

		cl::NDRange global(3*w, h-240);
		cl::NDRange local = cl::NullRange;
		cl::NDRange offset = cl::NDRange(0, 240);

		const size_t offsetFlat = 240 * 3 * w;
		static int change = 50;

		this->testKernel.setArg(0, sizeof(cl_uchar *), &in);
		this->testKernel.setArg(1, sizeof(cl_uchar *), &out);
		this->testKernel.setArg(2, 3*w);
		this->testKernel.setArg(3, change);

		queue.enqueueNDRangeKernel(this->testKernel, offset, global, local);
		queue.finish();

		queue.enqueueReadBuffer(out, CL_TRUE, offsetFlat, size-offsetFlat, image+offsetFlat);

		drawTrackRect(mat);

	} catch(...)
	{
		std::rethrow_exception(std::current_exception());
	}
}