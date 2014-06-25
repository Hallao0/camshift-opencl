#include "CamShift.hpp"

CamShift::CamShift(void)
{
	try	{

		std::string programStr = common::get_file_content(std::string(KERNELS_SOURCE_FILE));
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
		this->kernelRGBA2RG_HIST_IDX_4= cl::Kernel(program, "RGBA2RG_HIST_IDX_4");
		this->kernelHistRG = cl::Kernel(program, "histRG");
		this->queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		this->trackRect = cv::Rect(0, 0, TRACK_RECT_W, TRACK_RECT_H);
		this->trackRectColor = cvScalar(200, 0, 0);

		this->tracking = false;

	} catch(cl::Error &e)
	{		         	
#ifndef __CS_DEBUG_OFF__
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
#endif
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
	cl_uint* dataRGBA = new cl_uint[mat.total()];
	cv::Mat matRGBA(mat.size(), CV_8UC4, dataRGBA);
	cv::cvtColor(mat, matRGBA, CV_BGR2RGBA, 4);

	// ZMIANA RGBA -> RG
	int size_rgba_bytes = w*h*sizeof(cl_uchar4);
	int size_rg_bytes = this->trackRect.width * this->trackRect.height * sizeof(cl_uchar);

	cl::Buffer in(context, CL_MEM_READ_ONLY, size_rgba_bytes);
	cl::Buffer out(context, CL_MEM_READ_WRITE, size_rg_bytes);

	queue.enqueueWriteBuffer(in, CL_TRUE, 0, size_rgba_bytes, matRGBA.data);

	// Szerokoœæ zmniejszamy o 4 bo kernel analizuje po 4 jednoczeœnie
	// Liczba pikseli musi byæ podzielna przez 4, no ale ka¿da szanuj¹ca siê
	// rozdzielczoœæ spe³nia ten warunek.
	const int w4 = this->trackRect.width / 4;
	cl::NDRange global(w4, this->trackRect.height);
	cl::NDRange local = cl::NullRange;
	cl::NDRange offset = cl::NDRange(this->trackRect.x/4, this->trackRect.y);

	this->kernelRGBA2RG_HIST_IDX_4.setArg(0, sizeof(cl_uint4 *), &in);
	this->kernelRGBA2RG_HIST_IDX_4.setArg(1, sizeof(cl_uchar4 *), &out);
	this->kernelRGBA2RG_HIST_IDX_4.setArg(2, w/4);

	queue.enqueueNDRangeKernel(this->kernelRGBA2RG_HIST_IDX_4, offset, global, local);
	queue.finish();

#ifndef __CS_DEBUG_OFF__
	
	std::cout << "global size: " << this->trackRect.width * this->trackRect.height/4 << "\n";
	std::cout << "rect X: " << this->trackRect.x << "\n";
	std::cout << "rect Y: " << this->trackRect.y << "\n";

	uchar* dataRG = new uchar[size_rg_bytes];
	queue.enqueueReadBuffer(out, CL_TRUE, 0, size_rg_bytes, dataRG);
	// Wypisuje 10 pierwszych wyników przekszta³cenia BGR -> RGBA -> R+16*G(indeks na histogramie 16x16 RxG)
	// dla rêcznego sprawdzenia poprawnoœci obliczeñ
	for(int i = 0; i < 0; i++)
	{
		
		std::cout << i << "\t Hist idx (RxG): " << int(dataRG[i]) << "\n";
		std::cout << i+144*640+192 << "RGBA: \t R:" << int(dataRGBA[(i*4)+144*640+192]) << "\t G:" << int(dataRGBA[(i*4)+1+144*640+192])
			<< "\t B:" << int(dataRGBA[(i*4)+2+144*640+192]) << "\t A:" << int(dataRGBA[(i*4)+3+144*640+192]) << "\n";
		std::cout << i << "BGR: \t R:" << int(mat.data[(i*3)+2]) << "\t G:" << int(mat.data[(i*3)+1])
			<< "\t B:" << int(mat.data[(i*3)]) << "\n";		

	}
	int histCPU[HISTOGRAM_LEVELS];
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		histCPU[i]=0;
	}
	for(int i = 0; i < this->trackRect.height; i++)
	{
		for(int j = 0; j < this->trackRect.width; j++)
		{
			histCPU[int(dataRG[j + i * this->trackRect.width])]++;
		}
	}

#endif

	const int workGroupSize = HISTOGRAM_LEVELS;
	const int n4VectorsPerWorkItem = 1;
	// Rozmiar pix to jeden bajt
	// Wczytujemy jednoczeœnie 4 int'y, zatem 16 bajtów
	const int n4Vectors = this->trackRect.width * this->trackRect.height / 16;
	// rozmiar: liczba wektorów uint4 przez liczbe wektorów na workItem
	const int globalSize = n4Vectors/n4VectorsPerWorkItem;
	const int nWorkGroups = (globalSize / workGroupSize);

	// Globalnie: jeden wymiar, 
	global =  cl::NDRange(globalSize);
	local = cl::NDRange(workGroupSize);
	offset = cl::NDRange(0);
	const int size_global_rg_hist_bytes = nWorkGroups * HISTOGRAM_LEVELS * sizeof(cl_uint);
	cl::Buffer globalHist(context, CL_MEM_READ_WRITE, size_global_rg_hist_bytes);

	this->kernelHistRG.setArg(0, sizeof(cl_uint4 *), &out);
	this->kernelHistRG.setArg(1, sizeof(cl_uint *), &globalHist);
	this->kernelHistRG.setArg(2, n4VectorsPerWorkItem);

	queue.enqueueNDRangeKernel(this->kernelHistRG, offset, global, local);
	queue.finish();

#ifndef __CS_DEBUG_OFF__

	// Pobieram histogramy wygenerowane
	// przez ka¿d¹ workGroup
	cl_uint* dataHIST = new cl_uint[nWorkGroups * HISTOGRAM_LEVELS];
	queue.enqueueReadBuffer(globalHist, CL_TRUE, 0, size_global_rg_hist_bytes, dataHIST);

	// Suma przyznanych w histogramach punktów
	int sumGPU = 0;
	for(int i = 0; i < nWorkGroups * HISTOGRAM_LEVELS; i++)
	{
		sumGPU += int(dataHIST[i]);
	}
	// Redukcja do jednego histogramu
	// TODO: Przenieœæ to do GPU
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		int bin = 0;
		for(int j = 0; j< nWorkGroups;j++)
		{
			bin += dataHIST[i + j*HISTOGRAM_LEVELS];
		}
		dataHIST[i] = bin;
	}
	// Sprawdzenie, czy histogram obliczony przez GPU jest identyczny
	// z obliczonym tradycyjnie.
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		if(histCPU[i] != dataHIST[i])
		{
			std::cout << i << "\tCPU: " << histCPU[i] << "\tGPU: " << dataHIST[i] << "\n";
		}
	}	
	for(int i = 0; i < 256; i++)
	{
		std::cout << i << ":\t"<< dataHIST[i] << "\n";		
	}
	// Czy liczba przydzielonych "punktów" jest OK
	if(w*h != sumGPU)
	{
		std::cout <<  "\t SUM CPU: " << this->trackRect.width * this->trackRect.height << "\t SUM GPU: " << sumGPU << "\n";
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