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

		this->kernelRGBA2RG_HIST_IDX_4= cl::Kernel(program, "RGBA2RG_HIST_IDX_4");
		this->kernelHistRG = cl::Kernel(program, "histRG");
		this->kernelRGBA2HistScore = cl::Kernel(program, "RGBA2HistScore");
		this->kernelMoments = cl::Kernel(program, "moments");
		this->queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		this->trackRect = cv::Rect(0, 0, TRACK_RECT_W, TRACK_RECT_H);
		this->trackRectBorderColor = cvScalar(200, 0, 100);

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
	cv::Rect border = trackRect;
	int borderWidth = 2;
	border.x -= borderWidth;
	border.y -= borderWidth;
	border.width += 2*borderWidth;
	border.height += 2*borderWidth;
	cv::rectangle(mat, border, trackRectBorderColor, borderWidth);
}

void CamShift::startTracking(cv::Mat& mat)
{
	this->tracking = true;
	getTrackedObjHist(mat);
}

void CamShift::stopTracking()
{
	this->tracking = false;
	this->trackRect.width = TRACK_RECT_W;
	this->trackRect.height = TRACK_RECT_H;
}

void CamShift::getTrackedObjHist(cv::Mat& mat)
{
	// Img width
	int w = mat.cols;
	// Img height
	int h = mat.rows;

	// ZMIANA BGR -> RGBA
	uchar* dataRGBA = new uchar[mat.total()*4];
	cv::Mat matRGBA(mat.size(), CV_8UC4, dataRGBA);
	cv::cvtColor(mat, matRGBA, CV_BGR2RGBA, 4);

	// ZMIANA RGBA -> RB Histogram indeks
	// START
	int size_img_rgba_bytes = w*h*sizeof(cl_uchar4);
	int size_rect_rg_bytes = this->trackRect.width * this->trackRect.height * sizeof(cl_uchar);

	cl::Buffer img(context, CL_MEM_READ_ONLY, size_img_rgba_bytes);
	cl::Buffer out(context, CL_MEM_READ_WRITE, size_rect_rg_bytes);

	queue.enqueueWriteBuffer(img, CL_TRUE, 0, size_img_rgba_bytes, matRGBA.data);

	// Szerokoœæ zmniejszamy 4-krotnie bo kernel analizuje po 4 jednoczeœnie
	// Liczba pikseli musi byæ podzielna przez 4, no ale ka¿da szanuj¹ca siê
	// rozdzielczoœæ spe³nia ten warunek.
	const int w4 = this->trackRect.width / 4;
	cl::NDRange global(w4, this->trackRect.height);
	cl::NDRange local = cl::NullRange;
	cl::NDRange offset = cl::NDRange(0, 0);

	this->kernelRGBA2RG_HIST_IDX_4.setArg(0, sizeof(cl_uint *), &img);
	this->kernelRGBA2RG_HIST_IDX_4.setArg(1, sizeof(cl_uchar4 *), &out);
	this->kernelRGBA2RG_HIST_IDX_4.setArg(2, w4);
	this->kernelRGBA2RG_HIST_IDX_4.setArg(3, w);
	this->kernelRGBA2RG_HIST_IDX_4.setArg(4, this->trackRect.x/4);
	this->kernelRGBA2RG_HIST_IDX_4.setArg(5, this->trackRect.y);

	queue.enqueueNDRangeKernel(this->kernelRGBA2RG_HIST_IDX_4, offset, global, local);
	queue.finish();
	// END

#ifndef __CS_DEBUG_OFF__

	std::cout << "global size: " << this->trackRect.width * this->trackRect.height/4 << "\n";
	std::cout << "rect X: " << this->trackRect.x << "\n";
	std::cout << "rect Y: " << this->trackRect.y << "\n";

	uchar* dataRG = new uchar[size_rg_bytes];
	queue.enqueueReadBuffer(out, CL_TRUE, 0, size_rg_bytes, dataRG);
	// Wypisuje 10 pierwszych wyników przekszta³cenia BGR -> RGBA -> R+16*G(indeks na histogramie 16x16 RxG)
	// dla rêcznego sprawdzenia poprawnoœci obliczeñ
	for(int i = 100; i < 150; i++)
	{		
		std::cout << i << "\t Hist idx (RxG): " << int(dataRG[i]) << "\n";
		std::cout << i+this->trackRect.y*w+this->trackRect.x
			<< "\tRGBA: \t R:" << int(matRGBA.data[4*(i+this->trackRect.y*w+this->trackRect.x)])
			<< "\t G:" << int(matRGBA.data[1+4*(i+this->trackRect.y*w+this->trackRect.x)])
			<< "\t B:" << int(matRGBA.data[2+4*(i+this->trackRect.y*w+this->trackRect.x)])
			<< "\t A:" << int(matRGBA.data[3+4*(i+this->trackRect.y*w+this->trackRect.x)]) << "\n";
		std::cout << i+this->trackRect.y*w+this->trackRect.x 
			<< "\tBGR: \t R:" << int(mat.data[3*(i+this->trackRect.y*w+this->trackRect.x)+2])
			<< "\t G:" << int(mat.data[3*(i+this->trackRect.y*w+this->trackRect.x)+1])
			<< "\t B:" << int(mat.data[3*(i+this->trackRect.y*w+this->trackRect.x)]) << "\n";		
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
	int maxCPU = 0;
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		if(histCPU[i] > maxCPU)
		{
			maxCPU = histCPU[i];
		}
	}
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		histCPU[i] = (histCPU[i] * 255)/maxCPU;
	}

#endif

	const int workGroupSize = HISTOGRAM_LEVELS;
	const int n4VectorsPerWorkItem = 1;
	// Rozmiar pix to jeden bajt.
	// Wczytujemy jednoczeœnie 4 int'y, zatem 16 bajtów.
	const int n4Vectors = this->trackRect.width * this->trackRect.height / 16;
	// Rozmiar: liczba wektorów uint4 przez liczbe wektorów na workItem
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

	// Pobieram histogramy wygenerowane
	// przez ka¿d¹ workGroup
	cl_uint* dataHIST = new cl_uint[nWorkGroups * HISTOGRAM_LEVELS];
	queue.enqueueReadBuffer(globalHist, CL_TRUE, 0, size_global_rg_hist_bytes, dataHIST);

	// TODO: Przenieœc do oddzielnej metody/metod (START, END)
	// START

	// Redukcja do jednego histogramu + maksimum dla normalizacji.
	int max = 0;
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		if(!(i <= 88 && i >= 81)) // Indeksy które odpowiadaj¹ odcieniom szaroœci s¹ pomijane.
		{
			int bin = 0;
			for(int j = 0; j< nWorkGroups;j++)
			{
				bin += dataHIST[i + j*HISTOGRAM_LEVELS];
			}			
			dataHIST[i] = max(0, bin);
			if(bin > max)
			{
				max = bin;
			}
		}
		else
		{
			dataHIST[i] = 0;
		}
	}

	// Normalizacja do 255

	// Dodatkowy fix histogramu.
	// Wszystkie wartoœci mniejsze od fix s¹ sprowadzane do zera.
	// Eliminujemy szumy tym samym niepotrzebne szumy, nieznaczne kolory.
	const int fix = 20;
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		dataHIST[i] = (dataHIST[i] * 255)/max;
		if(dataHIST[i] <= fix)
		{
			dataHIST[i] = 0;
		}
	}
	// END

	// Zapis
	memcpy(this->trackedObjHist, dataHIST, HISTOGRAM_LEVELS * sizeof(cl_uint));

#ifndef __CS_DEBUG_OFF__

	std::cout << "HISTOGRAM trackedObjHist\n";
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		std::cout << i << ": " << this->trackedObjHist[i] << "\n";
	}
	// Suma przyznanych w histogramach punktów
	int sumGPU = 0;
	for(int i = 0; i < HISTOGRAM_LEVELS; i++)
	{
		sumGPU += int(dataHIST[i]);
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

#endif
}

void CamShift::process(cv::Mat& mat)
{
	try {
		int s = meanShift(mat);
		resizeTrackRect(mat, s);
		drawTrackRect(mat);
	} catch(...)
	{
		std::rethrow_exception(std::current_exception());
	}
}

void CamShift::resizeTrackRect(cv::Mat& mat, int resize_rate)
{
	const int w = mat.cols;
	const int h = mat.rows;
	
	int s = max(MIN_TRACK_RECT_W, min(resize_rate, MAX_TRACK_RECT_W));
	int d = (s - this->trackRect.width) / 2;
	this->trackRect.width = s;
	this->trackRect.height = s;

	this->trackRect.x = min(max(this->trackRect.x - d, 0), w - this->trackRect.width);
	this->trackRect.y = min(max(this->trackRect.y - d, 0), h - this->trackRect.height);
}

int CamShift::meanShift(cv::Mat& mat)
{
	const int w = mat.cols;
	const int h = mat.rows;

	// RGBA to Histogram Score
	// START

	// ZMIANA BGR -> RGBA
	cl_uint * dataRGBA = new cl_uint[mat.total()];
	cv::Mat matRGBA(mat.size(), CV_8UC4, dataRGBA);
	cv::cvtColor(mat, matRGBA, CV_BGR2RGBA, 4);

	cl::Buffer img_rgba(context, CL_MEM_READ_ONLY, w * h * sizeof(cl_uchar4));
	cl::Buffer hist(context, CL_MEM_READ_ONLY, sizeof(cl_uint) * HISTOGRAM_LEVELS);
	cl::Buffer img_score(context, CL_MEM_READ_WRITE, w * h * sizeof(float));

	queue.enqueueWriteBuffer(img_rgba, CL_TRUE, 0, w * h * sizeof(cl_uchar4), matRGBA.data);		
	queue.enqueueWriteBuffer(hist, CL_TRUE, 0, sizeof(cl_uint) * HISTOGRAM_LEVELS, this->trackedObjHist);		

	this->kernelRGBA2HistScore.setArg(0, sizeof(cl_uint *), &img_rgba);
	this->kernelRGBA2HistScore.setArg(1, w);
	this->kernelRGBA2HistScore.setArg(2, sizeof(cl_float *), &img_score);
	this->kernelRGBA2HistScore.setArg(3, sizeof(cl_uint *), &hist);

	cl::NDRange global = cl::NDRange(w, h);
	cl::NDRange local = cl::NullRange;
	cl::NDRange offset = cl::NDRange(0, 0);

	queue.enqueueNDRangeKernel(this->kernelRGBA2HistScore, offset, global, local);
	queue.finish();

	// END
	// RGBA to Histogram Score

	// MEANSHIFT
	// START
#ifndef __CS_DEBUG_OFF__
	float * data_img_score = new float[w*h];
	queue.enqueueReadBuffer(img_score, CL_TRUE, 0, w * h * sizeof(float), data_img_score);
#endif
	//MOMENTY GPU USTAWIENIA
	const int local_scratch_size = 128;
	const int nWorkGroups = 40;
	float * reduction_result_host = new float[nWorkGroups * 4];
	cl::Buffer reduction_result(context, CL_MEM_READ_WRITE, nWorkGroups * sizeof(cl_float4));
	cl_float4 result;

	const int niters = 100;
	int s = this->trackRect.width;
	for( int i = 0; i < niters; i++ )
	{     
		this->trackRect.width = max(this->trackRect.width, 1);
		this->trackRect.height = max(this->trackRect.height, 1);

		// MOMENTS
		// START

		//GPU
		//START
		this->kernelMoments.setArg(0, sizeof(cl_float *), &img_score);
		this->kernelMoments.setArg(1, cl::Local(local_scratch_size * sizeof(cl_float4)));
		this->kernelMoments.setArg(2, this->trackRect.width * this->trackRect.height);
		this->kernelMoments.setArg(3, this->trackRect.width);
		this->kernelMoments.setArg(4, w);
		this->kernelMoments.setArg(5, this->trackRect.x);
		this->kernelMoments.setArg(6, this->trackRect.y);
		this->kernelMoments.setArg(7, sizeof(cl_float4 *), &reduction_result);

		global = cl::NDRange(nWorkGroups * local_scratch_size);
		local = cl::NDRange(local_scratch_size);
		offset = cl::NDRange(0);

		queue.enqueueNDRangeKernel(this->kernelMoments, offset, global, local);
		queue.finish();
		queue.enqueueReadBuffer(reduction_result, CL_TRUE, 0, nWorkGroups * sizeof(cl_float4), reduction_result_host);
		result = common::reduceHost(reduction_result_host, nWorkGroups * 4);		
		//END
		//GPU

#ifndef __CS_DEBUG_OFF__
		//CPU
		//START
		float m01, m10, m00, tmp;
		m00 = m01 = m10 = 0;		

		for(int i = 0; i < this->trackRect.width; i++)
		{
			for(int j = 0; j < this->trackRect.height; j++)
			{
				tmp = data_img_score[(i + this->trackRect.x)+w*(j + this->trackRect.y)];
				m00 += tmp;
				m10 += i * tmp;
				m01 += j * tmp;
			}
		}
		// END
		// CPU
		if(fabs(m00 - result.s[0]) > 0.01*m00
			|| fabs(m10 - result.s[1]) > 0.01*m10
			|| fabs(m01 - result.s[2]) > 0.01*m01)
		{
			std::cout << "m00 CPU:" << m00 << "\t GPU:" << result.s[0] << "\n";
			std::cout << "m10 CPU:" << m10 << "\t GPU:" << result.s[1] << "\n";
			std::cout << "m01 CPU:" << m01 << "\t GPU:" << result.s[2] << "\n";
		}
#endif
		// END
		// MOMENTS		

		if( fabs(result.s[0]) < DBL_EPSILON )
			break;

		// Zmienna potrzebna do zmiany rozmiaru okna
		s = int((1.8*sqrt(result.s[0]/235))/4.0)*4;

		int dx = cvRound( result.s[1]/result.s[0] - this->trackRect.width*0.5 );
		int dy = cvRound( result.s[2]/result.s[0] - this->trackRect.height*0.5 );

		int nx = min(max(this->trackRect.x + dx, 0), w - this->trackRect.width);
		int ny = min(max(this->trackRect.y + dy, 0), h - this->trackRect.height);

		dx = nx - this->trackRect.x;
		dy = ny - this->trackRect.y;
		this->trackRect.x = nx;
		this->trackRect.y = ny;

		if( dx*dx + dy*dy < DBL_EPSILON*DBL_EPSILON)
			break;
	}
	delete[] reduction_result_host;
#ifndef __CS_DEBUG_OFF__
	delete[] data_img_score;
#endif
	// END
	// MEANSHIFT
	return s;
}

