#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define __CL_ENABLE_EXCEPTIONS
#include <CL\cl.hpp>

#include "defines.hpp"
#include "CamShift.hpp"

using namespace cv;
using namespace std;

int main() {	
	try {
		cvNamedWindow("main",CV_WINDOW_NORMAL);

		//0 is the id of video device.0 if you have only one camera.
		VideoCapture stream(0);  
		//check if video device has been initialised
		if (!stream.isOpened()) {
			cout << "Cannot open camera.";
		}

		CamShift camshift;
		Mat cameraFrame;

		stream >> cameraFrame;	
		std::cout << cameraFrame.size() << std::endl;
		std::cout << cameraFrame.rows << std::endl;
		std::cout << cameraFrame.cols << std::endl;

		while(cvWaitKey(33) != 27)
		{
			stream >> cameraFrame;	
			camshift.process(cameraFrame.data, cameraFrame.rows, cameraFrame.cols);
			imshow("main", cameraFrame);
		}	

		cvDestroyAllWindows();
		return EXIT_SUCCESS;
	}
	catch (cl::Error &e) {
		std::cerr << "OpenCL error: " << e.what() << ".\n";
	}
	catch (cv::Exception &e) {
		std::cerr << "OpenCV exception: " << e.what() << ".\n";
	}
	catch (std::exception &e) {
		std::cerr << "STD exception: "<< e.what() << ".\n";
	}
	catch (...) {
		std::cerr << "Unhandled exception of unknown type reached the top of main.\n";		
	}
	getchar();
	return EXIT_FAILURE;
}
