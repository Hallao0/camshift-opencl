#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <CL\cl.hpp>

namespace common
{
	std::string get_file_content(const std::string filename);
	cl_float4 reduceHost(float * data, const int size);
}