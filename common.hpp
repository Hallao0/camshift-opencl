#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

namespace common
{
	std::string get_file_content(const std::string filename);
}