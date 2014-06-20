#include "common.hpp"

namespace common 
{
	std::string get_file_content(const std::string filename)
	{
		std::ifstream in;
		in.exceptions(std::ifstream::badbit | std::ifstream::failbit);
		try
		{
			in.open(filename.c_str(), std::ios::in | std::ios::binary);
			if(in.is_open())
			{
				in.exceptions(std::ifstream::badbit);
				std::string contents;
				in.seekg(0, std::ios::end);
				contents.resize(in.tellg());
				in.seekg(0, std::ios::beg);
				in.read(&contents[0], contents.size());
				in.close();				
				return contents;
			}
			else
			{
				throw std::logic_error("Error occurred while opening file: " + filename);
			}
		} catch (...)
		{
			throw std::logic_error("Error occurred while opening file: " + filename);
		}
	}
}