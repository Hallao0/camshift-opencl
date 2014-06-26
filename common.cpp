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

	/**
	* Algorytm sumacyjny Kahana dla ostatecznej redukcji (obliczenia momentów).
	*/
	cl_float4 reduceHost(float * data, const int size)
	{
		cl_float4 result = {0.,0.,0.,0.};
		float c00 = 0.0f;   
		float c10 = 0.0f; 
		float c01 = 0.0f; 
		for (int i = 0; i < size; i+=4)
		{
			float y = data[i] - c00;  
			float t = result.s[0] + y;      
			c00 = (t - result.s[0]) - y;  
			result.s[0] = t;     

			y = data[i+1] - c10;  
			t = result.s[1] + y;      
			c10 = (t - result.s[1]) - y;  
			result.s[1] = t;  

			y = data[i+2] - c01;  
			t = result.s[2] + y;      
			c01 = (t - result.s[2]) - y;  
			result.s[2] = t;  
		}
		return result;
	}

}