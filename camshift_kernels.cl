__kernel void test(
	__global uchar * src,
	__global uchar * dst,
	const int width,
	const int change)
{    
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);

	int idx = ix + iy * width;	
	dst[idx] = min(255, max(src[idx] + change, 0));		
}

__kernel void RGBAtoRG(
	__global uchar16 * src,
	__global uchar8 * dst,
	const int width)
{    
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);

	int idx = ix + iy * width;	
	uchar16 rgba4 = src[idx];
	uint8 rg4 = convert_uint8_sat(rgba4.s014589cd);
	rg4 *= (uint8)(15);
	rg4 /= (uint8)(255);
	dst[idx] = convert_uchar8_sat(rg4);
}
