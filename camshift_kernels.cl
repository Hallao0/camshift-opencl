__kernel void test(
	__global uchar * src,
	__global uchar * dst,
	const int width)
{    
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);

	int idx = ix + iy * width;

	dst[idx] = min(src[idx] + 100, 255);
}

