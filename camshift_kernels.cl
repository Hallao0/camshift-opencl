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

__kernel void RGBAtoRG_4(
	__global uchar16 * src,
	__global uchar8 * dst)
{    
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);
	const int width = get_global_size(0);

	int idx = ix + iy * width;	
	uchar16 rgba4 = src[idx];
	uint8 rg4 = convert_uint8_sat(rgba4.s014589cd);
	rg4 *= (uint8)(15);
	rg4 /= (uint8)(255);
	dst[idx] = convert_uchar8_sat(rg4);
}

#define R_LEVELS 16
#define G_LEVELS 16

#define MASK_RIGHT_MOST_BYTE (uint4)(255)

//TODO: Lepsze nazewnictwo kernela

__kernel void RGBAtoRxG16_4(
	__global uint4 * src,
	__global uchar4 * dst)
{    	
	const uint idx = get_global_id(0) +  get_global_id(1) * get_global_size(0);	
	uint4 rgba4 = src[idx];	
	
	// OBLICZAM SUME R+G+B dla ka¿dego pix
	// dodaje r
	uint4 r = (rgba4) & MASK_RIGHT_MOST_BYTE;
	//uint4 sum_rgb = r;
	// dodaje g
	uint4 g = (rgba4 >> 8) & MASK_RIGHT_MOST_BYTE;
	//sum_rgb += g;
	// dodaje b
	//sum_rgb += (rgba4 >> 16) & MASK_RIGHT_MOST_BYTE;
		
	// 16 * g
	uint4 rg_4 = g * (uint4)(16);
	// r + 16 * g
	rg_4 += r;
	// (r + 16 * g)
	// Dzielenie, konwersja do uchar4 i zapis
	// 15*(R+16*G) / (255)
	dst[idx] = convert_uchar4_sat(rg_4/(uint4)(17));
}

// 256 kube³ków (16x16)
#define HIST_BINS 256

// Ile banków pamiêci u¿ywamy
// Ze wzglêdu na to, ¿e SIMD wykonuje jednoczeœnie æwieræ wavefrontu (16 workitemów)
// i bêdziemy u¿ywaæ operacji atomowy optymalnie wychodzi 16 chocia¿ jest 32.
#define NBANKS 16

__kernel void histRG(
	__global uint4 * srcRG,
	__global uint * globalHistRG,
	uint n4VectorsPerWorkItem)
{
	__local uint subhists[NBANKS * HIST_BINS];

	uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	uint Stride = get_global_size(0);
	
	uint4 tmp1, tmp2;	 
}