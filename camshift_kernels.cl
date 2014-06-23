
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

__kernel void RGBA2RG_HIST_IDX_4(
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

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

// 256 kube³ków (16x16)
#define HIST_BINS 256
// Ile banków pamiêci u¿ywamy
// Ze wzglêdu na to, ¿e SIMD wykonuje jednoczeœnie æwieræ wavefrontu (16 workitemów)
// i bêdziemy u¿ywaæ operacji atomowy optymalnie wychodzi 16 chocia¿ jest 32.
#define NBANKS 16
#define BITS_PER_VALUE 8

__kernel __attribute__((reqd_work_group_size(HIST_BINS,1,1)))
void histRG(
	__global uint4 * srcRG,
	__global uint * globalHistRG,
	uint n4VectorsPerWorkItem)
{
	__local uint subhists[NBANKS * HIST_BINS];

	uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	uint Stride = get_global_size(0);
	
	const uint shift = BITS_PER_VALUE;
	const uint offset = lid % (uint)(NBANKS);
	uint4 tmp1, tmp2;	 
	
	// ZERUJE __local subhists
    uint localItemsPerWorkItem = NBANKS * HIST_BINS / get_local_size(0);
	uint localWorkItems = get_local_size(0);
	// Zerujemy po 4 jednoczesnie, bedzie szybciej i tak siegamy od innych banków
	__local uint4 *p = (__local uint4 *) subhists; 
    if( lid < localWorkItems )
    {
       for(uint i=0, idx=lid; i<localItemsPerWorkItem/4; i++, idx+=localWorkItems)
       {
          p[idx] = 0;
       }
    }
	barrier( CLK_LOCAL_MEM_FENCE );

	// Przegl¹dam "obrazek" i wype³niam lokalny histogram
	for(uint i=0, idx=gid; i<n4VectorsPerWorkItem; i++, idx += Stride )
    {
       tmp1 = srcRG[idx];
       tmp2 = (tmp1 & MASK_RIGHT_MOST_BYTE) * (uint4) NBANKS + offset;

       (void) atom_inc( subhists + tmp2.x );
       (void) atom_inc( subhists + tmp2.y );
       (void) atom_inc( subhists + tmp2.z );
       (void) atom_inc( subhists + tmp2.w );

       tmp1 = tmp1 >> shift;
       tmp2 = (tmp1 & MASK_RIGHT_MOST_BYTE) * (uint4) NBANKS + offset;

       (void) atom_inc( subhists + tmp2.x );
       (void) atom_inc( subhists + tmp2.y );
       (void) atom_inc( subhists + tmp2.z );
       (void) atom_inc( subhists + tmp2.w );

       tmp1 = tmp1 >> shift;
       tmp2 = (tmp1 & MASK_RIGHT_MOST_BYTE) * (uint4) NBANKS + offset;
       
       (void) atom_inc( subhists + tmp2.x );
       (void) atom_inc( subhists + tmp2.y );
       (void) atom_inc( subhists + tmp2.z );
       (void) atom_inc( subhists + tmp2.w );

       tmp1 = tmp1 >> shift;
       tmp2 = (tmp1 & MASK_RIGHT_MOST_BYTE) * (uint4) NBANKS + offset;
       
       (void) atom_inc( subhists + tmp2.x );
       (void) atom_inc( subhists + tmp2.y );
       (void) atom_inc( subhists + tmp2.z );
       (void) atom_inc( subhists + tmp2.w );
    }
    barrier( CLK_LOCAL_MEM_FENCE );
	
	// Sumuje 16 lokalnych histogramów w jeden histogram dla ca³ej workGroup
	// Sumuje tak, ze ka¿dy z 256 w¹tków sumuje po jednym "kube³ku"(?) histogramu
	for(uint binIdx = lid; binIdx < HIST_BINS; binIdx += localWorkItems)
	{
		uint bin = 0;
		for( uint i = 0; i < NBANKS; i++)
		{
			bin += subhists[(lid*NBANKS) + ((i+lid) % NBANKS)];
		}
		globalHistRG[(get_group_id(0) * HIST_BINS) + binIdx] = bin;
	}	 
}
