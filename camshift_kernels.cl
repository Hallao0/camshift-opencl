#define R_LEVELS 16
#define G_LEVELS 16

#define MASK_RIGHT_MOST_BYTE_4	(uint4)(255)
#define MASK_RIGHT_MOST_BYTE	(uint)(255)

// 256 kube³ków (16x16)
#define HIST_BINS 256

__kernel void RGBA2RG_HIST_IDX_4(
	__global uint4 * src,
	__global uchar4 * dst,
	uint rgba_width)
{    	
	const uint idx = get_global_id(0) + get_global_id(1) * rgba_width;	
	const uint dst_idx = idx - (get_global_offset(0) + get_global_offset(1) * rgba_width + (get_global_id(1)-get_global_offset(1)) * 2 * get_global_offset(0));
	__global uint * src_uint = (__global uint *) src;
	uint4 rgba4 = (uint4)(src_uint[idx],src_uint[idx+1],src_uint[idx+2],src_uint[idx+3]);		
	
	// OBLICZAM SUME R+G+B dla ka¿dego pix
	// dodaje r
	uint4 r = (rgba4) & MASK_RIGHT_MOST_BYTE_4;
	uint4 sum_rgb = r;
	// dodaje g
	uint4 g = (rgba4 >> 8) & MASK_RIGHT_MOST_BYTE_4;
	sum_rgb += g;
	// dodaje b
	sum_rgb += (rgba4 >> 16) & MASK_RIGHT_MOST_BYTE_4;
		
	// 16 * g
	uint4 rg_4 = g * (uint4)(16);
	// r + 16 * g
	rg_4 += r;
	// 15 * (r + 16 * g)
	rg_4 *= 15;	
	// Dzielenie, konwersja do uchar4 i zapis
	// 15*(R+16*G) / (r+g+b)
	dst[dst_idx] = convert_uchar4_sat(rg_4/sum_rgb);
}

__kernel void RGBA2HistScore(
	__global uint * src,
	const uint width,
	__global float * dst,
	__constant uint * histogram
	)
{
	uint idx = get_global_id(0) + get_global_id(1) * width;	

	uint rgba = src[idx];
	uint r = (rgba) & MASK_RIGHT_MOST_BYTE;
	uint g = (rgba >> 8) & MASK_RIGHT_MOST_BYTE;
	// b = (rgba >> 16) & MASK_RIGHT_MOST_BYTE

	uint hist_idx = ((r + 16 * g)*15)/(r+g+(rgba >> 16) & MASK_RIGHT_MOST_BYTE);

	dst[idx] = (float)(histogram[hist_idx]);
}

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

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
       tmp2 = (tmp1 & MASK_RIGHT_MOST_BYTE_4) * (uint4) NBANKS + offset;

       (void) atom_inc( subhists + tmp2.x );
       (void) atom_inc( subhists + tmp2.y );
       (void) atom_inc( subhists + tmp2.z );
       (void) atom_inc( subhists + tmp2.w );

       tmp1 = tmp1 >> shift;
       tmp2 = (tmp1 & MASK_RIGHT_MOST_BYTE_4) * (uint4) NBANKS + offset;

       (void) atom_inc( subhists + tmp2.x );
       (void) atom_inc( subhists + tmp2.y );
       (void) atom_inc( subhists + tmp2.z );
       (void) atom_inc( subhists + tmp2.w );

       tmp1 = tmp1 >> shift;
       tmp2 = (tmp1 & MASK_RIGHT_MOST_BYTE_4) * (uint4) NBANKS + offset;
       
       (void) atom_inc( subhists + tmp2.x );
       (void) atom_inc( subhists + tmp2.y );
       (void) atom_inc( subhists + tmp2.z );
       (void) atom_inc( subhists + tmp2.w );

       tmp1 = tmp1 >> shift;
       tmp2 = (tmp1 & MASK_RIGHT_MOST_BYTE_4) * (uint4) NBANKS + offset;
       
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

/**
* Oblicza momenty m00, m10, m01 dla prostok¹tnego obszaru wewn¹trz podanego obrazu.
*/
__kernel void moments(
    __global float * img, // Caly obraz
    __local float4 * scratch,
    const uint size, // Rozmiar obszaru dla ktorego obliczamy momenty
    const uint rect_width, // Szerokosc prostok¹tengo obszaru dla którego obliczamy momenty
	const uint img_width, // Szerokoœæ ca³ego obrazka
	const uint offset_x,  // Offset x, y 
	const uint offset_y,  // Offset x, y 
    __global float4* result // Wynik czêœciowej redukcji, jeszcze trzeba dokoñczyæ redukcje po stronie hosta
	) 
{
    uint rect_idx = get_global_id(0);
    float4 accumulator = (float4) 0;

	// MOMENTY
	// START
	{		
		float frect_width = convert_float(rect_width);
			
		// Wspó³rzedne w prostok¹tnym obszarze
		int rect_y = trunc(((float)rect_idx)/frect_width);
		int rect_x = rect_idx - rect_y * rect_width;

		// Wektor do wyliczania momentow
		// m00, m10, m01, -
		float4 m = (float4)(1.0, (float)rect_x, (float)rect_y, 0.0);

		// Czêœciowa redukcja po³¹czona z obliczaniem
		// momentów m00, m10, m01.
		while (rect_idx < size) 
		{
			float4 element = m * (float4)img[rect_x + offset_x + (rect_y + offset_y) * img_width];
			accumulator += element;
			rect_idx += get_global_size(0);
    
			// Wspó³rzedne w prostok¹tnym obszarze
			rect_y = trunc(((float)rect_idx)/frect_width);
			rect_x = rect_idx - rect_y * rect_width;
			// Wektor do wyliczania momentów
			m = (float4)(1.0, (float)rect_x, (float)rect_y, 0.0);
		}
	}
	// END
	// MOMENTY

    // REDUKCJA RÓWNOLEG£A
	// START
    int lid = get_local_id(0);
    scratch[lid] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) 
    {
        if (lid < offset) 
        {
            float4 other = scratch[lid + offset];
            float4 mine = scratch[lid];
            scratch[lid] = mine + other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
	// Zapis wyniku redukcji
    if (lid == 0) 
    {
        result[get_group_id(0)] = scratch[0];
    }
	// END
	// REDUKCJA
}