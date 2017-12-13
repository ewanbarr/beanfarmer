/*
Copyright (c) 2017 Ewan D. Barr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 

Maintainer: Ewan D. Barr (ebarr@mpifr-bonn.mpg.de) 
*/

#include "cuda_tools.cuh"
#include "params.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <cuda.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <iostream>

struct ComplexInt8
{
  int8_t x;
  int8_t y;
};

struct char4x2
{
    char4 x;
    char4 y;
};

struct char2x4
{
    char2 x;
    char2 y;
    char2 z;
    char2 w;
};

__forceinline__ __device__
void dp4a(int &c, const int &a, const int &b) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c)); 
#else
  char4 &a4 = *((char4*)&a);
  char4 &b4 = *((char4*)&b);
  c += a4.x*b4.x;
  c += a4.y*b4.y;
  c += a4.z*b4.z;
  c += a4.w*b4.w;
#endif
}

__forceinline__ __device__
int2 int2_transpose(int2 const &input)
{
    char2x4 a;
    char4x2 b;
    a = (*(char2x4*)&input);
    b.x.x = a.x.x;
    b.x.y = a.y.x;
    b.x.z = a.z.x;
    b.x.w = a.w.x;
    b.y.x = a.x.y;
    b.y.y = a.y.y;
    b.y.z = a.z.y;
    b.y.w = a.w.y;
    return (*(int2*)&b);
}

/**
 * @brief      Perform beamforming followed by detection and integration in time.
 *
 * @param      aptf_voltages  Raw voltages in antenna, polarisation, time, frequency order (fastest to slowest)
 * @param      apbf_weights   Beamforming weights in antenna, time, beam, frequency order (fastest to slowest)
 * @param      tbf_powers     Output detected integrated powers in time, beam, frequency order (fastest to slowest)
 */
__global__
void bf_aptf_general_k(
    int2 const* __restrict__ aptf_voltages,
    int2 const* __restrict__ apbf_weights,
    float* __restrict__ tbf_powers)
{
    /**
     * Perform compile time checks on requested beamforming parameters.
     */
    static_assert(NBEAMS%WARP_SIZE==0,
        "Kernel can only process a multiple of 32 beams.");
    static_assert(NSAMPLES%NSAMPLES_PER_BLOCK==0,
        "Kernel can only process a multiple of (NWARPS_PER_BLOCK * NACCUMULATE) samples.");
    static_assert(NTHREADS%WARP_SIZE==0,
        "Number of threads must be an integer multiple of WARP_SIZE.");
    static_assert(NANTENNAS%4==0,
        "Number of antennas must be a multiple of 4.");
    static_assert(NANTENNAS<=128,
        "Number of antennas must be less than or equal to 128.");

    /**
     * Allocated shared memory to store beamforming weights and temporary space for antenna data.
     */
    __shared__ int2 shared_apb_weights[NANTENNAS/4][NPOL][WARP_SIZE];
    __shared__ int2 shared_antennas[NTHREADS/WARP_SIZE][NANTENNAS/4];
    int const warp_idx = threadIdx.x / 0x20;
    int const lane_idx = threadIdx.x & 0x1f;

    /**
     * Each warp processes 32 beams (i.e. one beam per lane).
     */
    int const start_beam_idx = blockIdx.z * WARP_SIZE;

    /**
     * Complex multiply accumulators
     */
    int xx, yy, xy, yx;
    
    float amplitude, power = 0.0f;
    int2 antennas, weights;
    int antenna_group_idx;

    /**
     * Here we load all the beamforming weights neccessary for this block. Implicit assumption here is that we do not
     * need to change the weights over the timescale of the data processed in one block. This is almost certainly OK
     * if the input data has already been rotated to telescope boresight and we are only applying parallactic angle  
     * tracking updates.
     * 
     * The global load is coalesced 8-byte (vectorised int2). 
     */
    int const apbf_weights_offset = NANTENNAS/4 * NPOL * (NBEAMS * blockIdx.y + (WARP_SIZE * blockIdx.z + warp_idx));

    for (int pol_idx=0; pol_idx < NPOL; ++pol_idx)
    {
        for (antenna_group_idx = lane_idx; antenna_group_idx < NANTENNAS/4; antenna_group_idx+=WARP_SIZE)
        {
	    shared_apb_weights[antenna_group_idx][pol_idx][warp_idx] = int2_transpose(apbf_weights[apbf_weights_offset + pol_idx * NANTENNAS/4 + antenna_group_idx]);
        }
    }
    //wait for all weights to load.
    __syncthreads();

    /**
     * Below is the main loop of the kernel. Here the kernel reads all the antennas for a given sample and
     * computes 32 beams. Each thread computes only 1 beam and access to all the antennas required for that
     * computation is achieved via a shared memory broadcasts.
     */
    int sample_offset = NACCUMULATE * (blockIdx.x * NWARPS_PER_BLOCK + warp_idx);
    for (int sample_idx = sample_offset; sample_idx < (sample_offset + NACCUMULATE); ++sample_idx)
    {
        int aptf_voltages_partial_idx = NANTENNAS/4 * NPOL * (NSAMPLES * blockIdx.y + sample_idx);
        for (int pol_idx=0; pol_idx < NPOL; ++pol_idx)
        {
            // Set the complex accumulator to zero before adding the next polarisation
            xx = 0;
            yy = 0;
            xy = 0;
            yx = 0;

	    /** 
	     * Load all antennas antennas required for this sample into shared memory.
	     * Without an outer loop to allow for more antennas (which would also require more shared memory),
	     * this kernel is limited to a max of 32 * 4 = 128 antennas in a sub-array.
	     */ 
            if (lane_idx < NANTENNAS/4)
            {
                shared_antennas[warp_idx][lane_idx] = int2_transpose(aptf_voltages[aptf_voltages_partial_idx + lane_idx + NANTENNAS/4 * pol_idx]);
            }

	    /*Required to synchronise across all the blocks*/
	    __threadfence_block();
	    
            for (antenna_group_idx=0; antenna_group_idx < NANTENNAS/4; ++antenna_group_idx)
            {
	        //broadcast load 4 antennas
                antennas = shared_antennas[warp_idx][antenna_group_idx];
		//load corresponding 4 weights
                weights = shared_apb_weights[antenna_group_idx][pol_idx][lane_idx];
		//dp4a multiply add 
                dp4a(xx,weights.x,antennas.x);
                dp4a(yy,weights.y,antennas.y);
                dp4a(xy,weights.x,antennas.y);
                dp4a(yx,weights.y,antennas.x);
            }
            int r = xx - yy;
            int i = xy + yx;
            //be careful of overflow
	    power += (float)(r*r + i*i);
        }
    }

    /**
     * As we have looped over both polarisation and sample in the above loop we are now free to simply
     * write back to global memory. Here we write back uncoalesced to get the data in time beam order.
     * The performance penalty here is very small compared to the compute time in the rest of the kernel
     * as the total volume of data being written out is a factor of NACCUMULATE * NANTENNAS / WARP_SIZE
     * smaller than the input (e.g. for 64 antennas and 16 integrated samples this is a factor of 32).
     */
    int output_idx = (NWARPS_PER_BLOCK * gridDim.x) * (NBEAMS * blockIdx.y
        + (start_beam_idx+lane_idx))
    + sample_offset / NACCUMULATE;
    tbf_powers[output_idx] = power;
}


/**
 * Reference model implementation of beamforming used for correctness testing of  
 * kernel output.
 */
void c_reference_int8
(
 ComplexInt8 const* __restrict__ aptf_voltages,
 ComplexInt8 const* __restrict__ apbf_weights,
 float* __restrict__ tbf_powers
)
{
  int xx,yy,xy,yx;
  for (int channel_idx = 0; channel_idx < NCHANNELS; ++channel_idx)
    {
      printf("[C reference model]: processing channel %d\n",channel_idx);
      for (int sample_idx = 0; sample_idx < NSAMPLES; sample_idx+=NACCUMULATE)
	{
	  for (int beam_idx = 0; beam_idx < NBEAMS; ++beam_idx)
	    {
	      float power = 0.0f;
	      for (int sample_offset = 0; sample_offset < NACCUMULATE; ++sample_offset)
		{
		  for (int pol_idx = 0; pol_idx < NPOL; ++pol_idx)
		    {
		      int2 accumulator = {0,0}; 
		      for (int antenna_idx = 0; antenna_idx < NANTENNAS; ++antenna_idx)
			{
			  int aptf_voltages_idx = NANTENNAS * NPOL * NSAMPLES * channel_idx
			    + NANTENNAS * NPOL * (sample_idx + sample_offset)
			    + NANTENNAS * pol_idx
			    + antenna_idx;
			  ComplexInt8 datum = aptf_voltages[aptf_voltages_idx];
			  
			  int apbf_weights_idx = NANTENNAS * NPOL * NBEAMS * channel_idx
			    + NANTENNAS * NPOL * beam_idx
			    + NANTENNAS * pol_idx
			    + antenna_idx;
			  ComplexInt8 weight = apbf_weights[apbf_weights_idx];
			  
			  xx = datum.x * weight.x;
			  yy = datum.y * weight.y;
			  xy = datum.x * weight.y;
			  yx = datum.y * weight.x;
			  accumulator.x += xx - yy;
			  accumulator.y += xy + yx;
			}
		      int r = accumulator.x;
		      int i = accumulator.y;
		      power += (float)(r*r + i*i);
		    }
		}
	      int tbf_powers_idx = NSAMPLES/NACCUMULATE * NBEAMS * channel_idx
		+ NSAMPLES/NACCUMULATE * beam_idx
		+ sample_idx/NACCUMULATE;
	      tbf_powers[tbf_powers_idx] = power;
	    }
	}
    }
}

bool is_same(float* a, float*b, std::size_t size, float tolerance)
{
  for (std::size_t idx = 0; idx < size; ++idx)
    {
      if (abs((a[idx]-b[idx])/a[idx]) >= tolerance)
	{
	  std::cout << "Expected " << a[idx] << " got " << b[idx] << "\n";
	  return false;
	}
    }
  return true;
}

template <typename ComplexType>
void populate(ComplexType* data, std::size_t size, int lower, int upper)
{
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> distr(lower, upper);
  for(std::size_t n = 0 ; n < size; ++n)
    {
      data[n].x = distr(eng);
      data[n].y = distr(eng);
    }
}


int main()
{
    std::size_t aptf_voltages_size = NPOL * NSAMPLES * NANTENNAS * NCHANNELS;
    std::size_t apbf_weights_size = NANTENNAS * NPOL * NBEAMS * NCHANNELS;
    std::size_t tbf_powers_size = NSAMPLES/NACCUMULATE * NBEAMS * NCHANNELS;
    std::size_t shared_mem_size = sizeof(int2) * NANTENNAS/4 * ( NPOL * WARP_SIZE + NTHREADS/WARP_SIZE);
    std::cout << "PTA array size: " << aptf_voltages_size << "\n";
    std::cout << "Weights size: " << apbf_weights_size << "\n";
    std::cout << "output size: " << tbf_powers_size << "\n";
    std::cout << "Global memory required: "
    << (tbf_powers_size * sizeof(float)
        + apbf_weights_size*sizeof(char2)
        + aptf_voltages_size*sizeof(ComplexInt8))/1.0e9
    << "GB \n";
    std::cout << "Shared memory required: " << shared_mem_size << " bytes \n";
    

    /**
     * Currently we are only considering 4k mode on the channeliser
     */;
    float duration = TSAMP * NSAMPLES;
    std::cout << "Duration of data: " << duration << " seconds" << std::endl;

    CUDA_ERROR_CHECK(cudaSetDevice(0));
    CUDA_ERROR_CHECK(cudaDeviceReset());

    /**
     * Below we set default values for the arrays. Beamforming this data should result in
     * every output having the same value.
     *
     */
#ifdef TEST_CORRECTNESS
    std::cout << "Generating host test vectors...\n";
    ComplexInt8 default_value = {0,0};
    thrust::host_vector<ComplexInt8> pta_vector_h(aptf_voltages_size,default_value);
    thrust::host_vector<ComplexInt8> weights_vector_h(apbf_weights_size,default_value);
    populate<ComplexInt8>(pta_vector_h.data(),aptf_voltages_size,-10,10);
    populate<ComplexInt8>(weights_vector_h.data(),apbf_weights_size,-10,10);
    thrust::device_vector<ComplexInt8> pta_vector = pta_vector_h;
    thrust::device_vector<ComplexInt8> weights_vector = weights_vector_h;
#else
    std::cout << "NOT generating host test vectors...\n";
    thrust::device_vector<ComplexInt8> pta_vector(aptf_voltages_size);
    thrust::device_vector<ComplexInt8> weights_vector(apbf_weights_size);
#endif //TEST_CORRECTNESS
    
    thrust::device_vector<float> output_vector(tbf_powers_size,0.0f);
    ComplexInt8 const* aptf_voltages = thrust::raw_pointer_cast(pta_vector.data());
    ComplexInt8 const* apbf_weights = thrust::raw_pointer_cast(weights_vector.data());
    float* tbf_powers = thrust::raw_pointer_cast(output_vector.data());
    dim3 grid(NSAMPLES/(NWARPS_PER_BLOCK*NACCUMULATE), NCHANNELS, NBEAMS/WARP_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Executing warm up\n";
    //Warp up
    for (int jj=0; jj<NITERATIONS; ++jj)
      bf_aptf_general_k<<<grid,NTHREADS>>>((int2*) aptf_voltages, (int2*) apbf_weights, tbf_powers);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    std::cout << "Starting benchmarking\n";
    cudaEventRecord(start);
    for (int ii=0; ii<NITERATIONS; ++ii)
      bf_aptf_general_k<<<grid,NTHREADS>>>((int2*) aptf_voltages, (int2*) apbf_weights, tbf_powers);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total kernel duration (ms): " << milliseconds << "\n";
    
#ifdef TEST_CORRECTNESS
    std::cout << "Testing correctness...\n";
    thrust::host_vector<float> gpu_output = output_vector;
    thrust::host_vector<float> cpu_output(tbf_powers_size);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    c_reference_int8(pta_vector_h.data(),weights_vector_h.data(),cpu_output.data());
    if (!is_same(cpu_output.data(),gpu_output.data(), NSAMPLES/NACCUMULATE*NBEAMS*NCHANNELS, 0.001))
      std::cout << "FAILED!\n";
    else
      std::cout << "PASSED!\n";
    
#endif

    return 0;
}
