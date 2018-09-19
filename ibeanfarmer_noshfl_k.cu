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
#include "params.h"
#include "cuda_tools.cuh"
#include "cuda.h"
#include "cuComplex.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <stdexcept>

struct ComplexInt8
{
  int8_t x;
  int8_t y;
};

/**
 * @brief      Perform beamforming followed by detection and integration in time.
 *
 * @param      aptf_voltages  Raw voltages in antenna, polarisation, time, frequency order (fastest to slowest)
 * @param      tbf_powers     Output detected integrated powers in time, frequency order (fastest to slowest)
 */
__global__
void icbf_aptf_general_k
(
 char2 const* __restrict__ aptf_voltages,
 float* __restrict__ tf_powers,
 float const* __restrict__ weights)
{
  /**
   * Each warp reads all the data it requires and performs detection followed
   * by a warp reduce. The resultant sums are transposed and the first warp of
   * each block is left to write all results back to global memory.
   */
  static_assert(NSAMPLES%NSAMPLES_PER_BLOCK==0,
		"Kernel can only process a multiple of (NWARPS_PER_BLOCK * NACCUMULATE) samples.");
  static_assert(NTHREADS%WARP_SIZE==0,
		"Number of threads must be an integer multiple of WARP_SIZE.");

  volatile __shared__ float temp[WARP_SIZE][WARP_SIZE];
  volatile __shared__ float shared_weights[NANTENNAS];
  int const warp_idx = threadIdx.x / 0x20;
  int const lane_idx = threadIdx.x & 0x1f;
  int sample_offset = NACCUMULATE * (blockIdx.x * NWARPS_PER_BLOCK + warp_idx);
  int aptf_voltages_partial_idx = NANTENNAS * NPOL * (NSAMPLES * blockIdx.y + sample_offset);


  for (int antenna_idx = threadIdx.x; antenna_idx < NANTENNAS; antenna_idx += blockDim.x)
  {
    shared_weights[antenna_idx] = weights[antenna_idx];
  }


  //Accumulators for 8-bit complex detection and additions
  int xx = 0, yy = 0;

  for (int offset = lane_idx; offset < NANTENNAS*NPOL*NACCUMULATE; offset += WARP_SIZE)
    {
      int antenna_idx = offset % NANTENNAS;
      float weight = shared_weights[antenna_idx];
      char2 ant = aptf_voltages[aptf_voltages_partial_idx  + offset];
      xx += (ant.x * ant.x) * weight;
      yy += (ant.y * ant.y) * weight;
    }
  //Form power and write to shared memory
  temp[warp_idx][lane_idx] = (float)(xx + yy);
  __syncthreads();

  //Warp reduce
  if (lane_idx < 16)
    {
      for (int src_lane = 16; src_lane > 0; src_lane >>= 1)
	temp[warp_idx][lane_idx] += temp[warp_idx][lane_idx+src_lane];
    }
  __syncthreads();

  //Transpose shared memory
  if (lane_idx==0)
      temp[0][warp_idx] = temp[warp_idx][0];
  __syncthreads();

  //First warp writes back to global memory
  if (warp_idx==0)
    {
      int output_idx = (NWARPS_PER_BLOCK * gridDim.x) * blockIdx.y
	+ (blockIdx.x * NWARPS_PER_BLOCK + lane_idx);
      tf_powers[output_idx] = temp[0][lane_idx];
    }
}

void icbf_reference_cpp
(
 ComplexInt8 const* __restrict__ aptf_voltages,
 float* __restrict__ tf_powers,
 float const* __restrict__ weights)
{
  for (int channel_idx = 0; channel_idx < NCHANNELS; ++channel_idx)
    {
      for (int sample_idx = 0; sample_idx < NSAMPLES; sample_idx+=NACCUMULATE)
	{
	  float power = 0.0f;
	  for (int sample_offset = 0; sample_offset < NACCUMULATE; ++sample_offset)
	    {
	      for (int pol_idx = 0; pol_idx < NPOL; ++pol_idx)
		{
		  cuComplex accumulator = make_cuComplex(0.0f,0.0f);
		  for (int antenna_idx = 0; antenna_idx < NANTENNAS; ++antenna_idx)
		    {
		      int aptf_voltages_idx = NANTENNAS * NPOL * NSAMPLES * channel_idx
			+ NANTENNAS * NPOL * (sample_idx + sample_offset)
			+ NANTENNAS * pol_idx
			+ antenna_idx;
		      ComplexInt8 ant = aptf_voltages[aptf_voltages_idx];
		      power += weights[antenna_idx] * (ant.x*ant.x + ant.y*ant.y);
		    }
		}
	    }
	  int tf_powers_idx = NSAMPLES/NACCUMULATE  * channel_idx
	    + sample_idx/NACCUMULATE;
	  tf_powers[tf_powers_idx] = power;
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
  std::size_t tbf_powers_size = NSAMPLES/NACCUMULATE * NCHANNELS;
  std::cout << "PTA array size: " << aptf_voltages_size << "\n";
  std::cout << "output size: " << tbf_powers_size << "\n";
  std::cout << "Global memory required: "
	    << (tbf_powers_size * sizeof(float)
		+ aptf_voltages_size*sizeof(ComplexInt8))/1.0e9
	    << "GB \n";

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
  thrust::host_vector<float> weights_vector_h(NANTENNAS, 1.0f);
  populate<ComplexInt8>(pta_vector_h.data(),aptf_voltages_size,-10,10);
  weights_vector_h[NANTENNAS/2] = 0.0f;
  weights_vector_h[NANTENNAS/4] = 0.0f;
  thrust::device_vector<ComplexInt8> pta_vector = pta_vector_h;
  thrust::device_vector<float> weights_vector = weights_vector_h;
#else
  std::cout << "NOT generating host test vectors...\n";
  thrust::device_vector<ComplexInt8> pta_vector(aptf_voltages_size);
  thrust::device_vector<float> weights_vector(NANTENNAS, 1.0f);
#endif //TEST_CORRECTNESS

  thrust::device_vector<float> output_vector(tbf_powers_size,0.0f);
  ComplexInt8 const* aptf_voltages = thrust::raw_pointer_cast(pta_vector.data());
  float const* weights_ptr = thrust::raw_pointer_cast(weights_vector.data());
  float* tbf_powers = thrust::raw_pointer_cast(output_vector.data());
  dim3 grid(NSAMPLES/(NWARPS_PER_BLOCK*NACCUMULATE), NCHANNELS);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Executing warm up\n";
  //Warm up
  for (int jj=0; jj<NITERATIONS; ++jj)
    icbf_aptf_general_k<<<grid,NTHREADS>>>((char2*)aptf_voltages, tbf_powers, weights_ptr);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  std::cout << "Starting benchmarking\n";
  cudaEventRecord(start);
  for (int ii=0; ii<NITERATIONS; ++ii)
    icbf_aptf_general_k<<<grid,NTHREADS>>>((char2*)aptf_voltages, tbf_powers, weights_ptr);
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
  icbf_reference_cpp(pta_vector_h.data(), cpu_output.data(), weights_vector_h.data());
  if (!is_same(cpu_output.data(),gpu_output.data(), NSAMPLES/NACCUMULATE*NCHANNELS, 0.001))
    std::cout << "FAILED!\n";
  else
    std::cout << "PASSED!\n";

#endif

  return 0;
}
