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


__global__
void icbf_taftp_general_k
(
    char4 const* __restrict__ taftp_voltages,
    float* __restrict__ tf_powers
)
{
    static_assert(NSAMPLES_PER_TIMESTAMP % IBF_TSCRUNCH == 0,
        "tscrunch must divide 256");
    static_assert(NCHANNELS % IBF_FSCRUNCH == 0,
        "Fscrunch must divide nchannels");

    const int output_size = NSAMPLES_PER_TIMESTAMP/IBF_TSCRUNCH * NCHANNELS/IBF_FSCRUNCH;
    volatile __shared__ float accumulation_buffer[NCHANNELS/IBF_FSCRUNCH][NSAMPLES_PER_TIMESTAMP];
    volatile __shared__ char output_staging[output_size];

    //TAFTP
    const int tp = NSAMPLES_PER_TIMESTAMP;
    const int ftp = NCHANNELS * tp;
    const int aftp = NANTENNAS * ftp;
    const int output_offset = output_size * blockIdx.x;

    for (int timestamp_idx = blockIdx.x; timestamp_idx < NTIMESTAMPS; timestamp_idx += gridDim.x)
    {

        for (int sample_idx = threadIdx.x; sample_idx < NSAMPLES_PER_TIMESTAMP; sample_idx += blockDim.x)
        {

            float xx = 0.0f, yy = 0.0f, zz = 0.0f, ww = 0.0f;

            //Must start with the right number of threads in the y dimension
            for (int channel_idx = IBF_FSCRUNCH * threadIdx.y;
                channel_idx < channel_idx + IBF_FSCRUNCH;
                ++channel_idx)
            {
                for (int antenna_idx = 0; antenna_idx < NANTENNAS; ++antenna_idx)
                {
                    int input_index = timestamp_idx * aftp + antenna_idx * ftp + channel_idx * tp + sample_idx;
                    char4 ant = taftp_voltages[input_index];
                    xx += ((float) ant.x) * ant.x;
                    yy += ((float) ant.y) * ant.y;
                    zz += ((float) ant.z) * ant.z;
                    ww += ((float) ant.w) * ant.w;
                }
            }
            accumulation_buffer[threadIdx.y][sample_idx] = (xx + yy + zz + ww);
        }

        __threadfence_block();

        if (threadIdx.x < NSAMPLES_PER_TIMESTAMP/IBF_TSCRUNCH)
        {
            float val = 0.0f;
            for (int sample_idx = threadIdx.x * IBF_TSCRUNCH; sample_idx < (threadIdx.x + 1) * IBF_TSCRUNCH; ++sample_idx)
            {
                val += accumulation_buffer[threadIdx.y][sample_idx];
            }
            output_staging[threadIdx.x * gridDim.y + threadIdx.y] = val;
        }

        __threadfence_block();
        for (int idx = threadIdx.x; idx < output_size; idx += gridDim.x)
        {
            tf_powers[output_offset + idx] = output_staging[idx];
        }
    }
}


void icbf_reference_cpp
(
    char4 const* __restrict__ taftp_voltages,
    float* __restrict__ tf_powers)
{
    const int tp = NSAMPLES_PER_TIMESTAMP;
    const int ftp = NCHANNELS * tp;
    const int aftp = NANTENNAS * ftp;

    for (int timestamp_idx = 0; timestamp_idx < NTIMESTAMPS; ++timestamp_idx)
    {
        for (int antenna_idx = 0; antenna_idx < NANTENNAS; ++antenna_idx)
        {
            for (int subband_idx = 0; subband_idx < NCHANNELS/IBF_FSCRUNCH; ++subband_idx)
            {
                int subband_start = subband_idx * IBF_FSCRUNCH;
                for (int subint_idx = 0; subint_idx < NSAMPLES_PER_TIMESTAMP/IBF_TSCRUNCH; ++subint_idx)
                {
                    int subint_start = subint_idx * IBF_TSCRUNCH;
                    float xx = 0.0f, yy = 0.0f, zz = 0.0f, ww = 0.0f;
                    for (int channel_idx = subband_start; channel_idx < subband_start + IBF_FSCRUNCH;  ++channel_idx)
                    {
                        for (int sample_idx = subint_start; sample_idx < subint_start + IBF_TSCRUNCH; ++sample_idx)
                        {
                            int input_index = timestamp_idx * aftp + antenna_idx * ftp + channel_idx * tp + sample_idx;
                            char4 ant = taftp_voltages[input_index];
                            xx += ((float) ant.x) * ant.x;
                            yy += ((float) ant.y) * ant.y;
                            zz += ((float) ant.z) * ant.z;
                            ww += ((float) ant.w) * ant.w;
                        }
                    }
                    int time_idx = timestamp_idx * NSAMPLES_PER_TIMESTAMP/IBF_TSCRUNCH + subint_idx;
                    int output_idx = time_idx * NCHANNELS/IBF_FSCRUNCH + subband_idx;
                    tf_powers[output_idx] = (xx + yy + zz + ww);
                }
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
	  std::cout << "Error at index " << idx << "\n";
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
    const int NSAMPLES = (NTIMESTAMPS * NSAMPLES_PER_TIMESTAMP);
    std::size_t taftp_voltages_size =  NTIMESTAMPS * NANTENNAS * NCHANNELS * NSAMPLES_PER_TIMESTAMP * NPOL;
    std::size_t tf_powers_size = NSAMPLES/IBF_TSCRUNCH * NCHANNELS/IBF_FSCRUNCH;
    std::cout << "TAFTP array size: " << taftp_voltages_size << "\n";
    std::cout << "output size: " << tf_powers_size << "\n";
    std::cout << "Global memory required: "
	    << (tf_powers_size * sizeof(float)
        + taftp_voltages_size*sizeof(ComplexInt8))/1.0e9
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
  thrust::host_vector<ComplexInt8> taftp_vector_h(taftp_voltages_size, default_value);
  populate<ComplexInt8>(taftp_vector_h.data(), taftp_voltages_size, -10, 10);
  thrust::device_vector<ComplexInt8> taftp_vector = taftp_vector_h;
#else
  std::cout << "NOT generating host test vectors...\n";
  thrust::device_vector<ComplexInt8> taftp_vector(taftp_voltages_size);
#endif //TEST_CORRECTNESS

  thrust::device_vector<float> output_vector(tf_powers_size,0.0f);
  ComplexInt8 const* taftp_voltages = thrust::raw_pointer_cast(taftp_vector.data());
  float* tf_powers = thrust::raw_pointer_cast(output_vector.data());

  int nblocks = NTIMESTAMPS;
  dim3 threads(NSAMPLES_PER_TIMESTAMP, NCHANNELS/IBF_FSCRUNCH, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Executing warm up\n";
  //Warm up
  for (int jj = 0; jj < NITERATIONS; ++jj)
    icbf_taftp_general_k<<<nblocks, threads>>>((char4*)taftp_voltages, tf_powers);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  std::cout << "Starting benchmarking\n";
  cudaEventRecord(start);
  for (int ii = 0; ii < NITERATIONS; ++ii)
    icbf_taftp_general_k<<<nblocks, threads>>>((char4*)taftp_voltages, tf_powers);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Total kernel duration (ms): " << milliseconds << "\n";

#ifdef TEST_CORRECTNESS
  std::cout << "Testing correctness...\n";
  thrust::host_vector<float> gpu_output = output_vector;
  thrust::host_vector<float> cpu_output(tf_powers_size);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  icbf_reference_cpp(taftp_vector_h.data(), cpu_output.data());
  if (!is_same(cpu_output.data(), gpu_output.data(), tf_powers_size, 0.001))
    std::cout << "FAILED!\n";
  else
    std::cout << "PASSED!\n";

#endif

  return 0;
}
