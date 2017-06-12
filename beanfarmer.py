from __future__ import print_function
"""
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
"""

import sys
from subprocess import Popen,PIPE

DEBUG = False
VERBOSE = False
SOURCE_FILE = "./beanfarmer_dp4a_noshfl_k.cu"
EXECUTABLE = "./beanfarmer_autogen"

class Tee(object):
    def __init__(self,filename):
        self.f = open(filename,"w")

    def write(self,msg):
        sys.stdout.write(msg)
        self.f.write(msg)

    def __del__(self):
        self.f.close()
        
class Parameters(object):
    def __init__(self):
        self.nantennas = 48
        self.nbeams = 32 * 32
        self.npol = 2
        self.naccumulate = 16
        self.nchannels = 64
        self.tsamp = 4096/856.0e6
        self.nchunks = 100
        self.warp_size = 32
        self.nthreads = 1024
        self.check_correctness = False
        self.niterations = 100

    def sample_count(self):
        return self.nthreads/self.warp_size * self.naccumulate * self.nchunks

    def duration(self):
        return self.sample_count() * self.tsamp

    def _validate(self):
        if self.nantennas%4 != 0:
            raise ValueError("Number of antennas must be a multiple of 4")
        if 32%self.warp_size != 0:
            raise ValueError("Warp size must divide 32")
        if self.nbeams%32 !=0:
            raise ValueError("Number of beams must be a multiple of 32")
        if self.nthreads%self.warp_size !=0:
            raise ValueError("Number of threads must be a multiple of the warp size")

    def op_count(self):
        per_sample = (8*self.nantennas - 1) * self.nbeams
        ops = per_sample * self.sample_count() * self.npol * self.nchannels
        return ops
        
    def to_file(self,):
        self._validate()
        f = open("params.h","w")
        print("#define WARP_SIZE %d"%self.warp_size,file=f)
        print("#define NTHREADS %d"%self.nthreads,file=f)
        print("#define NWARPS_PER_BLOCK NTHREADS/WARP_SIZE",file=f)
        print("#define NANTENNAS %d"%self.nantennas,file=f)
        print("#define NPOL %d"%self.npol,file=f)
        print("#define NBEAMS %d"%self.nbeams,file=f)
        print("#define NACCUMULATE %d"%self.naccumulate,file=f)
        print("#define NSAMPLES_PER_BLOCK (NACCUMULATE * NTHREADS/WARP_SIZE)",file=f)
        print("#define NSAMPLES (NSAMPLES_PER_BLOCK * %d)"%self.nchunks,file=f)
        print("#define NCHANNELS %d"%self.nchannels,file=f)
        print("#define TSAMP %f"%self.tsamp,file=f)
        print("#define NITERATIONS %d"%self.niterations,file=f)
        if self.check_correctness:
            print("#define TEST_CORRECTNESS",file=f)
        f.close()
        
def eprint(*args, **kwargs):
    if VERBOSE:
        print(*args, file=sys.stderr, **kwargs)
        
def compile(params):

    if DEBUG:
        args = ["nvcc", "-std=c++11", "-G", "-g",
                SOURCE_FILE, "-o", EXECUTABLE]
    else:
        args = ["nvcc", "-O3", "-restrict",
                "-use_fast_math", "-std=c++11",
                "-arch=sm_61", SOURCE_FILE,
                "-o", EXECUTABLE]

    eprint("Writing updated headers...")
    params.to_file()
        
    eprint("Compiling...")
    p = Popen(args,stdout=PIPE,stderr=PIPE)
    p.wait()
    for line in p.stdout.read().decode().splitlines():
        eprint(line)
    if p.returncode != 0:
        raise Exception(p.stderr.read().decode())
    return p

def run():
    eprint("Running compiled executable...")
    p = Popen([EXECUTABLE],stdout=PIPE,stderr=PIPE)
    p.wait()
    if p.returncode != 0:
        raise Exception(p.stderr)
    lines = [line.decode() for line in p.stdout.read().splitlines()]
    out = None
    for line in lines:
        eprint("[beanfarmer output]: ",line)
        if line.startswith("Total kernel duration (ms):"):
            out = float(line.split(":")[-1])
    return out

def integer_list(string):
    try:
        int_list = [int(val) for val in string.split(",")]
    except Exception as e:
        raise argparse.ArgumentTypeError(str(e))
    return int_list

def main(args):
    f = Tee(args.outfile)
    print("#Nants",end="  ",file=f)
    print("#Nbeams",end="  ",file=f)
    print("#Npol",end="  ",file=f)
    print("#Nchans",end="  ",file=f)
    print("#Naccumulate",end="  ",file=f)
    print("#Nchunks",end="  ",file=f)
    print("#Nthreads",end="  ",file=f)
    print("#Nops",end="  ",file=f)
    print("#Benchmark(s)",end="  ",file=f)
    print("#Performance(Tops/s)",end="  ",file=f)
    print("#Realtime_fraction",end="\n",file=f)
    for nbeams in args.nbeams:
        for nants in args.nantennas:
            for npol in args.npol:
                for nacc in args.naccumulate:
                    for nchan in args.nchannels:
                        for nchunks in args.nchunks:
                            for nwarps_per_block in args.nwarps_per_block:
                                params = Parameters()
                                params.nbeams = nbeams
                                params.nantennas = nants
                                params.npol = npol
                                params.naccumulate = nacc
                                params.nchannels = nchan
                                params.nchunks = nchunks
                                params.nthreads = nwarps_per_block * 32
                                params.tsamp = 1/args.channel_bandwidth
                                params.check_correctness = args.check_correctness
                                params.niterations = args.niterations
                                compile(params)
                                elapsed_time = run()
                                if elapsed_time is None: raise Exception("No timing output for kernel run.")
                                seconds_per_run = (elapsed_time*1e-3)/params.niterations
                                real_time_fraction = seconds_per_run/params.duration()
                                performance = params.op_count()/seconds_per_run
                                tops = performance/1e12
                                print(nants,end="\t",file=f)
                                print(nbeams,end="\t",file=f)
                                print(npol,end="\t",file=f)
                                print(nchan,end="\t",file=f)
                                print(nacc,end="\t",file=f)
                                print(nchunks,end="\t",file=f)
                                print(params.nthreads,end="\t",file=f)
                                print("%g"%params.op_count(),end="\t",file=f)
                                print("%g"%seconds_per_run,end="\t",file=f)
                                print("%g"%tops,end="\t",file=f)
                                print("%g"%real_time_fraction,end="\n",file=f)
                                
    
if __name__ == "__main__":
    import sys
    import argparse

    msg = """
    Benchmarking tool for dp4a-based filterbanking beamformer application.

    Arguments can be passed as comma separated lists.
    """
    
    parser = argparse.ArgumentParser(usage=msg)
    parser.add_argument("--nantennas", type=integer_list, default="32", required=False,
                        help="Number of antennas.")
    parser.add_argument("--nbeams", type=integer_list, default="512", required=False,
                        help="Number of beams.")
    parser.add_argument("--npol", type=integer_list, default="2", required=False,
                        help="Number of polarisations.")
    parser.add_argument("--naccumulate", type=integer_list, default="16", required=False,
                        help="Number of time samples to add.")
    parser.add_argument("--nchannels", type=integer_list, default="64", required=False,
                        help="Number of channels")
    parser.add_argument("--nchunks", type=integer_list, default="100", required=False,
                        help="Number of sample chunks to process in a batch")
    parser.add_argument("--nwarps_per_block", type=integer_list, default="32", required=False,
                        help="Number of warps per block. [NOTE: Currently there is a bug that means tests only pass when this is set to 32.]")
    parser.add_argument("--check_correctness", action="store_true", required=False,
                        help="Verify results against C++ implementation.")
    parser.add_argument("--channel_bandwidth", type=float, default=856e6/4096.0, required=False,
                        help="Bandwidth of each channel being processed. Used to work out real-time fracion.")
    parser.add_argument("--niterations", type=int, default=100, required=False,
                        help="Number of iterations used for timing "
                        "(10 iterations will always be used for warm-up).")
    parser.add_argument("-o","--outfile", type=str, default="beanfarmer_benchmarking.txt", required=False,
                        help="Output filename for benchmarking results.")
    parser.add_argument("-v","--verbose", action="store_true", required=False,
                        help="Turn on verbose messages.")
    parser.add_argument("--debug", action="store_true", required=False,
                        help="Build with debug flags.")
    args = parser.parse_args()
    VERBOSE = args.verbose
    DEBUG = args.debug
    main(args)
