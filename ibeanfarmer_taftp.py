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
SOURCE_FILE = "./ibeanfarmer_taftp_k.cu"
EXECUTABLE = "./ibeanfarmer_taftp_autogen"
NSAMPLES_PER_TIMESTAMP = 256

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
        self.npol = 2
        self.ibf_tscrunch = 16
        self.ibf_fscrunch = 1
        self.nchannels = 64
        self.tsamp = 4096/856.0e6
        self.ntimestamps = 100
        self.warp_size = 32
        self.nthreads = 1024
        self.check_correctness = False
        self.niterations = 100

    def sample_count(self):
        return self.nthreads/self.warp_size * self.ibf_tscrunch * self.ntimestamps

    def duration(self):
        return self.sample_count() * self.tsamp

    def _validate(self):
        if self.nantennas%2 != 0:
            raise ValueError("Number of antennas must be a multiple of 4")
        if 32%self.warp_size != 0:
            raise ValueError("Warp size must divide 32")
        if self.nthreads%self.warp_size !=0:
            raise ValueError("Number of threads must be a multiple of the warp size")

    def op_count(self):
        per_sample = (4*self.nantennas - 1)
        ops = per_sample * self.sample_count() * self.npol * self.nchannels
        return ops

    def to_file(self,):
        self._validate()
        f = open("params.h","w")
        print("#define NANTENNAS %d"%self.nantennas,file=f)
        print("#define NPOL %d"%self.npol,file=f)
        print("#define IBF_TSCRUNCH %d"%self.ibf_tscrunch,file=f)
        print("#define IBF_FSCRUNCH %d"%self.ibf_fscrunch,file=f)
        print("#define NSAMPLES_PER_TIMESTAMP %d"%NSAMPLES_PER_TIMESTAMP,file=f)
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
    print("#Npol",end="  ",file=f)
    print("#Nchans",end="  ",file=f)
    print("#Tscrunch",end="  ",file=f)
    print("#Fscrunch",end="  ",file=f)
    print("#Ntimestamps",end="  ",file=f)
    print("#Nops",end="  ",file=f)
    print("#Benchmark(s)",end="  ",file=f)
    print("#Performance(Tops/s)",end="  ",file=f)
    print("#Realtime_fraction",end="\n",file=f)
    npol = 2
    for nants in args.nantennas:
        for fscrunch in args.ibf_fscrunch:
            for tscrunch in args.ibf_tscrunch:
                for nchan in args.nchannels:
                    for ntimestamps in args.ntimestamps:
                        for nwarps_per_block in args.nwarps_per_block:
                            params = Parameters()
                            params.nantennas = nants
                            params.npol = npol
                            params.ibf_tscrunch = tscrunch
                            params.nchannels = nchan
                            params.ntimestamps = ntimestamps
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
                            print(npol,end="\t",file=f)
                            print(nchan,end="\t",file=f)
                            print(tscrunch,end="\t",file=f)
                            print(fscrunch,end="\t",file=f)
                            print(ntimestamps,end="\t",file=f)
                            print("%g"%params.op_count(),end="\t",file=f)
                            print("%g"%seconds_per_run,end="\t",file=f)
                            print("%g"%tops,end="\t",file=f)
                            print("%g"%real_time_fraction,end="\n",file=f)


if __name__ == "__main__":
    import sys
    import argparse

    msg = """
    Benchmarking tool for incoherent filterbanking beamformer application.

    Arguments can be passed as comma separated lists.
    """

    parser = argparse.ArgumentParser(usage=msg)
    parser.add_argument("--nantennas", type=integer_list, default="32", required=False,
                        help="Number of antennas.")
    parser.add_argument("--ibf_tscrunch", type=integer_list, default="16", required=False,
                        help="Number of time samples to add.")
    parser.add_argument("--ibf_fscrunch", type=integer_list, default="1", required=False,
                        help="Number of frequency channels to add.")
    parser.add_argument("--nchannels", type=integer_list, default="64", required=False,
                        help="Number of channels")
    parser.add_argument("--ntimestamps", type=integer_list, default="100", required=False,
                        help="Number of sample chunks to process in a batch")
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
