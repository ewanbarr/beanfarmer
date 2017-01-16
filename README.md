# beanfarmer
Benchmarking for a CUDA based beamformer implementation

To use, run beanfarmer.py with any of the following arguments.  

~~~~
Arguments can be passed as comma separated lists.
optional arguments:
  -h, --help            show this help message and exit
  --nantennas NANTENNAS
                        Number of antennas.
  --nbeams NBEAMS       Number of beams.
  --npol NPOL           Number of polarisations.
  --naccumulate NACCUMULATE
                        Number of time samples to add.
  --nchannels NCHANNELS
                        Number of channels
  --nchunks NCHUNKS     Number of sample chunks to process in a batch
  --nwarps_per_block NWARPS_PER_BLOCK
                        Number of warps per block. [NOTE: Currently there is a
                        bug that means tests only pass when this is set to
                        32.]
  --check_correctness   Verify results against C++ implementation.
  --channel_bandwidth CHANNEL_BANDWIDTH
                        Bandwidth of each channel being processed. Used to
                        work out real-time fracion.
  --niterations NITERATIONS
                        Number of iterations used for timing (10 iterations
                        will always be used for warm-up).
  -o OUTFILE, --outfile OUTFILE
                        Output filename for benchmarking results.
  -v, --verbose         Turn on verbose messages.
  --debug               Build with debug flags.
~~~~

Typical example:

~~~~
python beanfarmer.py --nchannels=1 --verbose --nbeams=128,512 --nantennas=32,48,64 --nchunks=100 --npol=2 
~~~~