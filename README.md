# beanfarmer
Benchmarking for a CUDA based filterbanking beamformer implementation

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
python beanfarmer.py --nchannels=1 --verbose --nbeams=128,512 --nantennas=32,48,64 --nchunks=2 --npol=2 --check_correctness
~~~~

This will execute beanfarmer with 1 channel, 2 polarisations, 2 chunks of time and will loop through all combinations of beams and antennas. This will also compare the beanfarmer GPU kernel output against and idealised C++ beamformer implementation. The above example will not push the performance of the code. For a more realistic performance measure try something like:

~~~~
python beanfarmer.py --nchannels=64 --nbeams=128,512,1024 --nantennas=48,64,128 --nchunks=100 --npol=2
~~~~