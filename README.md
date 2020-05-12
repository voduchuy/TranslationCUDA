# TranslationCUDA

An attempt to accelerate the sampling of translation model trajectories using CUDA-enabled GPUs.

## System requirements

1) Your computer need to have access to a CUDA-enabled GPU. The CUDA Toolkit must be installed on your system (see instructions from NVidia website https://developer.nvidia.com/cuda-toolkit for how to download and install CUDA toolkit).

2) CMake version 3.8 or higher. Make sure you can run CMake executable (i.e., the command 'cmake') from terminal.

3) GNU Make.

## Installation instructions

To install this library and the example programs, clone this repository to your local hard drive. Open a terminal/command line prompt. Follow these steps for a quick start:
1) Create a build directory, say 'build'. Change directory there.
2) To generate build files, use the command

```Shell
  cmake -DCMAKE_BUILD_TYPE=<build_type> \
  -DCMAKE_INSTALL_PREFIX=<path to where you want to install library and header files> \
  <path to TranslationCUDA source directory>
 ```
    
Here, build_type is either Release or Debug. If you omit the ```DCMAKE_INSTALL_PREFIX``` option, the library will be installed to the system default folder (on Linux/Mac, it is '/usr/local/').

3) Type 'make'. Wait for the compilation to finish.
4) Type 'make install'. In some systems, you may need administrative privilege (e.g., 'sudo' in Linux) for the files to be proprely copied to the destination folder.

### Additional build options

```-DBUILD_TESTS=<ON/OFF>``` whether to build the test programs.\
```-DBUILD_EXAMPLES=<ON/OFF>``` whether to build the example programs.\
```-DTESTS_INSTALL_DIR=<path>``` directory to install the test programs (if chosen to build tests).\
```-DEXAMPLES_INSTALL_DIR=<path>``` directory to install the example programs (if chosen to build examples).

For example, my machine is running Ubuntu Linux. If I have cloned this repository to a local folder called 'cutrans' in my home directory, and I want to install my libraries to '/usr/local/' but my examples and tests to a subfolder of Home called 'cool_programs', I'll use

```
cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/usr/local/ \
-DBUILD_TESTS=ON -DTESTS_INSTALL_DIR=~/cool_programs/ \
-DBUILD_EXAMPLES=ON -DEXAMPLES_INSTALL_DIR=~/cool_programs/
~/cutrans
```

## Running the demo program

The easiest way to run the demo program is to copy all .txt files in the folder 'examples/demo_sample_inputs' from the source directory into the folder that you installed the examples. Then just run 
```./demo```
Just press enter for all questions the program ask you, until it asks about the number of samples and the ribosome exclusion parameter. Then wait for a few seconds, all trajectories will be written to a file called 'output.txt', where line i column j stores the signal intensity of sample i at the j-th timepoint.

The filenames are self-explanatory. The file 'x0.txt' stores the initial ribosome locations. If x[i] = 0, that means the i-th ribosome has not been initiated yet. If x[i] = 1 then the ribosome has just been initiated.

Note that 'rates.txt' contain ```(gene_len + 1)``` entries, where ```gene_len``` is the number of codons in the gene. Therefore, the first entry of rates.txt is initiation, and the rest are elongation rates.

The 'c.txt' stores the probe design vector entries. Note that in my code c is assumed to also have ```(gene_len + 1)``` entry, and it always start with 0 (free ribosome emits no light). 





      
      
