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

  cmake -DCMAKE_BUILD_TYPE=<build_type> \
  -DCMAKE_INSTALL_PREFIX=<path to where you want to install library and header files> \
  <path to TranslationCUDA source directory>
    
Here, build_type is either Release or Debug.

3) Type 'make'. Wait for the compilation to finish.
4) Type 'make install'. In some systems, you may need administrative privilege (e.g., 'sudo' in Linux) for the files to be proprely copied to the destination folder.




      
      
