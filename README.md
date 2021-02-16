# A OpenMP Target port of CloverLeaf

This is a port of [CloverLeaf](https://github.com/UoB-HPC/cloverleaf_kokkos) from MPI+Kokkos to MPI+OpenMP Target.

## Tested configurations

The program was compiled and tested on the following configurations.

TODO 

## Building

Prerequisites:

 * CentOS 7
 * cmake3
 * openmpi, opemmpi-devel
 * [devtoolset-7](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/)
 
 
First, generate a build:
 
    cmake3 -Bbuild -H. -DCMAKE_BUILD_TYPE=Release  
    
Flags: 
 * `MPI_AS_LIBRARY` - `BOOL(ON|OFF)`, enable if CMake is unable to detect the correct MPI implementation or if you want to use a specific MPI installation. Use this a last resort only as your MPI implementation may pass on extra linker flags.
   * Set `MPI_C_LIB_DIR` to  <mpi_root_dir>/lib
   * Set `MPI_C_INCLUDE_DIR` to  <mpi_root_dir>/include
   * Set `MPI_C_LIB` to the library name, for exampe: mpich for libmpich.so
 * `CXX_EXTRA_FLAGS` - `STRING`, appends extra flags that will be passed on to the compiler, applies to all configs
 * `CXX_EXTRA_LINKER_FLAGS` - `STRING`, appends extra linker flags (the comma separated list after the `-Wl` flag) to the linker, applies to all configs
 * `OMP_OFFLOAD_FLAGS` - OpenMP 4.5 target offload flags that will passed directly to the compiler and linker, see examples flag combinations below.
    * GCC+NVIDIA - `"-foffload=nvptx-none -foffload=-lm  -fno-fast-math -fno-associative-math"`
    * GCC+Radeon - `"-foffload=amdgcn-amdhsa='-march=gfx906' -foffload=-lm  -fno-fast-math -fno-associative-math"`
    * LLVM+NVIDIA - `"-fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75"`
    * ICC - `"-qnextgen -fiopenmp -fopenmp-targets=spir64"`
    * CCE+NVIDIA - `"-fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_60"`
 * `OMP_ALLOW_HOST` - `BOOL(ON|OFF)`, enabled by default, set to false if the compiler is unable to support dynamic selection of host/target devices. If disabled, running the binary with `--no-target` emits an error.



If parts of your toolchain are installed at different places, you'll have to specify it manually, for example:

    cmake3 -Bbuild -H.  \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DOMP_OFFLOAD_FLAGS="-foffload=nvptx-none -foffload=-lm  -fno-fast-math -fno-associative-math"
    
Proceed with compiling:
    
    cmake3 --build build --target cloverleaf --config Release -j $(nproc)

## Known issues

 * ICC 2021.1 Beta 20200602 requires `-DOMP_ALLOW_HOST=OFF`
 

## Running

The main `cloverleaf` executable takes a `clover.in` file as parameter and outputs `clover.out` at working directory.

For example, after successful compilation, at **project root**:

    ./build/cloverleaf --file InputDecks/clover_bm16_short.in

See [Tested configurations](#tested-configurations) for tested platforms and drivers.  

For help, use the `-h` flag:
```
Options:
  -h  --help               Print the message
      --list               List available devices
      --no-target          Use OMP fallback
      --device <INDEX>     Select device at INDEX from output of --list
      --file               Custom clover.in file (defaults to clover.in if unspecified)
```

