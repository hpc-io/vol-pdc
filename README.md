# Installation Instructions for Cori Machines

## Install libfabric
```
0. wget https://github.com/ofiwg/libfabric/archive/v1.11.2.tar.gz
1. tar xvzf v1.11.2.tar.gz
2. cd libfabric-1.11.2
3. mkdir install
4. export LIBFABRIC_DIR=$(pwd)/install
5. ./autogen.sh
6. ./configure --prefix=$LIBFABRIC_DIR CC=cc CFLAG="-O2"
7. make -j8
8. make install
9. export LD_LIBRARY_PATH="$LIBFABRIC_DIR/lib:$LD_LIBRARY_PATH"
10. export PATH="$LIBFABRIC_DIR/include:$LIBFABRIC_DIR/lib:$PATH"
```

## Install Mercury
Make sure the ctest passes. PDC may not work without passing all the tests of Mercury.

Step 2 in the following is not required. It is a stable commit that has been used to test when these these instructions were written. One may skip it to use the current master branch of Mercury.
```
0. git clone https://github.com/mercury-hpc/mercury.git
1. cd mercury
2. git checkout e741051fbe6347087171f33119d57c48cb438438
3. git submodule update --init
4. export MERCURY_DIR=$(pwd)/install
5. mkdir install
6. cd install
7. cmake ../ -DCMAKE_INSTALL_PREFIX=$MERCURY_DIR -DCMAKE_C_COMPILER=cc -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=ON -DNA_USE_OFI=ON -DNA_USE_SM=OFF -DCMAKE_C_FLAGS="-dynamic" -DCMAKE_CXX_FLAGS="-dynamic"
8. make
9. make install
10. ctest -DMPI_RUN_CMD=srun
11. export LD_LIBRARY_PATH="$MERCURY_DIR/lib:$LD_LIBRARY_PATH"
12. export PATH="$MERCURY_DIR/include:$MERCURY_DIR/lib:$PATH"
```
## Install PDC
ctest contains both sequential and MPI tests for the PDC settings. These can be used to perform regression tests.
```
0. git clone https://github.com/hpc-io/pdc.git
1. cd pdc
2. git checkout stable
3. cd src
4. mkdir install
5. cd install
6. export PDC_DIR=$(pwd)
7. cmake ../ -DBUILD_MPI_TESTING=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=ON -DCMAKE_INSTALL_PREFIX=$PDC_DIR -DPDC_ENABLE_MPI=ON -DMERCURY_DIR=$MERCURY_DIR -DCMAKE_C_COMPILER=cc -DCMAKE_C_FLAGS="-dynamic"
8. make -j8
9. ctest -DMPI_RUN_CMD=srun 
```
## Environmental variables
During installation, we have set some environmental variables. These variables may disappear after the close the current session ends.
We recommend adding the following lines to ~/.bashrc. (One may also execute them manually after logging in).
The MERCURY_DIR and LIBFABRIC_DIR variables should be identical to the values that were set during the installation of Mercury and libfabric.
The install path is the path containing bin and lib directory, instead of the one containing the source code.
```
export PDC_DIR="where/you/installed/your/pdc"
export MERCURY_DIR="where/you/installed/your/mercury"
export LIBFABRIC_DIR="where/you/installed/your/libfabric"
export LD_LIBRARY_PATH="$LIBFABRIC_DIR/lib:$MERCURY_DIR/lib:$LD_LIBRARY_PATH"
export PATH="$LIBFABRIC_DIR/include:$LIBFABRIC_DIR/lib:$MERCURY_DIR/include:$MERCURY_DIR/lib:$PATH"
```

## Install HDF5
```
0. wget "https://www.hdfgroup.org/package/hdf5-1-12-1-tar-gz/?wpdmdl=15727&refresh=612559667d6521629837670"
1. mv index.html?wpdmdl=15727&refresh=612559667d6521629837670 hdf5-1.12.1.tar.gz
2. tar zxf hdf5-1.12.1.tar.gz
3. cd hdf5-1.12.1
4. ./configure --prefix=/global/homes/../<username>/hdf5
5. make
6. make check
7. make install
8. make check-install
```

## Building vol-pdc
CMakeLists.txt seem to be heavily adapted from the [mercury](https://github.com/mercury-hpc/mercury) git repo. Building should be very similar.
```
0. git clone https://github.com/hpc-io/vol-pdc.git
1. cd vol-pdc
2. mkdir build
3. cd build
4. ccmake ..
5. set BUILD_SHARED_LIBS to ON (default OFF)
6. set HDF5_DIR to where HDF5 is installed
7. set HDF5_INCLUDE_DIR to the include directory in the HDF5 package
8. set HDF5_LIBRARY to the lib directory in the HDF5 package
9. press [c] to configure then exit
10. cmake .. 
```

# Notes

The following functions have yet to be implemented and are either currently do nothing, or don't do anything relevant to the VOL:

- H5VL_pdc_attr_get
- H5VL_pdc_attr_close
- H5VL_pdc_group_close
- H5VL_pdc_group_get
- H5VL_pdc_introspect_opt_query

The following functions have been modified to work in the context of the VOL, but contain extraneous code that is either never called by the VOL or isn't relevant to the VOL:
- H5VL_pdc_dataset_get
- H5VL_pdc_file_specific
