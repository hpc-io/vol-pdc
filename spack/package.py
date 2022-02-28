# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class VolPdc(CMakePackage):
    """Proactive Data Containers (PDC) software provides an object-centric
    API and a runtime system with a set of data object management services.
    These services allow placing data in the memory and storage hierarchy,
    performing data movement asynchronously, and providing scalable
    metadata operations to find data objects."""
    
    homepage = "https://github.com/hpc-io/vol-pdc"
    url      = "https://github.com/hpc-io/vol-pdc/archive/refs/tags/v0.1.tar.gz"

    maintainers = ['houjun']


    version('0.1', sha256='1619b5defc4b5988f93ca0a8ec06518bf38f48d0bc66cd7db3612ea9f3e4f298')

    conflicts('%clang')
    depends_on('hdf5@1.12')
    depends_on('pdc')

    root_cmakelists_dir = 'src'

    def cmake_args(self):
        args = [
            self.define('MPI_C_COMPILER', self.spec['mpi'].mpicc),
            self.define('BUILD_SHARED_LIBS', 'ON')
        ]
        if self.spec.satisfies('platform=cray'):
            args.append("-DRANKSTR_LINK_STATIC=ON")
        return args
