#include "mpi.h"
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

herr_t ierr;
hid_t  file_id, dset_id, grp_id, fapl;
hid_t  filespace, memspace;
hid_t  plist_id, dcpl_id;

// Variables and dimensions
long      numparticles = 8388608; // 8  meg particles per process
long long total_particles, offset;

float *x, *y, *z;
float *px, *py, *pz;
int *  id1, *id2;
int    x_dim = 64;
int    y_dim = 64;
int    z_dim = 64;
hid_t  dsets[8];

// Uniform random number
inline double
uniform_random_number()
{
    return (((double)rand()) / ((double)(RAND_MAX)));
}

// Initialize particle data
void
init_particles()
{
    int i;
    for (i = 0; i < numparticles; i++) {
        id1[i] = i;
        id2[i] = i * 2;
        x[i]   = uniform_random_number() * x_dim;
        y[i]   = uniform_random_number() * y_dim;
        z[i]   = ((double)id1[i] / numparticles) * z_dim;
        px[i]  = uniform_random_number() * x_dim;
        py[i]  = uniform_random_number() * y_dim;
        pz[i]  = ((double)id2[i] / numparticles) * z_dim;
    }
}

void
create_h5_datasets()
{
    dsets[0] = H5Dcreate(grp_id, "x", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    dsets[1] = H5Dcreate(grp_id, "y", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    dsets[2] = H5Dcreate(grp_id, "z", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    dsets[3] = H5Dcreate(grp_id, "id1", H5T_NATIVE_INT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    dsets[4] = H5Dcreate(grp_id, "id2", H5T_NATIVE_INT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    dsets[5] = H5Dcreate(grp_id, "px", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    dsets[6] = H5Dcreate(grp_id, "py", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    dsets[7] = H5Dcreate(grp_id, "pz", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
}

void
write_h5_datasets()
{
    ierr = H5Dwrite(dsets[0], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, x);
    assert(ierr == 0);
    ierr = H5Dwrite(dsets[1], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, y);
    assert(ierr == 0);
    ierr = H5Dwrite(dsets[2], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, z);
    assert(ierr == 0);
    ierr = H5Dwrite(dsets[3], H5T_NATIVE_INT, memspace, filespace, plist_id, id1);
    assert(ierr == 0);
    ierr = H5Dwrite(dsets[4], H5T_NATIVE_INT, memspace, filespace, plist_id, id2);
    assert(ierr == 0);
    ierr = H5Dwrite(dsets[5], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, px);
    assert(ierr == 0);
    ierr = H5Dwrite(dsets[6], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, py);
    assert(ierr == 0);
    ierr = H5Dwrite(dsets[7], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, pz);
    assert(ierr == 0);
}

void
close_h5_datasets()
{
    H5Dclose(dsets[0]);
    H5Dclose(dsets[1]);
    H5Dclose(dsets[2]);
    H5Dclose(dsets[3]);
    H5Dclose(dsets[4]);
    H5Dclose(dsets[5]);
    H5Dclose(dsets[6]);
    H5Dclose(dsets[7]);
}

// Create HDF5 file and write data
void
create_and_write_synthetic_h5_data(int rank)
{
    // Note: printf statements are inserted basically
    // to check the progress. Other than that they can be removed
    dset_id = H5Dcreate(file_id, "x", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr    = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, x);
    H5Dclose(dset_id);
    if (rank == 0)
        printf("Written variable 1 \n");

    dset_id = H5Dcreate(file_id, "y", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr    = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, y);
    H5Dclose(dset_id);
    if (rank == 0)
        printf("Written variable 2 \n");

    dset_id = H5Dcreate(file_id, "z", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr    = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, z);
    H5Dclose(dset_id);
    if (rank == 0)
        printf("Written variable 3 \n");

    dset_id = H5Dcreate(file_id, "id1", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr    = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, id1);
    H5Dclose(dset_id);
    if (rank == 0)
        printf("Written variable 4 \n");

    dset_id = H5Dcreate(file_id, "id2", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr    = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, id2);
    H5Dclose(dset_id);
    if (rank == 0)
        printf("Written variable 5 \n");

    dset_id = H5Dcreate(file_id, "px", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr    = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, px);
    H5Dclose(dset_id);
    if (rank == 0)
        printf("Written variable 6 \n");

    dset_id = H5Dcreate(file_id, "py", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr    = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, py);
    H5Dclose(dset_id);
    if (rank == 0)
        printf("Written variable 7 \n");

    dset_id = H5Dcreate(file_id, "pz", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr    = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, pz);
    H5Dclose(dset_id);
    if (rank == 0)
        printf("Written variable 8 \n");
}

int
main(int argc, char *argv[])
{
    int      my_rank, num_procs, nstep, i, sleeptime;
    MPI_Comm comm = MPI_COMM_WORLD;
    double   t0, t1, t2, t3, tw = 0;
    char  grp_name[128];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &num_procs);

    if (argc >= 3)
        numparticles = (atoi(argv[2])) * 1024 * 1024;
    else
        numparticles = 8 * 1024 * 1024;

    if (argc >= 4)
        nstep = atoi(argv[3]);
    else
        nstep = 5;

    if (argc >= 5)
        sleeptime = atoi(argv[4]);
    else
        sleeptime = 0;

    if (my_rank == 0) {
        fprintf(stderr, "Number of paritcles: %ld \n", numparticles);
        fprintf(stderr, "Number of steps: %d \n", nstep);
        fprintf(stderr, "Sleep time: %d \n", sleeptime);
    }

    x = (float *)malloc(numparticles * sizeof(double));
    y = (float *)malloc(numparticles * sizeof(double));
    z = (float *)malloc(numparticles * sizeof(double));

    px = (float *)malloc(numparticles * sizeof(double));
    py = (float *)malloc(numparticles * sizeof(double));
    pz = (float *)malloc(numparticles * sizeof(double));

    id1 = (int *)malloc(numparticles * sizeof(int));
    id2 = (int *)malloc(numparticles * sizeof(int));

    init_particles();

    if (my_rank == 0)
        printf("Finished initializing particles \n");

    MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, comm);
    offset -= numparticles;

    fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Alignment and metadata block size
    int alignment = 16 * 1024 * 1024;
    int blocksize = 4 * 1024 * 1024;
    H5Pset_alignment(fapl, alignment, alignment);
    H5Pset_meta_block_size(fapl, blocksize);

    // Collective metadata ops
    H5Pset_coll_metadata_write(fapl, true);
    H5Pset_all_coll_metadata_ops(fapl, true);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    /* hsize_t chunk_dims[1]; */
    /* chunk_dims[0] = (hsize_t)numparticles; */
    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    /* H5Pset_chunk(dcpl_id, 1, chunk_dims); */

    memspace = H5Screate_simple(1, (hsize_t *)&numparticles, NULL);
    MPI_Barrier(comm);
    t0 = MPI_Wtime();

    file_id = H5Fcreate(argv[1], H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    assert(file_id >= 0);

    if (my_rank == 0)
        printf("Created HDF5 file [%s] \n", argv[1]);

    for (i = 0; i < nstep; i++) {
        sprintf(grp_name, "Timestep_%d", i);
        grp_id = H5Gcreate(file_id, grp_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        filespace = H5Screate_simple(1, (hsize_t *)&total_particles, NULL);
        create_h5_datasets();

        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, (hsize_t *)&offset, NULL, (hsize_t *)&numparticles,
                            NULL);

        MPI_Barrier(comm);
        t1 = MPI_Wtime();

        write_h5_datasets();

        MPI_Barrier(comm);
        t2 = MPI_Wtime();
        if (my_rank == 0)
            printf("Wrote one file, took %.2f\n", t2 - t1);
        tw += (t2 - t1);

        close_h5_datasets();

        H5Gclose(grp_id);
        H5Sclose(filespace);

        if (my_rank == 0)
            printf("Sleep %d\n", sleeptime);
        fflush(stdout);
        if (i < nstep - 1)
            sleep(sleeptime);
    }

    H5Sclose(memspace);
    H5Pclose(plist_id);
    H5Pclose(fapl);
    H5Pclose(dcpl_id);
    H5Fclose(file_id);

    MPI_Barrier(comm);
    t3 = MPI_Wtime();
    if (my_rank == 0)
        printf("Wrote %d steps, took %.2f, actual write took %.2f\n", nstep, t3 - t0, tw);

    free(x);
    free(y);
    free(z);
    free(px);
    free(py);
    free(pz);
    free(id1);
    free(id2);

    MPI_Finalize();
    return 0;
}
