#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <hdf5.h>

#define ANAME  "Float attribute"     /* Name of the array attribute */
#define ANAMES "Character attribute" /* Name of the string attribute */
#define ADIM1  2
#define ADIM2  3

double
uniform_random_number()
{
    return (((double)rand()) / ((double)(RAND_MAX)));
}

int
main(int argc, char *argv[])
{
    hid_t    file_id, group_id, fapl_id, dxpl_id;
    hid_t    dset_id1, dset_id2, dset_id3, dset_id4, dset_id5, dset_id6, dset_id7, dset_id8;
    hid_t    r_dset_id1, r_dset_id2;
    hid_t    filespace, memspace;
    hid_t    attr1;
    hid_t    aid1;
    herr_t   ierr;
    int      point_out; /* Buffer to read scalar attribute back */
    int      point = 1;
    int      loop;
    MPI_Comm comm;
    int      my_rank, num_procs;
    my_rank   = 0;
    num_procs = 1;
    // Variables and dimensions
    long long numparticles = 8388608; // 8  meg particles per process
    /* long      numparticles = 4; */
    long long total_particles, offset;

    float *x, *y, *z;
    float *px, *py, *pz;
    int *  id1, *id2;
    int    x_dim = 64;
    int    y_dim = 64;
    int    z_dim = 64;
    int    i;
    double stime, etime;

    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &num_procs);

    MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, comm);
    offset -= numparticles;
    offset = 0;
    if (my_rank == 0)
        printf("Number of paritcles: %lld \n", numparticles);

    x   = (float *)malloc(numparticles * sizeof(float));
    y   = (float *)malloc(numparticles * sizeof(float));
    z   = (float *)malloc(numparticles * sizeof(float));
    px  = (float *)malloc(numparticles * sizeof(float));
    py  = (float *)malloc(numparticles * sizeof(float));
    pz  = (float *)malloc(numparticles * sizeof(float));
    id1 = (int *)malloc(numparticles * sizeof(int));
    id2 = (int *)malloc(numparticles * sizeof(int));

    for (i = 0; i < numparticles; i++) {
        id1[i] = i;
        id2[i] = i * 2;
        x[i]   = uniform_random_number() * x_dim;
        y[i]   = uniform_random_number() * y_dim;
        z[i]   = ((float)id1[i] / numparticles) * z_dim;
        px[i]  = uniform_random_number() * x_dim;
        py[i]  = uniform_random_number() * y_dim;
        pz[i]  = ((float)id2[i] / numparticles) * z_dim;
    }

    /* Set up FAPL */
    if ((fapl_id = H5Pcreate(H5P_FILE_ACCESS)) < 0)
        printf("H5Pcreate() error\n");
    H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    memspace  = H5Screate_simple(1, (hsize_t *)&numparticles, NULL);
    filespace = H5Screate_simple(1, (hsize_t *)&total_particles, NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    stime = MPI_Wtime();

    /* Create file */
    if ((file_id = H5Fcreate(argv[1], H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id)) < 0)
        printf("H5Fcreate() error\n");

    /* Create group */
    if ((group_id = H5Gcreate2(file_id, "group1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0)
        printf("H5Gcreate() error\n");

    /* Create dataset */
    dset_id1 = H5Dcreate(file_id, "x", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id2 = H5Dcreate(file_id, "y", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id3 = H5Dcreate(file_id, "z", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id4 = H5Dcreate(file_id, "px", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id5 = H5Dcreate(file_id, "py", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id6 = H5Dcreate(file_id, "pz", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id7 = H5Dcreate(file_id, "id1", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id8 = H5Dcreate(file_id, "id2", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    if (my_rank == 0)
        printf("File/Group/Dataset create took %.2f seconds\n", etime - stime);

    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, (hsize_t *)&offset, NULL, (hsize_t *)&numparticles, NULL);

    MPI_Barrier(comm);
    stime = MPI_Wtime();

    ierr = H5Dwrite(dset_id1, H5T_NATIVE_FLOAT, memspace, filespace, dxpl_id, x);
    if (ierr < 0)
        printf("write x failed\n");
    ierr = H5Dwrite(dset_id2, H5T_NATIVE_FLOAT, memspace, filespace, dxpl_id, y);
    if (ierr < 0)
        printf("write y failed\n");
    ierr = H5Dwrite(dset_id3, H5T_NATIVE_FLOAT, memspace, filespace, dxpl_id, z);
    if (ierr < 0)
        printf("write z failed\n");
    ierr = H5Dwrite(dset_id4, H5T_NATIVE_FLOAT, memspace, filespace, dxpl_id, px);
    if (ierr < 0)
        printf("write px failed\n");
    ierr = H5Dwrite(dset_id5, H5T_NATIVE_FLOAT, memspace, filespace, dxpl_id, py);
    if (ierr < 0)
        printf("write py failed\n");
    ierr = H5Dwrite(dset_id6, H5T_NATIVE_FLOAT, memspace, filespace, dxpl_id, pz);
    if (ierr < 0)
        printf("write pz failed\n");
    ierr = H5Dwrite(dset_id7, H5T_NATIVE_INT, memspace, filespace, dxpl_id, id1);
    if (ierr < 0)
        printf("write id1 failed\n");
    ierr = H5Dwrite(dset_id8, H5T_NATIVE_INT, memspace, filespace, dxpl_id, id2);
    if (ierr < 0)
        printf("write id2 failed\n");

    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    if (my_rank == 0)
        printf("Dataset write took %.2f seconds\n", etime - stime);

    r_dset_id1 = H5Dopen2(file_id, "id1", H5P_DEFAULT);
    r_dset_id2 = H5Dopen2(file_id, "id2", H5P_DEFAULT);

    ierr = H5Dread(r_dset_id1, H5T_NATIVE_INT, memspace, filespace, dxpl_id, id1);
    if (ierr < 0)
        printf("read dset1 failed\n");

    ierr = H5Dread(r_dset_id2, H5T_NATIVE_INT, memspace, filespace, dxpl_id, id2);
    if (ierr < 0)
        printf("read r_dset2 failed\n");

    if (H5Dclose(r_dset_id1) < 0)
        printf("H5Dclose dataset1 error\n");

    if (H5Dclose(r_dset_id2) < 0)
        printf("H5Dclose dataset2 error\n");

    for (loop = 0; loop < numparticles; loop++) {
        if (id1[loop] != loop) {
            fprintf(stderr, "Error id1: %d / %d\n", id1[loop], loop);
            break;
        }
    }

    for (loop = 0; loop < numparticles; loop++) {
        if (id2[loop] != loop * 2) {
            fprintf(stderr, "Error id2: %d / %d\n", id2[loop], loop * 2);
            break;
        }
    }

    /* create attribute */
    aid1  = H5Screate(H5S_SCALAR);
    attr1 = H5Acreate2(dset_id1, "Integer attribute", H5T_NATIVE_INT, aid1, H5P_DEFAULT, H5P_DEFAULT);

    /*
     * Write scalar attribute.
     */
    ierr = H5Awrite(attr1, H5T_NATIVE_INT, &point);
    if (ierr < 0)
        printf("write attr1 failed\n");

    ierr = H5Aread(attr1, H5T_NATIVE_INT, &point_out);
    if (ierr < 0)
        printf("read attr1 failed\n");

    if (point != point_out) {
        printf("read attribute value is incorrect\n");
    }
    /* close attribute */
    ierr = H5Aclose(attr1);
    if (ierr < 0)
        printf("close attr1 failed\n");

    /* Close */
    if (H5Dclose(dset_id1) < 0)
        printf("H5Dclose dataset1 error\n");
    if (H5Dclose(dset_id2) < 0)
        printf("H5Dclose dataset2 error\n");
    if (H5Dclose(dset_id3) < 0)
        printf("H5Dclose dataset3 error\n");
    if (H5Dclose(dset_id4) < 0)
        printf("H5Dclose dataset4 error\n");
    if (H5Dclose(dset_id5) < 0)
        printf("H5Dclose dataset5 error\n");
    if (H5Dclose(dset_id6) < 0)
        printf("H5Dclose dataset6 error\n");
    if (H5Dclose(dset_id7) < 0)
        printf("H5Dclose dataset7 error\n");
    if (H5Dclose(dset_id8) < 0)
        printf("H5Dclose dataset8 error\n");

    if (H5Sclose(memspace) < 0)
        printf("H5Sclose memspace error\n");
    if (H5Sclose(filespace) < 0)
        printf("H5Sclose filespace error\n");
    if (H5Gclose(group_id) < 0)
        printf("H5Gclose error\n");
    if (H5Fclose(file_id) < 0)
        printf("H5Fclose error\n");
    if (H5Pclose(fapl_id) < 0)
        printf("H5Pclose error\n");
    if (H5Pclose(dxpl_id) < 0)
        printf("H5Pclose error\n");

    if (my_rank == 0) {
        printf("Success\n");
    }

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
