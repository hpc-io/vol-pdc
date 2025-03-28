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
    hid_t    file_id, group_id, fapl_id;
    hid_t    r_dset_id1, r_dset_id2, r_dset_id3, r_dset_id4, r_dset_id5, r_dset_id6, r_dset_id7, r_dset_id8;
    hid_t    filespace, memspace;
    hid_t    attr1;
    herr_t   ierr;
    int      point_out; /* Buffer to read scalar attribute back */
    int      point = 1;
    int      loop;
    MPI_Comm comm;
    int      my_rank, num_procs;
    my_rank   = 0;
    num_procs = 1;
    // Variables and dimensions
    // long numparticles = 8388608;    // 8  meg particles per process
    long      numparticles = 4;
    long long total_particles, offset;

    float *xo;
    float *x, *y, *z;
    float *px, *py, *pz;
    int *  id1, *id2;

    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &num_procs);

    MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, comm);
    offset -= numparticles;
    offset = 0;
    if (my_rank == 0)
        printf("Number of paritcles: %ld \n", numparticles);

    xo = (float *)malloc(numparticles * sizeof(float));

    x = (float *)malloc(numparticles * sizeof(float));
    y = (float *)malloc(numparticles * sizeof(float));
    z = (float *)malloc(numparticles * sizeof(float));

    px = (float *)malloc(numparticles * sizeof(float));
    py = (float *)malloc(numparticles * sizeof(float));
    pz = (float *)malloc(numparticles * sizeof(float));

    id1 = (int *)malloc(numparticles * sizeof(int));
    id2 = (int *)malloc(numparticles * sizeof(int));

    /* Set up FAPL */
    if ((fapl_id = H5Pcreate(H5P_FILE_ACCESS)) < 0)
        printf("H5Pcreate() error\n");
    H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    /* Initialize VOL */
    //    pdc_vol_id = H5VLregister_connector_by_name("pdc", H5P_DEFAULT);   // not used if choosing to use
    //    environmental variable

    /* Open a file */
    if ((file_id = H5Fopen(argv[1], H5F_ACC_RDWR, fapl_id)) < 0)
        printf("H5Fopen() error\n");

    if ((group_id = H5Gopen1(file_id, "group1")) < 0)
        printf("H5Gopen() error\n");
    if (H5Gclose(group_id) < 0)
        printf("H5Gclose error\n");

    if ((group_id = H5Oopen(file_id, "group1", H5P_DEFAULT)) < 0) {
        fprintf(stderr, "Error with H5Oopen!\n");
    }
    if (H5Oclose(group_id) < 0)
        printf("H5Oclose group error\n");

    memspace  = H5Screate_simple(1, (hsize_t *)&numparticles, NULL);
    filespace = H5Screate_simple(1, (hsize_t *)&total_particles, NULL);

    fapl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, (hsize_t *)&offset, NULL, (hsize_t *)&numparticles, NULL);

    MPI_Barrier(comm);

    r_dset_id1 = H5Dopen2(file_id, "x", H5P_DEFAULT);
    r_dset_id2 = H5Dopen2(file_id, "y", H5P_DEFAULT);
    r_dset_id3 = H5Dopen2(file_id, "z", H5P_DEFAULT);
    r_dset_id4 = H5Dopen2(file_id, "px", H5P_DEFAULT);
    r_dset_id5 = H5Dopen2(file_id, "py", H5P_DEFAULT);
    r_dset_id6 = H5Dopen2(file_id, "pz", H5P_DEFAULT);
    r_dset_id7 = H5Dopen2(file_id, "id1", H5P_DEFAULT);
    r_dset_id8 = H5Dopen2(file_id, "id2", H5P_DEFAULT);

    ierr = H5Dread(r_dset_id1, H5T_NATIVE_FLOAT, memspace, filespace, fapl_id, x);
    if (ierr < 0)
        printf("read dset1 failed\n");
    ierr = H5Dread(r_dset_id2, H5T_NATIVE_FLOAT, memspace, filespace, fapl_id, y);
    if (ierr < 0)
        printf("read dset2 failed\n");
    ierr = H5Dread(r_dset_id3, H5T_NATIVE_FLOAT, memspace, filespace, fapl_id, z);
    if (ierr < 0)
        printf("read dset3 failed\n");
    ierr = H5Dread(r_dset_id4, H5T_NATIVE_FLOAT, memspace, filespace, fapl_id, px);
    if (ierr < 0)
        printf("read dset4 failed\n");
    ierr = H5Dread(r_dset_id5, H5T_NATIVE_FLOAT, memspace, filespace, fapl_id, py);
    if (ierr < 0)
        printf("read dset5 failed\n");
    ierr = H5Dread(r_dset_id6, H5T_NATIVE_FLOAT, memspace, filespace, fapl_id, pz);
    if (ierr < 0)
        printf("read dset6 failed\n");
    ierr = H5Dread(r_dset_id7, H5T_NATIVE_INT, memspace, filespace, fapl_id, id1);
    if (ierr < 0)
        printf("read dset7 failed\n");
    ierr = H5Dread(r_dset_id8, H5T_NATIVE_INT, memspace, filespace, fapl_id, id2);
    if (ierr < 0)
        printf("read dset8 failed\n");

    fprintf(stderr, "\nx vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%f ", x[loop]);
    }
    fprintf(stderr, "\ny vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%f ", y[loop]);
    }
    fprintf(stderr, "\nz vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%f ", y[loop]);
    }
    fprintf(stderr, "\npx vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%f ", px[loop]);
    }
    fprintf(stderr, "\npy vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%f ", py[loop]);
    }
    fprintf(stderr, "\npz vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%f ", pz[loop]);
    }
    fprintf(stderr, "\nid1 vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%i ", id1[loop]);
    }
    fprintf(stderr, "\nid2 vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%i ", id2[loop]);
    }
    fprintf(stderr, "\n");

    if (H5Dclose(r_dset_id1) < 0)
        printf("H5Dclose dataset error\n");

    r_dset_id1 = H5Oopen(file_id, "x", H5P_DEFAULT);
    if (r_dset_id1 <= 0) {
        fprintf(stderr, "Error with H5Oopen!\n");
    }

    ierr = H5Dread(r_dset_id1, H5T_NATIVE_FLOAT, memspace, filespace, fapl_id, xo);
    if (ierr < 0)
        printf("read dset1 w/ H5Oopen failed\n");

    fprintf(stderr, "\nOopen x vals:\n");
    for (loop = 0; loop < numparticles; loop++) {
        fprintf(stderr, "%f ", x[loop]);
    }

    attr1 = H5Aopen(r_dset_id1, "Integer attribute", H5P_DEFAULT);

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

    /* test combinations of object open */
    if (H5Dclose(r_dset_id8) < 0)
        printf("H5Dclose dataset error\n");
    r_dset_id8 = H5Oopen(file_id, "id2", H5P_DEFAULT);
    if (r_dset_id1 <= 0) {
        fprintf(stderr, "Error with H5Oopen!\n");
    }
    ierr = H5Dread(r_dset_id8, H5T_NATIVE_INT, memspace, filespace, fapl_id, id2);
    if (ierr < 0)
        printf("read dset8 failed\n");
    if ((group_id = H5Oopen(file_id, "group1", H5P_DEFAULT)) < 0) {
        fprintf(stderr, "Error with H5Oopen!\n");
    }
    if (H5Oclose(group_id) < 0)
        printf("H5Oclose group error\n");

    if (H5Oclose(r_dset_id1) < 0)
        printf("H5Dclose dataset error\n");
    if (H5Dclose(r_dset_id2) < 0)
        printf("H5Dclose dataset error\n");
    if (H5Dclose(r_dset_id3) < 0)
        printf("H5Dclose dataset error\n");
    if (H5Dclose(r_dset_id4) < 0)
        printf("H5Dclose dataset error\n");
    if (H5Dclose(r_dset_id5) < 0)
        printf("H5Dclose dataset error\n");
    if (H5Dclose(r_dset_id6) < 0)
        printf("H5Dclose dataset error\n");
    if (H5Dclose(r_dset_id7) < 0)
        printf("H5Dclose dataset error\n");
    if (H5Oclose(r_dset_id8) < 0)
        printf("H5Dclose dataset error\n");

    if (H5Sclose(memspace) < 0)
        printf("H5Sclose memspace error\n");
    if (H5Sclose(filespace) < 0)
        printf("H5Sclose filespace error\n");
    if (H5Fclose(file_id) < 0)
        printf("H5Fclose error\n");
    if (H5Pclose(fapl_id) < 0)
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

    (void)MPI_Finalize();
    return 0;
}
