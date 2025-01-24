#include <math.h>
#include <hdf5.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#define DTYPE float

// A simple timer based on gettimeofday
extern struct timeval start_time[3];
extern float          elapse[3];
#define timer_on(id) gettimeofday(&start_time[id], NULL)
#define timer_off(id)                                                                                        \
    {                                                                                                        \
        struct timeval result, now;                                                                          \
        gettimeofday(&now, NULL);                                                                            \
        timeval_subtract(&result, &now, &start_time[id]);                                                    \
        elapse[id] += result.tv_sec + (DTYPE)(result.tv_usec) / 1000000.;                                    \
    }

#define timer_msg(id, msg) printf("%f seconds elapsed in %s\n", (DTYPE)(elapse[id]), msg);

#define timer_reset(id) elapse[id] = 0

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

int
timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y)
{
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
       tv_usec is certainly positive. */
    result->tv_sec  = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}

struct timeval start_time[3];
float          elapse[3];

// HDF5 specific declerations
herr_t ierr;

// Variables and dimensions
long      numparticles = 8388608; // 8  meg particles per process
long long total_particles, offset;

float *x, *y, *z;
float *px, *py, *pz;
int *  id1, *id2;
int    x_dim = 64;
int    y_dim = 64;
int    z_dim = 64;

// Uniform random number
inline double
uniform_random_number()
{
    return (((double)rand()) / ((double)(RAND_MAX)));
}

void
print_data(int n)
{
    int i;
    for (i = 0; i < n; i++)
        printf("%f %f %f %d %d %f %f %f\n", x[i], y[i], z[i], id1[i], id2[i], px[i], py[i], pz[i]);
}

// Create HDF5 file and read data
void
read_h5_data(int rank, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t dxpl)
{
    dset_ids[0] = H5Dopen(loc, "x", H5P_DEFAULT);
    dset_ids[1] = H5Dopen(loc, "y", H5P_DEFAULT);
    dset_ids[2] = H5Dopen(loc, "z", H5P_DEFAULT);
    dset_ids[3] = H5Dopen(loc, "id1", H5P_DEFAULT);
    dset_ids[4] = H5Dopen(loc, "id2", H5P_DEFAULT);
    dset_ids[5] = H5Dopen(loc, "px", H5P_DEFAULT);
    dset_ids[6] = H5Dopen(loc, "py", H5P_DEFAULT);
    dset_ids[7] = H5Dopen(loc, "pz", H5P_DEFAULT);

    ierr = H5Dread(dset_ids[0], H5T_NATIVE_FLOAT, memspace, filespace, dxpl, x);

    ierr = H5Dread(dset_ids[1], H5T_NATIVE_FLOAT, memspace, filespace, dxpl, y);

    ierr = H5Dread(dset_ids[2], H5T_NATIVE_FLOAT, memspace, filespace, dxpl, z);

    ierr = H5Dread(dset_ids[3], H5T_NATIVE_INT, memspace, filespace, dxpl, id1);

    ierr = H5Dread(dset_ids[4], H5T_NATIVE_INT, memspace, filespace, dxpl, id2);

    ierr = H5Dread(dset_ids[5], H5T_NATIVE_FLOAT, memspace, filespace, dxpl, px);

    ierr = H5Dread(dset_ids[6], H5T_NATIVE_FLOAT, memspace, filespace, dxpl, py);

    ierr = H5Dread(dset_ids[7], H5T_NATIVE_FLOAT, memspace, filespace, dxpl, pz);

    if (rank == 0)
        printf("  Read 8 variable completed\n");

    // print_data(3);
}

void
print_usage(char *name)
{
    printf("Usage: %s /path/to/file #mega_particles #timestep sleep_sec \n", name);
}

int
main(int argc, char *argv[])
{
    int   my_rank, num_procs, nts, i, j, sleep_time;
    hid_t file_id, *grp_ids, **dset_ids;
    hid_t fapl, dxpl, filespace, memspace;
    char  grp_name[128];
    char *file_name;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 3) {
        print_usage(argv[0]);
        return 0;
    }

    file_name = argv[1];

    if (argc >= 3)
        numparticles = (atoi(argv[2])) * 1024 * 1024;
    else
        numparticles = 8 * 1024 * 1024;

    nts = atoi(argv[3]);
    if (nts <= 0) {
        print_usage(argv[0]);
        return 0;
    }

    sleep_time = atoi(argv[4]);
    if (sleep_time < 0) {
        print_usage(argv[0]);
        return 0;
    }

    if (my_rank == 0) {
        fprintf(stderr, "Number of paritcles: %ld \n", numparticles);
        fprintf(stderr, "Number of steps: %d \n", nts);
        fprintf(stderr, "Sleep time: %d \n", sleep_time);
    }

    x = (float *)malloc(numparticles * sizeof(double));
    y = (float *)malloc(numparticles * sizeof(double));
    z = (float *)malloc(numparticles * sizeof(double));

    px = (float *)malloc(numparticles * sizeof(double));
    py = (float *)malloc(numparticles * sizeof(double));
    pz = (float *)malloc(numparticles * sizeof(double));

    id1 = (int *)malloc(numparticles * sizeof(int));
    id2 = (int *)malloc(numparticles * sizeof(int));

    MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    offset -= numparticles;

    /* Set up FAPL */
    fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);

    dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    MPI_Barrier(MPI_COMM_WORLD);
    timer_on(0);

    /* Open file */
    file_id = H5Fopen(file_name, H5F_ACC_RDONLY, fapl);
    if (file_id < 0) {
        printf("Error with opening file [%s]!\n", file_name);
        goto done;
    }

    if (my_rank == 0)
        printf("Opened HDF5 file ... [%s]\n", file_name);

    filespace = H5Screate_simple(1, (hsize_t *)&total_particles, NULL);
    memspace  = H5Screate_simple(1, (hsize_t *)&numparticles, NULL);

    // printf("total_particles: %lld\n", total_particles);
    // printf("my particles   : %ld\n", numparticles);

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, (hsize_t *)&offset, NULL, (hsize_t *)&numparticles, NULL);

    dset_ids = (hid_t **)calloc(nts, sizeof(hid_t *));
    grp_ids  = (hid_t *)calloc(nts, sizeof(hid_t));
    for (i = 0; i < nts; i++)
        dset_ids[i] = (hid_t *)calloc(8, sizeof(hid_t));

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < nts; i++) {

        sprintf(grp_name, "Timestep_%d", i);
        grp_ids[i] = H5Gopen(file_id, grp_name, H5P_DEFAULT);

        if (my_rank == 0)
            printf("Read %s ... \n", grp_name);

        timer_on(1);
        timer_reset(2);
        timer_on(2);

        read_h5_data(my_rank, grp_ids[i], dset_ids[i], filespace, memspace, dxpl);

        MPI_Barrier(MPI_COMM_WORLD);
        timer_off(1);
        timer_off(2);
        if (my_rank == 0)
            timer_msg(2, "read 1 step");

        for (j = 0; j < 8; j++)
            H5Dclose(dset_ids[i][j]);
        H5Gclose(grp_ids[i]);

        if (i != nts - 1) {
            if (my_rank == 0)
                printf("  sleep for %ds\n", sleep_time);
            sleep(sleep_time);
        }
    }

    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Pclose(fapl);
    H5Pclose(dxpl);
    H5Fclose(file_id);

    MPI_Barrier(MPI_COMM_WORLD);
    timer_off(0);
    if (my_rank == 0) {
        printf("\nTiming results\n");
        printf("Total sleep time %ds\n", sleep_time * (nts - 1));
        timer_msg(1, "only read");
        timer_msg(0, "total time");
        printf("\n");
    }

    free(x);
    free(y);
    free(z);
    free(px);
    free(py);
    free(pz);
    free(id1);
    free(id2);
    for (i = 0; i < nts; i++)
        free(dset_ids[i]);
    free(dset_ids);
    free(grp_ids);

done:
    MPI_Finalize();
    return 0;
}
