/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the files COPYING and Copyright.html.  COPYING can be found at the root   *
 * of the source code distribution tree; Copyright.html can be found at the  *
 * root level of an installed copy of the electronic HDF5 document set and   *
 * is linked from the top-level documents page.  It can also be found at     *
 * http://hdfgroup.org/HDF5/doc/Copyright.html.  If you do not have          *
 * access to either file, you may request a copy from help@hdfgroup.org.     *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Purpose: The PDC VOL plugin where access is forwarded to the PDC
 * library
 */
#include "H5VLpdc.h" /* PDC plugin                         */
#include "H5PLextern.h"
#include "H5VLerror.h"

/* External headers needed by this file */
#include "H5linkedlist.h"
#include "pdc.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/****************/
/* Local Macros */
/****************/

#define ADDR_MAX              256
#define H5VL_PDC_SEQ_LIST_LEN 128
/* (Uncomment to enable) */
#define ENABLE_LOGGING

/* Remove warnings when connector does not use callback arguments */
#if defined(__cplusplus)
#define H5VL_ATTR_UNUSED
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#define H5VL_ATTR_UNUSED __attribute__((unused))
#else
#define H5VL_ATTR_UNUSED
#endif

/************************************/
/* Local Type and Struct Definition */
/************************************/

/* The VOL wrapper context */
typedef struct H5VL_pdc_wrap_ctx_t {
    hid_t under_vol_id;   /* VOL ID for under VOL */
    void *under_wrap_ctx; /* Object wrapping context for under VOL */
} H5VL_pdc_wrap_ctx_t;

/* Common object information */
typedef struct H5VL_pdc_obj_t {
    hid_t                   under_vol_id;
    void                   *under_object;
    pdcid_t                 obj_id;
    struct H5VL_pdc_file_t *file;
    char                    obj_name[ADDR_MAX];
    char                   *group_name;
    char                   *attr_name;
    psize_t                 attr_value_size;
    pdc_var_type_t          type;
    pdcid_t                 reg_id_from;
    pdcid_t                 reg_id_to;
} H5VL_pdc_obj_t;

/* The file struct */
typedef struct H5VL_pdc_file_t {
    hid_t                   under_vol_id;
    void                   *under_object;
    pdcid_t                 obj_id;
    struct H5VL_pdc_file_t *file;
    char                   *file_name;
    MPI_Comm                comm;
    MPI_Info                info;
    int                     my_rank;
    int                     num_procs;
    pdcid_t                 cont_id;
    int                     nobj;
    H5_LIST_HEAD(H5VL_pdc_dset_t) ids;
} H5VL_pdc_file_t;

/* The dataset struct */
typedef struct H5VL_pdc_dset_t {
    hid_t          under_vol_id;
    void          *under_object;
    pdcid_t        obj_id;
    H5VL_pdc_obj_t obj; /* Must be first */
    hid_t          type_id;
    hid_t          space_id;
    hbool_t        mapped;
    H5_LIST_ENTRY(H5VL_pdc_dset_t) entry;
} H5VL_pdc_dset_t;

/* PDC-specific file access properties */
typedef struct H5VL_pdc_info_t {
    void *under_vol_info;
    hid_t under_vol_id;
} H5VL_pdc_info_t;

/********************/
/* Local Prototypes */
/********************/

/* "Management" callbacks */
static herr_t H5VL_pdc_init(hid_t vipl_id);
static herr_t H5VL_pdc_obj_term(void);

/* VOL info callbacks */
static void  *H5VL_pdc_info_copy(const void *_old_info);
static herr_t H5VL_pdc_info_cmp(int *cmp_value, const void *_info1, const void *_info2);
static herr_t H5VL_pdc_info_free(void *info);
static herr_t H5VL_pdc_info_to_str(const void *info, char **str);
static herr_t H5VL_pdc_str_to_info(const char *str, void **info);

/* File callbacks */
static void  *H5VL_pdc_file_create(const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id,
                                   hid_t dxpl_id, void **req);
static void  *H5VL_pdc_file_open(const char *name, unsigned flags, hid_t fapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_file_close(void *file, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_file_specific(void *file, H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req);

/* Dataset callbacks */
static void  *H5VL_pdc_dataset_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                      hid_t lcpl_id, hid_t type_id, hid_t space_id, hid_t dcpl_id,
                                      hid_t dapl_id, hid_t dxpl_id, void **req);
static void  *H5VL_pdc_dataset_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                    hid_t dapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_dataset_read(void *dset, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id,
                                    hid_t plist_id, void *buf, void **req);
static herr_t H5VL_pdc_dataset_write(void *dset, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id,
                                     hid_t plist_id, const void *buf, void **req);
static herr_t H5VL_pdc_dataset_get(void *dset, H5VL_dataset_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_dataset_close(void *dset, hid_t dxpl_id, void **req);

/* Attribute callbacks */
static void  *H5VL_pdc_attr_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                   hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t dxpl_id,
                                   void **req);
static void  *H5VL_pdc_attr_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                 hid_t aapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_read(void *attr, hid_t mem_type_id, void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_write(void *attr, hid_t mem_type_id, const void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_close(void *attr, hid_t dxpl_id, void **req);

/* Container/connector introspection callbacks */
static herr_t H5VL_pdc_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl,
                                               const H5VL_class_t **conn_cls);
static herr_t H5VL_pdc_introspect_get_cap_flags(const void *info, unsigned *cap_flags);
static herr_t H5VL_pdc_introspect_opt_query(void *obj, H5VL_subclass_t cls, int opt_type, uint64_t *flags);

/* Group callbacks */
static void  *H5VL_pdc_group_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                    hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id, hid_t dxpl_id, void **req);
static void  *H5VL_pdc_group_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                  hid_t gapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_group_get(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_group_close(void *grp, hid_t dxpl_id, void **req);

/*******************/
/* Local variables */
/*******************/

/* The PDC VOL plugin struct */
static const H5VL_class_t H5VL_pdc_g = {
    H5VL_VERSION,                       /* version */
    (H5VL_class_value_t)H5VL_PDC_VALUE, /* value */
    H5VL_PDC_NAME_STRING,               /* name */
    0,                                  /* connector version TEMPORARY FIX*/
    0,                                  /* capability flags */
    H5VL_pdc_init,                      /* initialize */
    H5VL_pdc_obj_term,                  /* terminate */
    {
        /* info_cls */
        sizeof(H5VL_pdc_info_t), /* info size */
        H5VL_pdc_info_copy,      /* info copy */
        H5VL_pdc_info_cmp,       /* info compare ADD*/
        H5VL_pdc_info_free,      /* info free */
        H5VL_pdc_info_to_str,    /* to_str ADD*/
        H5VL_pdc_str_to_info,    /* from_str ADD*/
    },
    {
        /* wrap_cls */
        NULL, /* get_object */
        NULL, /* get_wrap_ctx */
        NULL, /* wrap_object */
        NULL, /* unwrap_object */
        NULL, /* free_wrap_ctx */
    },
    {
        /* attribute_cls ADD*/
        H5VL_pdc_attr_create, /* create ADD*/
        H5VL_pdc_attr_open,   /* open ADD*/
        H5VL_pdc_attr_read,   /* read ADD*/
        H5VL_pdc_attr_write,  /* write ADD*/
        H5VL_pdc_attr_get,    /* get ADD*/
        NULL,                 /* specific */
        NULL,                 /* optional */
        H5VL_pdc_attr_close   /* close ADD*/
    },
    {
        /* dataset_cls  */
        H5VL_pdc_dataset_create, /* create */
        H5VL_pdc_dataset_open,   /* open */
        H5VL_pdc_dataset_read,   /* read */
        H5VL_pdc_dataset_write,  /* write */
        H5VL_pdc_dataset_get,    /* get */
        NULL,                    /* specific */
        NULL,                    /* optional */
        H5VL_pdc_dataset_close   /* close */
    },
    {
        /* datatype_cls */
        NULL, /* commit */
        NULL, /* open */
        NULL, /* get_size */
        NULL, /* specific */
        NULL, /* optional */
        NULL  /* close */
    },
    {
        /* file_cls     */
        H5VL_pdc_file_create,   /* create */
        H5VL_pdc_file_open,     /* open */
        NULL,                   /* get */
        H5VL_pdc_file_specific, /* specific */
        NULL,                   /* optional */
        H5VL_pdc_file_close     /* close */
    },
    {
        /* group_cls    */
        H5VL_pdc_group_create, /* create */
        H5VL_pdc_group_open,   /* open */
        H5VL_pdc_group_get,    /* get */
        NULL,                  /* specific */
        NULL,                  /* optional */
        H5VL_pdc_group_close   /* close */
    },
    {
        /* link_cls     */
        NULL, /* create */
        NULL, /* copy */
        NULL, /* move */
        NULL, /* get */
        NULL, /* specific */
        NULL  /* optional */
    },
    {
        /* object_cls   */
        NULL, /* open */
        NULL, /* copy */
        NULL, /* get */
        NULL, /* specific */
        NULL  /* optional */
    },
    {
        /* introspect_cls */
        H5VL_pdc_introspect_get_conn_cls,  /* get_conn_cls */
        H5VL_pdc_introspect_get_cap_flags, /* get_cap_flags */
        H5VL_pdc_introspect_opt_query      /* opt_query */
    },
    {
        /* request_cls  */
        NULL, /* wait */
        NULL, /* notify */
        NULL, /* cancel */
        NULL, /* specific */
        NULL, /* optional */
        NULL  /* free */
    },
    {
        /* blob_cls */
        NULL, /* put */
        NULL, /* get */
        NULL, /* specific */
        NULL  /* optional */
    },
    {
        /* token_cls */
        NULL, /* cmp */
        NULL, /* to_str */
        NULL  /* from_str */
    },
    NULL /* optional    */
};

/* The connector identification number, initialized at runtime */
static hid_t   H5VL_PDC_g      = H5I_INVALID_HID;
static hbool_t H5VL_pdc_init_g = FALSE;

/* Error stack declarations */
hid_t H5VL_ERR_STACK_g = H5I_INVALID_HID;
hid_t H5VL_ERR_CLS_g   = H5I_INVALID_HID;

static pdcid_t pdc_id = 0;

/*---------------------------------------------------------------------------*/

/**
 * Public definitions.
 */

/*---------------------------------------------------------------------------*/
void
replace_multi_slash(char *str)
{
    char *dest = str;
    while (*str != '\0') {
        while (*str == '/' && *(str + 1) == '/')
            str++;
        *dest++ = *str++;
    }
    *dest = '\0';
}

/*---------------------------------------------------------------------------*/
H5PL_type_t
H5PLget_plugin_type(void)
{
    return H5PL_TYPE_VOL;
}

/*---------------------------------------------------------------------------*/
const void *
H5PLget_plugin_info(void)
{
    return &H5VL_pdc_g;
}

/*---------------------------------------------------------------------------*/
herr_t
H5VLpdc_term(void)
{
    FUNC_ENTER_VOL(herr_t, SUCCEED)

    /* Terminate the plugin */
    if (H5VL_pdc_obj_term() < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CLOSEERROR, FAIL, "can't terminate PDC VOL connector");

done:
    FUNC_LEAVE_VOL
}

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_init(hid_t H5VL_ATTR_UNUSED vipl_id)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered pdc_init\n");
    fflush(stdout);
#endif
    FUNC_ENTER_VOL(herr_t, SUCCEED)

    /* Check whether initialized */
    if (H5VL_pdc_init_g)
        HGOTO_ERROR(H5E_VOL, H5E_CANTINIT, FAIL, "attempting to initialize connector twice");

    /* Create error stack */
    if ((H5VL_ERR_STACK_g = H5Ecreate_stack()) < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTCREATE, FAIL, "can't create error stack");

    /* Register error class for error reporting */
    if ((H5VL_ERR_CLS_g =
             H5Eregister_class(H5VL_PDC_PACKAGE_NAME, H5VL_PDC_LIBRARY_NAME, H5VL_PDC_VERSION_STRING)) < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTREGISTER, FAIL, "can't register error class");

    /* Init PDC */
    pdc_id = PDCinit("pdc");
    if (pdc_id <= 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTINIT, FAIL, "could not initialize PDC");

    /* Initialized */
    H5VL_pdc_init_g = TRUE;

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_init() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_obj_term(void)
{
    FUNC_ENTER_VOL(herr_t, SUCCEED)

    if (!H5VL_pdc_init_g)
        HGOTO_DONE(SUCCEED);

    /* "Forget" plugin id.  This should normally be called by the library
     * when it is closing the id, so no need to close it here. */
    H5VL_PDC_g = H5I_INVALID_HID;

    H5VL_pdc_init_g = FALSE;

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_obj_term() */

/*---------------------------------------------------------------------------*/
static H5VL_pdc_obj_t *
H5VL_pdc_new_obj(void *under_obj, hid_t under_vol_id)
{
    H5VL_pdc_obj_t *new_obj;

    new_obj               = (H5VL_pdc_obj_t *)calloc(1, sizeof(H5VL_pdc_obj_t));
    new_obj->under_object = under_obj;
    new_obj->under_vol_id = under_vol_id;

    return new_obj;
} /* end H5VL__pdc_new_obj() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_free_obj(H5VL_pdc_obj_t *obj)
{
    hid_t err_id;

    err_id = H5Eget_current_stack();
    H5Eset_current_stack(err_id);
    free(obj);

    return 0;
} /* end H5VL__pdc_free_obj() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_info_copy(const void *_old_info)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered info_copy\n");
    fflush(stdout);
#endif
    const H5VL_pdc_info_t *old_info = (const H5VL_pdc_info_t *)_old_info;
    H5VL_pdc_info_t       *new_info = NULL;

    FUNC_ENTER_VOL(void *, NULL)

    if (NULL == (new_info = (H5VL_pdc_info_t *)malloc(sizeof(H5VL_pdc_info_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");

    FUNC_RETURN_SET(new_info);

done:
    if (FUNC_ERRORED) {
        /* cleanup */
        if (new_info && H5VL_pdc_info_free(new_info) < 0)
            HDONE_ERROR(H5E_PLIST, H5E_CANTFREE, NULL, "can't free fapl");
    }

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_info_copy() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_info_cmp(int *cmp_value, const void *_info1, const void *_info2)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered info_cmp\n");
    fflush(stdout);
#endif

    const H5VL_pdc_info_t *info1 = (const H5VL_pdc_info_t *)_info1;
    const H5VL_pdc_info_t *info2 = (const H5VL_pdc_info_t *)_info2;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(info1);
    assert(info2);

    *cmp_value = memcmp(info1, info2, sizeof(H5VL_pdc_info_t));

    FUNC_LEAVE_VOL
}

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_info_free(void *_info)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered info_free\n");
    fflush(stdout);
#endif

    H5VL_pdc_info_t *info = (H5VL_pdc_info_t *)_info;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(info);

    /* free the struct */
    free(info);

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_info_free() */

static herr_t
H5VL_pdc_info_to_str(const void *_info, char **str)
{
    const H5VL_pdc_info_t *info              = (const H5VL_pdc_info_t *)_info;
    H5VL_class_value_t     under_value       = (H5VL_class_value_t)-1;
    char                  *under_vol_string  = NULL;
    size_t                 under_vol_str_len = 0;

    /* Get value and string for underlying VOL connector */
    H5VLget_value(info->under_vol_id, &under_value);
    H5VLconnector_info_to_str(info->under_vol_info, info->under_vol_id, &under_vol_string);

    /* Determine length of underlying VOL info string */
    if (under_vol_string)
        under_vol_str_len = strlen(under_vol_string);

    /* Allocate space for our info */
    *str = (char *)H5allocate_memory(32 + under_vol_str_len, (hbool_t)0);
    assert(*str);

    /* Encode our info
     * Normally we'd use snprintf() here for a little extra safety, but that
     * call had problems on Windows until recently. So, to be as platform-independent
     * as we can, we're using sprintf() instead.
     */
    sprintf(*str, "under_vol=%u;under_info={%s}", (unsigned)under_value,
            (under_vol_string ? under_vol_string : ""));

    return 0;
} /* end H5VL_pdc_obj_to_str() */

hid_t
H5VL_pdc_register(void)
{
    /* Singleton register the pass-through VOL connector ID */
    if (H5VL_PDC_g < 0)
        H5VL_PDC_g = H5VLregister_connector(&H5VL_pdc_g, H5P_DEFAULT);

    return H5VL_PDC_g;
} /* end H5VL_pdc_register() */

static herr_t
H5VL_pdc_str_to_info(const char *str, void **_info)
{
    H5VL_pdc_info_t *info;
    unsigned         under_vol_value;
    const char      *under_vol_info_start, *under_vol_info_end;
    hid_t            under_vol_id;
    void            *under_vol_info = NULL;

    /* Retrieve the underlying VOL connector value and info */
    sscanf(str, "under_vol=%u;", &under_vol_value);
    under_vol_id         = H5VLregister_connector_by_value((H5VL_class_value_t)under_vol_value, H5P_DEFAULT);
    under_vol_info_start = strchr(str, '{');
    under_vol_info_end   = strrchr(str, '}');
    assert(under_vol_info_end > under_vol_info_start);
    if (under_vol_info_end != (under_vol_info_start + 1)) {
        char *under_vol_info_str;

        under_vol_info_str = (char *)malloc((size_t)(under_vol_info_end - under_vol_info_start));
        memcpy(under_vol_info_str, under_vol_info_start + 1,
               (size_t)((under_vol_info_end - under_vol_info_start) - 1));
        *(under_vol_info_str + (under_vol_info_end - under_vol_info_start)) = '\0';

        H5VLconnector_str_to_info(under_vol_info_str, under_vol_id, &under_vol_info);

        free(under_vol_info_str);
    } /* end else */

    /* Allocate new pass-through VOL connector info and set its fields */
    info                 = (H5VL_pdc_info_t *)calloc(1, sizeof(H5VL_pdc_info_t));
    info->under_vol_id   = under_vol_id;
    info->under_vol_info = under_vol_info;

    /* Set return value */
    *_info = info;

    return 0;
} /* end H5VL_pdc_str_to_info() */

/*---------------------------------------------------------------------------*/
static H5VL_pdc_file_t *
H5VL__pdc_file_init(const char *name, unsigned flags, H5VL_pdc_info_t *info, hid_t fapl_id)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered file_init\n");
    fflush(stdout);
#endif

    H5VL_pdc_file_t *file = NULL;

    FUNC_ENTER_VOL(void *, NULL)

    /* allocate the file object that is returned to the user */
    if (NULL == (file = malloc(sizeof(H5VL_pdc_file_t))))
        HGOTO_ERROR(H5E_FILE, H5E_CANTALLOC, NULL, "can't allocate PDC file struct");
    memset(file, 0, sizeof(H5VL_pdc_file_t));
    file->info = MPI_INFO_NULL;
    file->comm = MPI_COMM_NULL;
    file->file = file;

    /* Fill in fields of file we know */
    file->under_object = file;
    file->under_vol_id = H5VL_PDC_g;

    if (NULL == (file->file_name = strdup(name)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "can't copy file name");

    /* Duplicate communicator and Info object. */
    H5Pget_fapl_mpio(fapl_id, &file->comm, &file->info);

    /* Obtain the process rank and size from the communicator attached to the
     * fapl ID */
    //
    MPI_Comm_rank(file->comm, &file->my_rank);
    MPI_Comm_size(file->comm, &file->num_procs);

    file->nobj = 0;

    H5_LIST_INIT(&file->ids);

    FUNC_RETURN_SET((void *)file);

done:
    FUNC_LEAVE_VOL
} /* end H5VL__pdc_file_init() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL__pdc_file_close(H5VL_pdc_file_t *file)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered file_close\n");
    fflush(stdout);
#endif

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(file);

    /* Free file data structures */
    if (file->file_name)
        free(file->file_name);
    if (file->comm)
        MPI_Comm_free(&file->comm);
    file->comm = MPI_COMM_NULL;

    free(file);
    file = NULL;

    FUNC_LEAVE_VOL
} /* end H5VL__pdc_file_close() */

/*---------------------------------------------------------------------------*/
static H5VL_pdc_dset_t *
H5VL__pdc_dset_init(H5VL_pdc_obj_t *item)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered dset_init\n");
    fflush(stdout);
#endif

    H5VL_pdc_dset_t *dset = NULL;

    FUNC_ENTER_VOL(void *, NULL)

    /* Allocate the dataset object that is returned to the user */
    if (NULL == (dset = malloc(sizeof(H5VL_pdc_dset_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "can't allocate PDC dataset struct");
    memset(dset, 0, sizeof(H5VL_pdc_dset_t));

    dset->obj.reg_id_from = 0;
    dset->obj.reg_id_to   = 0;
    dset->mapped          = 0;
    dset->type_id         = FAIL;
    dset->space_id        = FAIL;

    /* Set return value */
    FUNC_RETURN_SET((void *)dset);

done:
    FUNC_LEAVE_VOL
} /* end H5VL__pdc_dset_init() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL__pdc_dset_free(H5VL_pdc_dset_t *dset)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered dset_free\n");
    fflush(stdout);
#endif

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    if (dset->space_id != FAIL && H5Idec_ref(dset->space_id) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTDEC, FAIL, "failed to close space");

    H5_LIST_REMOVE(dset, entry);
    free(dset);
    dset = NULL;

    FUNC_LEAVE_VOL
} /* end H5VL__pdc_dset_free() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL__pdc_sel_to_recx_iov(hid_t space_id, size_t type_size, uint64_t *off)
{
    hid_t   sel_iter_id;           /* Selection iteration info */
    hbool_t sel_iter_init = FALSE; /* Selection iteration info has been initialized */
    size_t  nseq;
    size_t  nelem;
    size_t  len[H5VL_PDC_SEQ_LIST_LEN];

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    /* Initialize selection iterator  */
    if ((sel_iter_id = H5Ssel_iter_create(space_id, type_size, 0)) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL, "unable to initialize selection iterator");
    sel_iter_init = TRUE; /* Selection iteration info has been initialized */

    /* Generate sequences from the file space until finished */
    do {
        /* Get the sequences of bytes */
        if (H5Ssel_iter_get_seq_list(sel_iter_id, (size_t)H5VL_PDC_SEQ_LIST_LEN, (size_t)-1, &nseq, &nelem,
                                     off, len) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "sequence length generation failed");
    } while (nseq == H5VL_PDC_SEQ_LIST_LEN);

done:
    /* Release selection iterator */
    if (sel_iter_init && H5Ssel_iter_close(sel_iter_id) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to release selection iterator");
    FUNC_LEAVE_VOL
} /* end H5VL__pdc_sel_to_recx_iov() */

/*---------------------------------------------------------------------------*/
void *
H5VL_pdc_file_create(const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id, hid_t dxpl_id,
                     void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered file_create\n");
    fflush(stdout);
#endif

    H5VL_pdc_info_t *info;
    H5VL_pdc_file_t *file = NULL;
    pdcid_t          cont_prop;

    FUNC_ENTER_VOL(void *, NULL)

    assert(name);
    hid_t under_vol_id;
    H5Pget_vol_id(fapl_id, &under_vol_id);

    /* Get information from the FAPL */
    if (H5Pget_vol_info(fapl_id, (void **)&info) < 0)
        HGOTO_ERROR(H5E_SYM, H5E_CANTGET, NULL, "can't get PDC info struct");

    /* Initialize file */
    if (NULL == (file = H5VL__pdc_file_init(name, flags, info, fapl_id)))
        HGOTO_ERROR(H5E_FILE, H5E_CANTINIT, NULL, "can't init PDC file struct");

    if ((cont_prop = PDCprop_create(PDC_CONT_CREATE, pdc_id)) <= 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCREATE, NULL, "can't create container property");
    if ((file->cont_id = PDCcont_create_col(name, cont_prop)) <= 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCREATE, NULL, "can't create container");

    if ((PDCprop_close(cont_prop)) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCREATE, NULL, "can't close container property");

    /* Free info */
    if (info && H5VL_pdc_info_free(info) < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTFREE, NULL, "can't free connector info");

    FUNC_RETURN_SET((void *)file);

done:
    if (FUNC_ERRORED) {
        /* Free info */
        if (info)
            if (H5VL_pdc_info_free(info) < 0)
                HDONE_ERROR(H5E_VOL, H5E_CANTFREE, NULL, "can't free connector info");

        /* Close file */
        if (file < 0)
            HDONE_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, NULL, "can't close file");
    } /* end if */

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_file_create() */

/*---------------------------------------------------------------------------*/
void *
H5VL_pdc_file_open(const char *name, unsigned flags, hid_t fapl_id, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered file_open\n");
    fflush(stdout);
#endif

    H5VL_pdc_info_t *info;
    H5VL_pdc_file_t *file = NULL;

    FUNC_ENTER_VOL(void *, NULL)

    /* Get information from the FAPL */
    if (H5Pget_vol_info(fapl_id, (void **)&info) < 0)
        HGOTO_ERROR(H5E_SYM, H5E_CANTGET, NULL, "can't get PDC info struct");

    /* Initialize file */
    if (NULL == (file = H5VL__pdc_file_init(name, flags, info, fapl_id)))
        HGOTO_ERROR(H5E_FILE, H5E_CANTINIT, NULL, "can't init PDC file struct");

    if ((file->cont_id = PDCcont_open(name, pdc_id)) <= 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTOPENFILE, NULL, "failed to create container");

    /* Free info */
    if (info && H5VL_pdc_info_free(info) < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTFREE, NULL, "can't free connector info");

    FUNC_RETURN_SET((void *)file);

done:
    if (FUNC_ERRORED) {
        /* Free info */
        if (info)
            if (H5VL_pdc_info_free(info) < 0)
                HGOTO_ERROR(H5E_VOL, H5E_CANTFREE, NULL, "can't free connector info");
        if (file < 0)
            HDONE_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, NULL, "can't close file");
    } /* end if */

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_file_open() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_file_close(void *_file, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered file_close\n");
    fflush(stdout);
#endif
    H5VL_pdc_file_t *file = (H5VL_pdc_file_t *)_file;
    H5VL_pdc_dset_t *dset = NULL;
    perr_t           ret;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(file);

    while (!H5_LIST_IS_EMPTY(&file->ids)) {
        H5_LIST_GET_FIRST(dset, &file->ids);
        H5_LIST_REMOVE(dset, entry);
        if (H5VL__pdc_dset_free(dset) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "failed to free dataset");
    }

    if ((ret = PDCcont_close(file->cont_id)) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, FAIL, "failed to close container");

    if ((ret = PDCclose(pdc_id)) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, FAIL, "failed to close PDC");

    /* Close the file */
    if (H5VL__pdc_file_close(file) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, FAIL, "failed to close file");

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_file_close() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_file_specific(void *file, H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered file_specific\n");
    fflush(stdout);
#endif
    H5VL_pdc_obj_t            *o = (H5VL_pdc_obj_t *)file;
    H5VL_pdc_obj_t            *new_o;
    H5VL_file_specific_args_t  my_args;
    H5VL_file_specific_args_t *new_args;
    H5VL_pdc_info_t           *info;
    hid_t                      under_vol_id = -1;
    herr_t                     ret_value;

    if (args->op_type == H5VL_FILE_IS_ACCESSIBLE) {

        /* Shallow copy the args */

        memcpy(&my_args, args, sizeof(my_args));

        /* Get copy of our VOL info from FAPL */
        H5Pget_vol_info(args->args.is_accessible.fapl_id, (void **)&info);
        /* Make sure we have info about the underlying VOL to be used */
        if (!info)
            return (-1);

        /* Keep the correct underlying VOL ID for later */
        under_vol_id = info->under_vol_id;

        /* Copy the FAPL */
        my_args.args.is_accessible.fapl_id = H5Pcopy(args->args.is_accessible.fapl_id);

        /* Set the VOL ID and info for the underlying FAPL */
        H5Pset_vol(my_args.args.is_accessible.fapl_id, info->under_vol_id, info->under_vol_info);

        /* Set argument pointer to new arguments */
        new_args = &my_args;

        /* Set object pointer for operation */
        new_o = NULL;
    } /* end else-if */
    else if (args->op_type == H5VL_FILE_DELETE) {
        /* Shallow copy the args */
        memcpy(&my_args, args, sizeof(my_args));

        /* Get copy of our VOL info from FAPL */
        H5Pget_vol_info(args->args.del.fapl_id, (void **)&info);
        /* Make sure we have info about the underlying VOL to be used */
        if (!info)
            return (-1);

        /* Keep the correct underlying VOL ID for later */
        under_vol_id = info->under_vol_id;

        /* Copy the FAPL */
        my_args.args.del.fapl_id = H5Pcopy(args->args.del.fapl_id);

        /* Set the VOL ID and info for the underlying FAPL */
        H5Pset_vol(my_args.args.del.fapl_id, info->under_vol_id, info->under_vol_info);

        /* Set argument pointer to new arguments */
        new_args = &my_args;

        /* Set object pointer for operation */
        new_o = NULL;
    } /* end else-if */
    else {
        /* Keep the correct underlying VOL ID for later */
        under_vol_id = o->under_vol_id;

        /* Set argument pointer to current arguments */
        new_args = args;

        /* Set object pointer for operation */
        new_o = o->under_object;
    } /* end else */
    if (args->op_type != H5VL_FILE_FLUSH) {
        ret_value = H5VLfile_specific(new_o, under_vol_id, new_args, dxpl_id, req);
    }

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, under_vol_id);

    if (args->op_type == H5VL_FILE_IS_ACCESSIBLE) {
        /* Close underlying FAPL */
        H5Pclose(my_args.args.is_accessible.fapl_id);

        /* Release copy of our VOL info */
        H5VL_pdc_info_free(info);
    } /* end else-if */
    else if (args->op_type == H5VL_FILE_DELETE) {
        /* Close underlying FAPL */
        H5Pclose(my_args.args.del.fapl_id);

        /* Release copy of our VOL info */
        H5VL_pdc_info_free(info);
    } /* end else-if */
    else if (args->op_type == H5VL_FILE_REOPEN) {
        /* Wrap file struct pointer for 'reopen' operation, if we reopened one */
        if (ret_value >= 0 && *args->args.reopen.file)
            *args->args.reopen.file = H5VL_pdc_new_obj(*args->args.reopen.file, under_vol_id);
    } /* end else */

    return ret_value;
} /* end H5VL_pdc_file_specific() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_dataset_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t lcpl_id,
                        hid_t type_id, hid_t space_id, hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id,
                        void **req)
{

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    int             buff_len;
    if (o->group_name) {
        buff_len = strlen(name) + strlen(o->file->file_name) + strlen(o->group_name) + 3;
    }
    else {
        buff_len = strlen(name) + strlen(o->file->file_name) + 2;
    }
    char new_name[buff_len];
    if (o->group_name) {
        strcpy(new_name, name);
        strcat(new_name, "/");
        strcat(new_name, o->group_name);
        strcat(new_name, "/");
        strcat(new_name, o->file->file_name);
    }
    else {
        strcpy(new_name, name);
        strcat(new_name, "/");
        strcat(new_name, o->file->file_name);
    }
    H5VL_pdc_dset_t *dset = NULL;
    pdcid_t          obj_prop, obj_id;
    int              ndim;
    hsize_t          dims[H5S_MAX_RANK];

    FUNC_ENTER_VOL(void *, NULL)

    if (!obj)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "parent object is NULL");
    if (!loc_params)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "location parameters object is NULL");
    if (!name)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "dataset name is NULL");

    /* Init dataset */
    if (NULL == (dset = H5VL__pdc_dset_init(o)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, NULL, "can't init PDC dataset struct");

    /* Finish setting up dataset struct */
    if ((dset->type_id = H5Tcopy(type_id)) < 0)
        HGOTO_ERROR(H5E_SYM, H5E_CANTCOPY, NULL, "failed to copy datatype");
    if ((dset->space_id = H5Scopy(space_id)) < 0)
        HGOTO_ERROR(H5E_SYM, H5E_CANTCOPY, NULL, "failed to copy dataspace");
    if (H5Sselect_all(dset->space_id) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTDELETE, NULL, "can't change selection");

    obj_prop = PDCprop_create(PDC_OBJ_CREATE, pdc_id);
    if (H5Tequal(H5T_NATIVE_INT, type_id) == TRUE) {
        PDCprop_set_obj_type(obj_prop, PDC_INT);
        dset->obj.type = PDC_INT;
    }
    else if (H5Tequal(H5T_NATIVE_FLOAT, type_id) == TRUE) {
        PDCprop_set_obj_type(obj_prop, PDC_FLOAT);
        dset->obj.type = PDC_FLOAT;
    }
    else if (H5Tequal(H5T_NATIVE_DOUBLE, type_id) == TRUE) {
        PDCprop_set_obj_type(obj_prop, PDC_DOUBLE);
        dset->obj.type = PDC_DOUBLE;
    }
    else if (H5Tequal(H5T_NATIVE_CHAR, type_id) == TRUE) {
        PDCprop_set_obj_type(obj_prop, PDC_CHAR);
        dset->obj.type = PDC_CHAR;
    }
    else {
        dset->obj.type = PDC_UNKNOWN;
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, NULL, "datatype is not supported yet");
    }
    /* Get dataspace extent */
    if ((ndim = H5Sget_simple_extent_ndims(space_id)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, NULL, "can't get number of dimensions");
    if (ndim != H5Sget_simple_extent_dims(space_id, dims, NULL))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, NULL, "can't get dimensions");
    PDCprop_set_obj_dims(obj_prop, ndim, dims);
    /* Create PDC object */
    obj_id       = PDCobj_create_mpi(o->file->cont_id, new_name, obj_prop, 0, o->file->comm);
    dset->obj_id = obj_id;
    strcpy(dset->obj.obj_name, name);
    o->file->nobj++;
    H5_LIST_INSERT_HEAD(&o->file->ids, dset, entry);

    if ((PDCprop_close(obj_prop)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, NULL, "can't close object property");

    /* Set return value */
    FUNC_RETURN_SET((void *)dset);

done:
    FUNC_LEAVE_VOL
}

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_dataset_open(void *obj, const H5VL_loc_params_t *loc_params, const char *_name, hid_t dapl_id,
                      hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered dataset_open\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t      *o    = (H5VL_pdc_obj_t *)obj;
    H5VL_pdc_dset_t     *dset = NULL;
    struct pdc_obj_info *obj_info;

    int buff_len;
    if (o->group_name) {
        buff_len = strlen(_name) + strlen(o->file->file_name) + strlen(o->group_name) + 3;
    }
    else {
        buff_len = strlen(_name) + strlen(o->file->file_name) + 2;
    }

    char name[buff_len];
    if (o->group_name) {
        strcpy(name, _name);
        strcat(name, "/");
        strcat(name, o->group_name);
        strcat(name, "/");
        strcat(name, o->file->file_name);
        /* Assume that the name, group name, and file_name do not include multiple consecutive
           slashes as a part of their names. */
        replace_multi_slash(name);
    }
    else {
        strcpy(name, _name);
        strcat(name, "/");
        strcat(name, o->file->file_name);
    }

    FUNC_ENTER_VOL(void *, NULL)

    if (!obj)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "parent object is NULL");
    if (!loc_params)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "location parameters object is NULL");
    if (!name)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "dataset name is NULL");

    /* Init dataset */
    if (NULL == (dset = H5VL__pdc_dset_init(o)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, NULL, "can't init PDC dataset struct");

    strcpy(dset->obj.obj_name, name);
    dset->obj_id    = PDCobj_open(name, pdc_id);
    pdcid_t id_name = (pdcid_t)name;
    obj_info        = PDCobj_get_info(dset->obj_id);
    dset->space_id  = H5Screate_simple(obj_info->obj_pt->ndim, obj_info->obj_pt->dims, NULL);
    dset->obj.type  = obj_info->obj_pt->type;
    o->file->nobj++;
    H5_LIST_INSERT_HEAD(&o->file->ids, dset, entry);

    /* Set return value */
    FUNC_RETURN_SET((void *)dset);

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_open() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_dataset_write(void *_dset, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id,
                       hid_t plist_id, const void *buf, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered dataset_write\n");
    fflush(stdout);
#endif

    H5VL_pdc_dset_t *dset = (H5VL_pdc_dset_t *)_dset;
    uint64_t        *offset;
    size_t           type_size;
    int              ndim;
    pdcid_t          region_x, region_xx;
    hsize_t          dims[H5S_MAX_RANK];
    perr_t           ret;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    if (file_space_id == H5S_ALL)
        file_space_id = dset->space_id;
    if (mem_space_id == H5S_ALL)
        mem_space_id = file_space_id;

    /* Get memory dataspace object */
    if ((ndim = H5Sget_simple_extent_ndims(mem_space_id)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get number of dimensions");
    if (ndim != H5Sget_simple_extent_dims(mem_space_id, dims, NULL))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get dimensions");

    offset = (uint64_t *)malloc(sizeof(uint64_t) * ndim);
    if (ndim == 1)
        offset[0] = 0;
    else if (ndim == 2) {
        offset[1] = 0;
        offset[0] = 0;
    }
    else if (ndim == 3) {
        offset[2] = 0;
        offset[1] = 0;
        offset[0] = 0;
    }
    else
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "data dimension not supported");
    region_x              = PDCregion_create(ndim, offset, dims);
    dset->obj.reg_id_from = region_x;

    type_size = H5Tget_size(mem_type_id);
    H5VL__pdc_sel_to_recx_iov(file_space_id, type_size, offset);

    region_xx           = PDCregion_create(ndim, offset, dims);
    dset->obj.reg_id_to = region_xx;
    free(offset);

    dset->mapped = 1;

    if (PDC_INT == dset->obj.type)
        PDCbuf_obj_map((void *)buf, PDC_INT, region_x, dset->obj_id, region_xx);
    else if (PDC_FLOAT == dset->obj.type)
        PDCbuf_obj_map((void *)buf, PDC_FLOAT, region_x, dset->obj_id, region_xx);
    else if (PDC_DOUBLE == dset->obj.type)
        PDCbuf_obj_map((void *)buf, PDC_DOUBLE, region_x, dset->obj_id, region_xx);
    else if (PDC_CHAR == dset->obj.type)
        PDCbuf_obj_map((void *)buf, PDC_CHAR, region_x, dset->obj_id, region_xx);
    else
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "data type not supported");

    if ((ret = PDCreg_obtain_lock(dset->obj_id, region_xx, PDC_WRITE, PDC_NOBLOCK)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "can't obtain lock");

    if ((ret = PDCreg_release_lock(dset->obj_id, region_xx, PDC_WRITE)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "can't release lock");
    if ((ret = PDCbuf_obj_unmap(dset->obj_id, dset->obj.reg_id_to)) < 0) {
        HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't unmap object");
    }

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_write() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_dataset_read(void *_dset, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t plist_id,
                      void *buf, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered dataset_read\n");
    fflush(stdout);
#endif

    H5VL_pdc_dset_t *dset = (H5VL_pdc_dset_t *)_dset;
    size_t           type_size;
    uint64_t        *offset;
    int              ndim;
    pdcid_t          region_x, region_xx;
    hsize_t          dims[H5S_MAX_RANK];
    perr_t           ret;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    /* Get memory dataspace object */
    if ((ndim = H5Sget_simple_extent_ndims(mem_space_id)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get number of dimensions");
    if (ndim != H5Sget_simple_extent_dims(mem_space_id, dims, NULL))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get dimensions");

    offset                = (uint64_t *)malloc(sizeof(uint64_t) * ndim);
    offset[0]             = 0;
    region_x              = PDCregion_create(ndim, offset, dims);
    dset->obj.reg_id_from = region_x;

    type_size = H5Tget_size(mem_type_id);
    H5VL__pdc_sel_to_recx_iov(file_space_id, type_size, offset);

    region_xx           = PDCregion_create(ndim, offset, dims);
    dset->obj.reg_id_to = region_xx;
    free(offset);

    dset->mapped = 1;
    if (PDC_INT == dset->obj.type)
        PDCbuf_obj_map((void *)buf, PDC_INT, region_x, dset->obj_id, region_xx);
    else if (PDC_FLOAT == dset->obj.type) {
        PDCbuf_obj_map((void *)buf, PDC_FLOAT, region_x, dset->obj_id, region_xx);
    }
    else if (PDC_DOUBLE == dset->obj.type)
        PDCbuf_obj_map((void *)buf, PDC_DOUBLE, region_x, dset->obj_id, region_xx);
    else if (PDC_CHAR == dset->obj.type)
        PDCbuf_obj_map((void *)buf, PDC_CHAR, region_x, dset->obj_id, region_xx);
    else
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "data type not supported");

    if ((ret = PDCreg_obtain_lock(dset->obj_id, region_xx, PDC_READ, PDC_NOBLOCK)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "can't obtain lock");

    if ((ret = PDCreg_release_lock(dset->obj_id, region_xx, PDC_READ)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "can't release lock");
    if ((ret = PDCbuf_obj_unmap(dset->obj_id, dset->obj.reg_id_to)) < 0) {
        HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't unmap object");
    }

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_read() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_dataset_get(void *_dset, H5VL_dataset_get_args_t *args, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered dataset_get\n");
    fflush(stdout);
#endif

    H5VL_pdc_dset_t *dset = (H5VL_pdc_dset_t *)_dset;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    H5VL_dataset_get_t get_type = (*args).op_type;

    switch (get_type) {
        case H5VL_DATASET_GET_DCPL: {
            hid_t dcpl_id = (*args).args.get_dcpl.dcpl_id;

            /* Retrieve the dataset's creation property list */
            if (dcpl_id < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get dset creation property list");

            break;
        } /* end block */
        case H5VL_DATASET_GET_DAPL: {
            hid_t dapl_id = (*args).args.get_dapl.dapl_id;

            /* Retrieve the dataset's access property list */
            if (dapl_id < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get dset access property list");

            break;
        } /* end block */
        case H5VL_DATASET_GET_SPACE: {
            (*args).args.get_space.space_id = H5Scopy(dset->space_id);

            /* Retrieve the dataset's dataspace */
            break;
        } /* end block */
        case H5VL_DATASET_GET_TYPE: {
            hid_t ret_id = (*args).args.get_type.type_id;

            /* Retrieve the dataset's datatype */
            if (ret_id < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get datatype ID of dataset");
            break;
        } /* end block */
        case H5VL_DATASET_GET_STORAGE_SIZE:
        default:
            HGOTO_ERROR(H5E_VOL, H5E_UNSUPPORTED, FAIL, "can't get this type of information from dataset");
    } /* end switch */

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_get() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_dataset_close(void *_dset, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered dataset_close\n");
    fflush(stdout);
#endif

    H5VL_pdc_dset_t *dset = (H5VL_pdc_dset_t *)_dset;
    perr_t           ret;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(dset);
    if ((ret = PDCobj_close(dset->obj_id)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't close object");
    if (dset->obj.reg_id_from != 0) {
        if ((ret = PDCregion_close(dset->obj.reg_id_from)) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't close region");
    }
    if (dset->obj.reg_id_to != 0) {
        if ((ret = PDCregion_close(dset->obj.reg_id_to) < 0))
            HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't close remote region");
    }

    if (dset->mapped == 1)
        H5_LIST_REMOVE(dset, entry);

    if (H5VL__pdc_dset_free(dset) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't free dataset");

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_close() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl, const H5VL_class_t **conn_cls)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered get_conn_cls\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    herr_t          ret_value;

    /* Check for querying this connector's class */
    if (H5VL_GET_CONN_LVL_CURR == lvl) {
        *conn_cls = &H5VL_pdc_g;
        ret_value = 0;
    } /* end if */
    else
        ret_value = H5VLintrospect_get_conn_cls(o->under_object, o->under_vol_id, lvl, conn_cls);

    return ret_value;
} /* end H5VL_pdc_introspect_get_conn_cls() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_introspect_get_cap_flags(const void *_info, unsigned *cap_flags)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered get_cap_flags\n");
    fflush(stdout);
#endif

    const H5VL_pdc_info_t *info = (const H5VL_pdc_info_t *)_info;
    herr_t                 ret_value;

    /* Invoke the query on the underlying VOL connector */
    ret_value = H5VLintrospect_get_cap_flags(info->under_vol_info, info->under_vol_id, cap_flags);

    /* Bitwise OR our capability flags in */
    if (ret_value >= 0)
        *cap_flags |= H5VL_pdc_g.cap_flags;

    return ret_value;
} /* end H5VL_pdc_introspect_get_cap_flags() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_introspect_opt_query(void *obj, H5VL_subclass_t cls, int opt_type, uint64_t *flags)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered opt_query\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    herr_t          ret_value;
    return 0;
} /* end H5VL_pdc_introspect_opt_query() */

/*---------------------------------------------------------------------------*/
static void *
// store group name in pdc_obj
// retrieve file name and concatenate with group name, create pdc container (check code in file create)
H5VL_pdc_group_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t lcpl_id,
                      hid_t gcpl_id, hid_t gapl_id, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered group_create\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t *group;
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    void           *under;
    char           *group_name = (char *)malloc(strlen(name) + 1);
    strcpy(group_name, name);

    group             = H5VL_pdc_new_obj(under, o->under_vol_id);
    group->file       = o->file;
    group->group_name = group_name;

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return (void *)group;
} /* end H5VL_pdc_group_create() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_group_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t gapl_id,
                    hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered group_open\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t *group;
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    void           *under;
    char           *group_name = (char *)malloc(strlen(name) + 1);
    strcpy(group_name, name);

    group             = H5VL_pdc_new_obj(under, o->under_vol_id);
    group->file       = o->file;
    group->group_name = group_name;

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return (void *)group;
} /* end H5VL_pdc_group_open() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_group_get(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered group_get\n");
    fflush(stdout);
#endif
    return 0;
} /* end H5VL_pdc_group_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_group_close(void *grp, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered group_close\n");
    fflush(stdout);
#endif

    return 0;
} /* end H5VL_pdc_group_close() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_attr_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t type_id,
                     hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered attr_create\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t *attr;
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    void           *under;
    psize_t         value_size;
    pdcid_t         obj_prop, obj_id;

    char *attr_name = (char *)malloc(strlen(name) + 1);
    strcpy(attr_name, name);
    attr                  = H5VL_pdc_new_obj(under, o->under_vol_id);
    attr->attr_name       = attr_name;
    value_size            = H5Sget_select_npoints(space_id) * H5Tget_size(type_id);
    attr->attr_value_size = value_size;
    attr->obj_id          = o->obj_id;

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return (void *)attr;
} /* end H5VL_pdc_attr_create() */
/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_attr_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t aapl_id,
                   hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered attr_open\n");
    fflush(stdout);
#endif
    H5VL_pdc_obj_t *attr;
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    void           *under;
    char           *attr_name = (char *)malloc(strlen(name) + 1);
    strcpy(attr_name, name);
    o->attr_name = attr_name;

    attr            = H5VL_pdc_new_obj(under, o->under_vol_id);
    attr->attr_name = attr_name;
    attr->obj_id    = o->obj_id;

    return (void *)attr;
} /* end H5VL_pdc_attr_open() */
/*---------------------------------------------------------------------------*/
static perr_t
H5VL_pdc_attr_read(void *attr, hid_t mem_type_id, void *buf, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered attr_read\n");
    fflush(stdout);
#endif
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)attr;
    void           *tag_value;
    psize_t        *value_size;
    perr_t          ret_value;

    ret_value = PDCobj_get_tag(o->obj_id, (char *)o->attr_name, &tag_value, &(o->attr_value_size));
    memcpy(buf, tag_value, o->attr_value_size);
    if (tag_value)
        free(tag_value);

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pdc_attr_read() */

/*---------------------------------------------------------------------------*/
static perr_t
H5VL_pdc_attr_write(void *attr, hid_t mem_type_id, const void *buf, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered attr_write\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)attr;
    herr_t          ret_value;

    ret_value = PDCobj_put_tag(o->obj_id, (char *)o->attr_name, (void *)buf, o->attr_value_size);
    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pdc_attr_write() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered attr_get\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    herr_t          ret_value;

    ret_value = H5VLattr_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pdc_attr_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_attr_close(void *attr, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "\nentered attr_close\n");
    fflush(stdout);
#endif

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)attr;
    herr_t          ret_value;
    ret_value = 0;

    return ret_value;
} /* end H5VL_pdc_attr_close() */
