/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of the HDF5 PDC VOL connector. The full copyright       *
 * notice, including terms governing use, modification, and redistribution,  *
 * is contained in the COPYING file, which can be found at the root of the   *
 * source code distribution tree.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Purpose:	The private header file for the PDC VOL plugin.
 */
#ifndef H5VLpdc_H
#define H5VLpdc_H
/* Include package's public header */
#include "H5VLpdc_public.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Identifier for the pass-through VOL connector */
#define H5VL_PDC (H5VL_pdc_register())
H5_DLL hid_t H5VL_pdc_register(void);

/* Nothing */

#ifdef __cplusplus
}
#endif

#endif /* H5VLpdc_H */
