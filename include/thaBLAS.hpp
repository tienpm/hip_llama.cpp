#pragma once

typedef enum
{
    THABLAS_STATUS_SUCCESS           = 0, /**< Function succeeds */
    THABLAS_STATUS_NOT_INITIALIZED   = 1, /**< HIPBLAS library not initialized */
    THABLAS_STATUS_ALLOC_FAILED      = 2, /**< resource allocation failed */
    THABLAS_STATUS_INVALID_VALUE     = 3, /**< unsupported numerical value was passed to function */
    THABLAS_STATUS_MAPPING_ERROR     = 4, /**< access to GPU memory space failed */
    THABLAS_STATUS_EXECUTION_FAILED  = 5, /**< GPU program failed to execute */
    THABLAS_STATUS_INTERNAL_ERROR    = 6, /**< an internal HIPBLAS operation failed */
    THABLAS_STATUS_NOT_SUPPORTED     = 7, /**< function not implemented */
    THABLAS_STATUS_ARCH_MISMATCH     = 8, /**< architecture mismatch */
    THABLAS_STATUS_HANDLE_IS_NULLPTR = 9, /**< hipBLAS handle is null pointer */
    THABLAS_STATUS_INVALID_ENUM      = 10, /**<  unsupported enum value was passed to function */
    THABLAS_STATUS_UNKNOWN           = 11, /**<  back-end returned an unsupported status code */
} thablasStatus_t;

typedef struct 
{
    int current_gpu_id;
} thablasHandle_t;

thablasStatus_t thablasCreate(thablasHandle_t* handle);

thablasStatus_t thablasDestroy(thablasHandle_t handle);

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

/*! @{
    \brief BLAS Level 1 API

    \details
        B = A / v

    @param[in]
    handle    THABLAS handle
    @param[in]
    n 
              number of elements of vector A and B
    @param[in]
    A          device pointer storing matrix A accessible from device.
    @param[in]
    B          device pointer storing matrix B accessible from device.
    @param[in]
    val        scalar number to divide

    ********************************************************************/
thablasStatus_t thablas_Svds(thablasHandle_t handle, int n, float* A, float* B, float val);

thablasStatus_t thablas_c2d_Svds(int n, float* A, float* B, float val, int max_num_gpus);

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

/*! @{
    \brief BLAS Level 3 API

    \details
        C = A*B

    @param[in]
    m         [int]
              number of rows of matrices A and C
    @param[in]
    n         [int]
              number of columns of matrices B and C
    @param[in]
    k         [int]
              number of columns of matrix A and number of rows of matrix B
    @param[in]
    A          pointer storing matrix A accessible from CPU.
    @param[in]
    B          pointer storing matrix B accessible from CPU.
    @param[in, out]
    C          pointer storing matrix C accessible from CPU.
    @param[in]
    max_num_gpus maximum number of GPU will be used

    ********************************************************************/
thablasStatus_t thablas_c2d_Sgemm(int m, int n, int k, float* A, float* B, float* C, int max_num_gpus);

/*! @{
    \brief BLAS Level 3 API

    \details
        C = A*B
    @param[in]
    handle    THABLAS handle
    @param[in]
    m         [int]
              number of rows of matrices A and C
    @param[in]
    n         [int]
              number of columns of matrices B and C
    @param[in]
    k         [int]
              number of columns of matrix A and number of rows of matrix B
    @param[in]
    A         pointer storing matrix A accessible from device.
    @param[in]
    B         pointer storing matrix B accessible from device.
    @param[in, out]
    C         pointer storing matrix C accessible from device.

    ********************************************************************/
thablasStatus_t thablas_Sgemm(thablasHandle_t handle, int m, int n, int k, float* A, float* B, float* C);

// thablasStatus_t thablas_Ssum(thablasHandle_t handle, int vector_size, float* vector_in,  float &sum_result);
