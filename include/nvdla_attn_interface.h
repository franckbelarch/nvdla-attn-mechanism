/*
 * NVDLA Attention Module Interface
 * 
 * This header defines the programming interface to the NVDLA attention module.
 */

#ifndef _NVDLA_ATTN_INTERFACE_H_
#define _NVDLA_ATTN_INTERFACE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Register addresses
#define NVDLA_ATTN_BASE_ADDRESS     0x7000

// Register offsets
#define NVDLA_ATTN_CONTROL          0x00    // Control register
#define NVDLA_ATTN_STATUS           0x04    // Status register
#define NVDLA_ATTN_SEQ_LENGTH       0x08    // Sequence length
#define NVDLA_ATTN_HEAD_DIM         0x0C    // Head dimension
#define NVDLA_ATTN_NUM_HEADS        0x10    // Number of heads
#define NVDLA_ATTN_Q_ADDR           0x14    // Q matrix base address
#define NVDLA_ATTN_K_ADDR           0x18    // K matrix base address
#define NVDLA_ATTN_V_ADDR           0x1C    // V matrix base address
#define NVDLA_ATTN_OUT_ADDR         0x20    // Output base address
#define NVDLA_ATTN_PERF_CYCLES      0x24    // Performance counter - cycles
#define NVDLA_ATTN_PERF_OPS         0x28    // Performance counter - operations

// Control register bit definitions
#define NVDLA_ATTN_CONTROL_ENABLE           (1 << 0)    // Enable attention operation
#define NVDLA_ATTN_CONTROL_MASK_ENABLE      (1 << 1)    // Enable attention mask
#define NVDLA_ATTN_CONTROL_INT_ENABLE       (1 << 2)    // Enable interrupt

// Status register bit definitions
#define NVDLA_ATTN_STATUS_DONE              (1 << 0)    // Operation complete
#define NVDLA_ATTN_STATUS_ERROR             (1 << 1)    // Error flag
#define NVDLA_ATTN_STATUS_STATE_MASK        (0xF << 4)  // Current state
#define NVDLA_ATTN_STATUS_STATE_SHIFT       4           // State shift

// Parameter limits
#define NVDLA_ATTN_MAX_SEQ_LENGTH           256         // Maximum sequence length
#define NVDLA_ATTN_MAX_HEAD_DIM             128         // Maximum head dimension
#define NVDLA_ATTN_MAX_NUM_HEADS            16          // Maximum number of heads

/**
 * Attention operation parameters
 */
typedef struct {
    uint32_t seq_length;    // Sequence length
    uint32_t head_dim;      // Head dimension
    uint32_t num_heads;     // Number of heads
    uint32_t q_addr;        // Q matrix base address
    uint32_t k_addr;        // K matrix base address
    uint32_t v_addr;        // V matrix base address
    uint32_t out_addr;      // Output base address
    uint8_t  mask_enable;   // Enable attention mask
    uint8_t  int_enable;    // Enable interrupt
} nvdla_attn_params_t;

/**
 * NVDLA error codes
 */
typedef enum {
    NVDLA_ATTN_SUCCESS                = 0,
    NVDLA_ATTN_ERR_INVALID_PARAM      = 1,
    NVDLA_ATTN_ERR_TIMEOUT            = 2,
    NVDLA_ATTN_ERR_HARDWARE_ERROR     = 3,
    NVDLA_ATTN_ERR_MEMORY_ACCESS      = 4
} nvdla_attn_error_t;

/**
 * Submit an attention operation to the NVDLA hardware
 *
 * @param handle   Device handle
 * @param params   Attention operation parameters
 * @return         Error code
 */
nvdla_attn_error_t NvDlaAttentionSubmit(void* handle, const nvdla_attn_params_t* params);

/**
 * Wait for an attention operation to complete
 *
 * @param handle   Device handle
 * @param timeout  Timeout in milliseconds
 * @return         Error code
 */
nvdla_attn_error_t NvDlaAttentionWait(void* handle, uint32_t timeout);

/**
 * Get the state of the attention module
 *
 * @param handle   Device handle
 * @param status   Pointer to store status value
 * @return         Error code
 */
nvdla_attn_error_t NvDlaAttentionGetStatus(void* handle, uint32_t* status);

/**
 * Get performance metrics for an attention operation
 *
 * @param handle     Device handle
 * @param cycles     Pointer to store cycle count
 * @param operations Pointer to store operation count
 * @return           Error code
 */
nvdla_attn_error_t NvDlaAttentionGetPerformance(void* handle, uint32_t* cycles, uint32_t* operations);

#ifdef __cplusplus
}
#endif

#endif /* _NVDLA_ATTN_INTERFACE_H_ */