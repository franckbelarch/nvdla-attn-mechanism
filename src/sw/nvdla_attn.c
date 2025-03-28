/*
 * NVDLA Attention Module Runtime Implementation
 *
 * This file implements the C interface for the NVDLA attention module.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "../../include/nvdla_attn_interface.h"

// Function declarations for internal NVDLA API functions 
// These would be defined in the NVDLA runtime library
extern int nvdla_write_reg(void *handle, uint32_t addr, uint32_t value);
extern int nvdla_read_reg(void *handle, uint32_t addr, uint32_t *value);
extern int nvdla_wait_interrupt(void *handle, uint32_t timeout);

/**
 * Calculate the full register address
 */
static inline uint32_t attn_reg_addr(uint32_t offset) {
    return NVDLA_ATTN_BASE_ADDRESS + offset;
}

/**
 * Validate attention parameters
 */
static nvdla_attn_error_t validate_params(const nvdla_attn_params_t *params) {
    if (!params) {
        return NVDLA_ATTN_ERR_INVALID_PARAM;
    }
    
    if (params->seq_length == 0 || params->seq_length > NVDLA_ATTN_MAX_SEQ_LENGTH) {
        return NVDLA_ATTN_ERR_INVALID_PARAM;
    }
    
    if (params->head_dim == 0 || params->head_dim > NVDLA_ATTN_MAX_HEAD_DIM) {
        return NVDLA_ATTN_ERR_INVALID_PARAM;
    }
    
    if (params->num_heads == 0 || params->num_heads > NVDLA_ATTN_MAX_NUM_HEADS) {
        return NVDLA_ATTN_ERR_INVALID_PARAM;
    }
    
    // Memory addresses should be appropriate for the hardware
    // Typically, they need to be aligned to a specific boundary
    if ((params->q_addr & 0x1F) != 0 || 
        (params->k_addr & 0x1F) != 0 || 
        (params->v_addr & 0x1F) != 0 || 
        (params->out_addr & 0x1F) != 0) {
        return NVDLA_ATTN_ERR_INVALID_PARAM;
    }
    
    return NVDLA_ATTN_SUCCESS;
}

/**
 * Submit an attention operation to the NVDLA hardware
 */
nvdla_attn_error_t NvDlaAttentionSubmit(void *handle, const nvdla_attn_params_t *params) {
    int ret;
    nvdla_attn_error_t error;
    uint32_t control_value = 0;
    
    // Validate parameters
    error = validate_params(params);
    if (error != NVDLA_ATTN_SUCCESS) {
        return error;
    }
    
    // Check if the attention module is idle
    uint32_t status;
    ret = nvdla_read_reg(handle, attn_reg_addr(NVDLA_ATTN_STATUS), &status);
    if (ret != 0) {
        return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    }
    
    if (status & NVDLA_ATTN_STATUS_DONE) {
        // Clear done flag by writing to the status register
        ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_STATUS), 0);
        if (ret != 0) {
            return NVDLA_ATTN_ERR_HARDWARE_ERROR;
        }
    }
    
    // Configure attention operation
    ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_SEQ_LENGTH), params->seq_length);
    if (ret != 0) return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    
    ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_HEAD_DIM), params->head_dim);
    if (ret != 0) return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    
    ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_NUM_HEADS), params->num_heads);
    if (ret != 0) return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    
    ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_Q_ADDR), params->q_addr);
    if (ret != 0) return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    
    ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_K_ADDR), params->k_addr);
    if (ret != 0) return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    
    ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_V_ADDR), params->v_addr);
    if (ret != 0) return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    
    ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_OUT_ADDR), params->out_addr);
    if (ret != 0) return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    
    // Set control register
    control_value = NVDLA_ATTN_CONTROL_ENABLE;
    if (params->mask_enable) {
        control_value |= NVDLA_ATTN_CONTROL_MASK_ENABLE;
    }
    if (params->int_enable) {
        control_value |= NVDLA_ATTN_CONTROL_INT_ENABLE;
    }
    
    ret = nvdla_write_reg(handle, attn_reg_addr(NVDLA_ATTN_CONTROL), control_value);
    if (ret != 0) return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    
    return NVDLA_ATTN_SUCCESS;
}

/**
 * Wait for an attention operation to complete
 */
nvdla_attn_error_t NvDlaAttentionWait(void *handle, uint32_t timeout) {
    int ret;
    uint32_t status;
    uint32_t start_time = (uint32_t)time(NULL);
    
    // If interrupts are enabled, wait for interrupt
    ret = nvdla_wait_interrupt(handle, timeout);
    if (ret == 0) {
        // Interrupt received, return success
        return NVDLA_ATTN_SUCCESS;
    }
    
    // Interrupt wait failed or not enabled, poll for completion
    while (1) {
        ret = nvdla_read_reg(handle, attn_reg_addr(NVDLA_ATTN_STATUS), &status);
        if (ret != 0) {
            return NVDLA_ATTN_ERR_HARDWARE_ERROR;
        }
        
        if (status & NVDLA_ATTN_STATUS_DONE) {
            // Operation completed
            return NVDLA_ATTN_SUCCESS;
        }
        
        if (status & NVDLA_ATTN_STATUS_ERROR) {
            // Hardware error occurred
            return NVDLA_ATTN_ERR_HARDWARE_ERROR;
        }
        
        // Check for timeout
        if (timeout > 0 && ((uint32_t)time(NULL) - start_time) > timeout) {
            return NVDLA_ATTN_ERR_TIMEOUT;
        }
        
        // Sleep briefly to avoid consuming CPU
        usleep(1000); // 1ms
    }
    
    return NVDLA_ATTN_SUCCESS;
}

/**
 * Get the state of the attention module
 */
nvdla_attn_error_t NvDlaAttentionGetStatus(void *handle, uint32_t *status) {
    int ret;
    
    if (!status) {
        return NVDLA_ATTN_ERR_INVALID_PARAM;
    }
    
    ret = nvdla_read_reg(handle, attn_reg_addr(NVDLA_ATTN_STATUS), status);
    if (ret != 0) {
        return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    }
    
    return NVDLA_ATTN_SUCCESS;
}

/**
 * Get performance metrics for an attention operation
 */
nvdla_attn_error_t NvDlaAttentionGetPerformance(void *handle, uint32_t *cycles, uint32_t *operations) {
    int ret;
    
    if (!cycles || !operations) {
        return NVDLA_ATTN_ERR_INVALID_PARAM;
    }
    
    ret = nvdla_read_reg(handle, attn_reg_addr(NVDLA_ATTN_PERF_CYCLES), cycles);
    if (ret != 0) {
        return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    }
    
    ret = nvdla_read_reg(handle, attn_reg_addr(NVDLA_ATTN_PERF_OPS), operations);
    if (ret != 0) {
        return NVDLA_ATTN_ERR_HARDWARE_ERROR;
    }
    
    return NVDLA_ATTN_SUCCESS;
}