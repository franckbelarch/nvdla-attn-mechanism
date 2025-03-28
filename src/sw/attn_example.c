/*
 * NVDLA Attention Module Example
 *
 * This example demonstrates how to use the NVDLA attention module.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../include/nvdla_attn_interface.h"

// External function declarations for NVDLA runtime API
extern void* NvDlaOpen(void);
extern void NvDlaClose(void* handle);
extern int NvDlaAllocMem(void* handle, uint32_t size, uint32_t* addr);
extern int NvDlaFreeMem(void* handle, uint32_t addr);
extern int NvDlaWriteMem(void* handle, uint32_t addr, const void* data, uint32_t size);
extern int NvDlaReadMem(void* handle, uint32_t addr, void* data, uint32_t size);

// Test parameters
#define SEQ_LEN        16
#define HEAD_DIM       64
#define NUM_HEADS      2

// Helper functions
void generate_test_data(void* handle, uint32_t* q_addr, uint32_t* k_addr, uint32_t* v_addr, uint32_t* out_addr);
void verify_results(void* handle, uint32_t out_addr);
void print_performance(void* handle);

int main(int argc, char** argv) {
    void* handle;
    nvdla_attn_params_t params;
    nvdla_attn_error_t error;
    uint32_t q_addr, k_addr, v_addr, out_addr;
    
    printf("NVDLA Attention Module Example\n");
    printf("==============================\n");
    
    // Open NVDLA device
    handle = NvDlaOpen();
    if (!handle) {
        fprintf(stderr, "Error: Failed to open NVDLA device\n");
        return 1;
    }
    
    // Generate test data
    printf("Generating test data...\n");
    generate_test_data(handle, &q_addr, &k_addr, &v_addr, &out_addr);
    
    // Configure attention parameters
    memset(&params, 0, sizeof(params));
    params.seq_length = SEQ_LEN;
    params.head_dim = HEAD_DIM;
    params.num_heads = NUM_HEADS;
    params.q_addr = q_addr;
    params.k_addr = k_addr;
    params.v_addr = v_addr;
    params.out_addr = out_addr;
    params.mask_enable = 0;
    params.int_enable = 1;
    
    // Submit attention operation
    printf("Submitting attention operation...\n");
    error = NvDlaAttentionSubmit(handle, &params);
    if (error != NVDLA_ATTN_SUCCESS) {
        fprintf(stderr, "Error: Failed to submit attention operation (error code %d)\n", error);
        NvDlaClose(handle);
        return 1;
    }
    
    // Wait for completion
    printf("Waiting for attention operation to complete...\n");
    error = NvDlaAttentionWait(handle, 5000); // 5 second timeout
    if (error != NVDLA_ATTN_SUCCESS) {
        fprintf(stderr, "Error: Attention operation failed (error code %d)\n", error);
        NvDlaClose(handle);
        return 1;
    }
    
    // Verify results
    printf("Verifying results...\n");
    verify_results(handle, out_addr);
    
    // Print performance metrics
    print_performance(handle);
    
    // Free memory
    NvDlaFreeMem(handle, q_addr);
    NvDlaFreeMem(handle, k_addr);
    NvDlaFreeMem(handle, v_addr);
    NvDlaFreeMem(handle, out_addr);
    
    // Close NVDLA device
    NvDlaClose(handle);
    
    printf("Example completed successfully!\n");
    return 0;
}

/**
 * Generate test data for attention operation
 */
void generate_test_data(void* handle, uint32_t* q_addr, uint32_t* k_addr, uint32_t* v_addr, uint32_t* out_addr) {
    uint32_t q_size = SEQ_LEN * HEAD_DIM * NUM_HEADS * sizeof(int16_t);
    uint32_t k_size = SEQ_LEN * HEAD_DIM * NUM_HEADS * sizeof(int16_t);
    uint32_t v_size = SEQ_LEN * HEAD_DIM * NUM_HEADS * sizeof(int16_t);
    uint32_t out_size = SEQ_LEN * HEAD_DIM * NUM_HEADS * sizeof(int16_t);
    
    // Allocate memory
    NvDlaAllocMem(handle, q_size, q_addr);
    NvDlaAllocMem(handle, k_size, k_addr);
    NvDlaAllocMem(handle, v_size, v_addr);
    NvDlaAllocMem(handle, out_size, out_addr);
    
    // Create test data buffers
    int16_t* q_data = (int16_t*)malloc(q_size);
    int16_t* k_data = (int16_t*)malloc(k_size);
    int16_t* v_data = (int16_t*)malloc(v_size);
    
    if (!q_data || !k_data || !v_data) {
        fprintf(stderr, "Error: Failed to allocate memory for test data\n");
        exit(1);
    }
    
    // Initialize with test pattern
    // This example uses identity matrices for simplicity
    memset(q_data, 0, q_size);
    memset(k_data, 0, k_size);
    memset(v_data, 0, v_size);
    
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int offset = h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + j;
                
                if (i == j) {
                    // Identity matrix: 1.0 on diagonal in fixed-point format (0x0100)
                    q_data[offset] = 0x0100;
                    k_data[offset] = 0x0100;
                    v_data[offset] = 0x0100;
                }
            }
        }
    }
    
    // Write data to hardware memory
    NvDlaWriteMem(handle, *q_addr, q_data, q_size);
    NvDlaWriteMem(handle, *k_addr, k_data, k_size);
    NvDlaWriteMem(handle, *v_addr, v_data, v_size);
    
    // Free local buffers
    free(q_data);
    free(k_data);
    free(v_data);
}

/**
 * Verify results of attention operation
 */
void verify_results(void* handle, uint32_t out_addr) {
    uint32_t out_size = SEQ_LEN * HEAD_DIM * NUM_HEADS * sizeof(int16_t);
    int16_t* out_data = (int16_t*)malloc(out_size);
    
    if (!out_data) {
        fprintf(stderr, "Error: Failed to allocate memory for result data\n");
        exit(1);
    }
    
    // Read output from hardware memory
    NvDlaReadMem(handle, out_addr, out_data, out_size);
    
    // For this simple example with identity matrices, 
    // the output should also be an identity matrix
    int errors = 0;
    for (int h = 0; h < NUM_HEADS; h++) {
        printf("Head %d:\n", h);
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int offset = h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + j;
                int16_t expected = (i == j) ? 0x0100 : 0x0000;
                
                // Allow some tolerance for fixed-point arithmetic
                int16_t diff = abs(out_data[offset] - expected);
                if (diff > 0x0010) { // Tolerance of 0.0625 in fixed-point
                    errors++;
                    if (errors < 10) { // Limit number of error messages
                        printf("Error at [%d,%d,%d]: Expected 0x%04x, got 0x%04x\n", 
                               h, i, j, expected, out_data[offset]);
                    }
                }
                
                // Print a sample of the output matrix
                if (h == 0 && i < 4 && j < 4) {
                    printf("%04x ", out_data[offset]);
                }
            }
            if (h == 0 && i < 4) {
                printf("\n");
            }
        }
    }
    
    if (errors > 0) {
        printf("Verification failed with %d errors\n", errors);
    } else {
        printf("Verification successful!\n");
    }
    
    free(out_data);
}

/**
 * Print performance metrics
 */
void print_performance(void* handle) {
    uint32_t cycles, operations;
    nvdla_attn_error_t error;
    
    error = NvDlaAttentionGetPerformance(handle, &cycles, &operations);
    if (error != NVDLA_ATTN_SUCCESS) {
        fprintf(stderr, "Error: Failed to get performance metrics\n");
        return;
    }
    
    printf("\nPerformance Metrics:\n");
    printf("-------------------\n");
    printf("Cycles: %u\n", cycles);
    printf("Operations: %u\n", operations);
    printf("Operations per cycle: %.2f\n", (float)operations / cycles);
}