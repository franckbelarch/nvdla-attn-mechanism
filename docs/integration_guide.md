# NVDLA Attention Module Integration Guide

This guide provides instructions for integrating the attention mechanism module into the NVDLA hardware architecture. The attention module extends NVDLA's capabilities to efficiently support transformer-based neural network architectures.

## Integration Overview

The attention module needs to be integrated with the existing NVDLA hardware at several levels:

1. **RTL Integration**: Adding the attention module to the NVDLA hardware design
2. **Interface Integration**: Connecting the module to NVDLA's memory and control interfaces
3. **Programming Model**: Extending NVDLA's programming interface to support attention operations
4. **Compiler Integration**: Updating the compiler to recognize and map attention operations

## RTL Integration

### 1. Module Placement

The attention module should be integrated at the same hierarchical level as other NVDLA processing units like CDMA, CMAC, and SDP. It will receive data from CBUF (Convolution Buffer) and connect to the global memory interface.

```
nvdla/
└── hw/
    └── vmod/
        └── nvdla/
            ├── cacc/     # Existing NVDLA modules
            ├── cbuf/
            ├── cmac/
            ├── csb/
            ├── sdp/
            └── attn/     # New attention module
                ├── NV_NVDLA_attn.v     # Top-level wrapper
                ├── nvdla_attention.sv  # Core implementation
                ├── nvdla_matrix_mult.sv
                └── nvdla_softmax.sv
```

### 2. Adding Module Files

Copy the attention module RTL files to the NVDLA codebase:

```bash
mkdir -p hw/vmod/nvdla/attn
cp src/rtl/nvdla_*.sv hw/vmod/nvdla/attn/
```

Create a Verilog wrapper module `NV_NVDLA_attn.v` that instantiates the SystemVerilog modules and provides the standard NVDLA interface.

### 3. Updating NVDLA Top-Level

Modify the NVDLA top-level file (`NV_NVDLA_partition_*.v`) to instantiate the attention module and connect it to the appropriate interfaces.

## Interface Integration

### 1. Memory Interface

The attention module needs to interface with NVDLA's memory subsystem:

- **CBUF Interface**: Connect to CBUF to read Q, K, V matrices and write results
- **CSB Interface**: Connect to Configuration Space Bus for register access
- **DBBIF Interface**: Connect to Data Backbone Interface for memory access

### 2. Control Interface

Add control signals to coordinate the attention operation with other NVDLA processing:

- **Attention Enable**: Control signal to start the attention operation
- **Done Signal**: Indicates completion of attention calculation
- **Status/Error Signals**: Report operation status and errors

## Programming Model Extension

### 1. Register Interface

Add new CSB registers for the attention configuration:

| Register Name | Address Offset | Description |
|---------------|---------------|-------------|
| ATTN_MODE     | 0x0000 | Configuration mode and enable |
| ATTN_SEQ_LEN  | 0x0004 | Sequence length for operation |
| ATTN_HEAD_DIM | 0x0008 | Dimension of each attention head |
| ATTN_NUM_HEADS| 0x000C | Number of attention heads |
| ATTN_Q_ADDR   | 0x0010 | Base address for Q matrix |
| ATTN_K_ADDR   | 0x0014 | Base address for K matrix |
| ATTN_V_ADDR   | 0x0018 | Base address for V matrix |
| ATTN_OUT_ADDR | 0x001C | Base address for output |
| ATTN_STATUS   | 0x0020 | Status and error flags |

### 2. Memory Layout

Define memory layout for attention operation:

- **Q Matrix**: Stored in row-major order
- **K Matrix**: Stored in row-major order
- **V Matrix**: Stored in row-major order
- **Output**: Stored in row-major order

Memory should be aligned to 32-byte boundaries for efficient access.

## Compiler Integration

### 1. Compile Flow Extensions

Update the NVDLA compiler to support attention operations:

1. Add attention operation to the compiler's operation library
2. Implement mapping from ML framework attention operations to hardware
3. Generate proper register configurations and memory layouts

### 2. Update Layer Support

Modify the layer parser to recognize transformer layers:

```python
def parseTransformerLayer(layer, network):
    # Extract Q, K, V matrices
    # Configure attention operation
    # Set up memory layout
    # Generate register configurations
```

### 3. Driver Updates

Update the NVDLA driver to support the new registers and operations:

```c
nvdla_error_t NvDlaAttentionSubmit(NvDlaDeviceHandle handle, 
                                   struct NvDlaAttentionParams *params)
{
    // Configure attention registers
    // Submit operation to hardware
    // Wait for completion
    // Return status
}
```

## Testing and Verification

### 1. Unit Testing

Verify the attention module in isolation:

```bash
cd hw/verif/tests
./run_test.py -t nvdla_attn_unit_test
```

### 2. Integration Testing

Test the attention module within the complete NVDLA system:

```bash
cd hw/verif/tests
./run_test.py -t nvdla_system_attn_test
```

### 3. Performance Validation

Measure and verify performance metrics:

- Throughput: operations/second
- Latency: cycles per attention operation
- Memory bandwidth utilization
- Resource utilization

## Example Usage

Below is an example of using the attention operation from the NVDLA runtime:

```c
void executeAttention(NvDlaDeviceHandle handle)
{
    // Configure attention parameters
    struct NvDlaAttentionParams params;
    params.seqLength = 128;
    params.headDim = 64;
    params.numHeads = 8;
    params.qAddress = q_buffer_address;
    params.kAddress = k_buffer_address;
    params.vAddress = v_buffer_address;
    params.outAddress = output_buffer_address;
    
    // Submit attention operation
    NvDlaAttentionSubmit(handle, &params);
    
    // Wait for completion
    NvDlaWaitForCompletion(handle);
}
```

## Known Limitations and Future Improvements

- Current implementation supports fixed-point operations only
- Maximum sequence length is limited by internal buffer size
- Future work to add support for sparse attention mechanisms
- Optimization of memory access patterns for better performance

## Conclusion

With these integration steps, the NVDLA hardware will be extended with attention mechanism support, enabling efficient execution of transformer-based neural networks for applications such as natural language processing, machine translation, and document understanding.