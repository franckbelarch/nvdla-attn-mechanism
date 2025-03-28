#!/usr/bin/env python3
"""
NVDLA Attention Module Benchmark Script
This script performs comprehensive benchmarking of the NVDLA attention module
"""

import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set plotting style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # For newer versions
except:
    try:
        plt.style.use('seaborn')  # Fallback
    except:
        pass  # Use default style if seaborn is not available

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6)
})

class AttentionBenchmark:
    """
    Benchmark for NVDLA attention module
    """
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        
        # Default benchmark parameters
        self.seq_lengths = [16, 32, 64, 128, 256]
        self.head_dims = [32, 64, 128]
        self.num_heads = [1, 2, 4, 8, 16]
        self.batch_sizes = [1, 2, 4, 8]
        
        # Results storage
        self.throughput_results = {}
        self.latency_results = {}
        self.resource_utilization = {}
        self.accuracy_results = {}
        
    def _reference_attention(self, query, key, value, mask=None):
        """
        Reference implementation of attention mechanism in NumPy
        """
        # Calculate dot products
        d_k = query.shape[-1]
        scores = np.matmul(query, np.transpose(key, (0, 2, 1))) / np.sqrt(d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores * mask + -1e9 * (1 - mask)
        
        # Apply softmax
        attention_weights = self._softmax(scores)
        
        # Calculate weighted sum
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def _softmax(self, x):
        """
        Stable softmax implementation
        """
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def _simulate_nvdla_attention(self, query, key, value, seq_len, head_dim, num_heads):
        """
        Simulate NVDLA attention module using the reference implementation
        Apply quantization and other hardware constraints
        """
        # Clock frequency (MHz)
        clock_freq = 1000
        
        # Conversion to fixed-point (16-bit, 8 fractional bits)
        frac_bits = 8
        scale = 2**frac_bits
        
        # Quantize inputs
        q_quant = np.round(query * scale).astype(np.int16)
        k_quant = np.round(key * scale).astype(np.int16)
        v_quant = np.round(value * scale).astype(np.int16)
        
        # Hardware constraints
        # Assume NVDLA can process 16 elements per cycle
        elements_per_cycle = 16
        
        # Fixed-point matrix multiplication (QK^T)
        start_time = time.time()
        scores_fixed = np.matmul(q_quant, np.transpose(k_quant, (0, 2, 1)))
        scores_fixed = scores_fixed / (np.sqrt(head_dim) * scale)
        
        # Fixed-point softmax approximation
        # Find max for numerical stability
        scores_max = np.max(scores_fixed, axis=-1, keepdims=True)
        scores_norm = scores_fixed - scores_max
        
        # Approximate exp with lookup table or piece-wise linear
        # Here we use a simplified model: y = 1 + x + x^2/2 for x >= 0, 1/(1-x) for x < 0
        exp_approx = np.zeros_like(scores_norm, dtype=np.float32)
        pos_mask = scores_norm >= 0
        exp_approx[pos_mask] = scale * (1.0 + scores_norm[pos_mask]/scale + 
                                       (scores_norm[pos_mask]/scale)**2 / 2)
        exp_approx[~pos_mask] = scale / (1.0 - scores_norm[~pos_mask]/scale)
        
        # Normalize
        softmax_denom = np.sum(exp_approx, axis=-1, keepdims=True)
        attn_weights_fixed = (exp_approx * scale) // softmax_denom
        
        # Matmul with values
        output_fixed = np.matmul(attn_weights_fixed.astype(np.int32), v_quant)
        output_fixed = output_fixed // scale
        
        # Convert back to float for comparison
        output_float = output_fixed.astype(np.float32) / scale
        end_time = time.time()
        
        # Calculate estimated cycles
        # QK^T: seq_len^2 * head_dim operations
        qk_ops = seq_len * seq_len * head_dim
        # Softmax: 5*seq_len operations (normalize, exp, sum, divide)
        softmax_ops = 5 * seq_len * seq_len
        # Output: seq_len^2 * head_dim operations
        out_ops = seq_len * seq_len * head_dim
        
        total_ops = qk_ops + softmax_ops + out_ops
        
        # Each cycle processes elements_per_cycle elements
        cycles = total_ops // elements_per_cycle
        # Add overhead for memory transfers and control
        cycles = int(cycles * 1.2)  # 20% overhead
        
        latency_us = (cycles * 1000) / clock_freq  # convert to microseconds
        
        # Hardware resource estimation based on model
        luts_per_mac = 40  # Approximate LUTs per MAC unit
        luts_per_softmax = 100  # Approximate LUTs for softmax unit
        luts_per_controller = 500  # Approximate LUTs for control logic
        
        # Total resource estimation
        total_luts = (elements_per_cycle * luts_per_mac) + luts_per_softmax + luts_per_controller
        
        # Power estimation (very rough approximation)
        # Assuming 1mW per MAC, 5mW for softmax, 10mW for control
        power_mw = (elements_per_cycle * 1) + 5 + 10
        
        # Calculate normalized ops per second
        ops = 2 * qk_ops + softmax_ops + 2 * out_ops  # Include multiply and add as separate ops
        ops_per_second = ops / (end_time - start_time)
        
        result = {
            "latency_us": latency_us,
            "cycles": cycles,
            "ops": ops,
            "ops_per_second": ops_per_second,
            "throughput_gops": ops_per_second / 1e9,
            "resource_luts": total_luts,
            "power_mw": power_mw
        }
        
        return output_float, result
    
    def _calculate_accuracy(self, reference, simulation):
        """
        Calculate accuracy metrics between reference and simulation
        """
        abs_error = np.abs(reference - simulation)
        rel_error = abs_error / (np.abs(reference) + 1e-10)
        
        result = {
            "mean_absolute_error": float(np.mean(abs_error)),
            "max_absolute_error": float(np.max(abs_error)),
            "mean_relative_error": float(np.mean(rel_error)),
            "max_relative_error": float(np.max(rel_error)),
            "rmse": float(np.sqrt(np.mean((reference - simulation) ** 2)))
        }
        
        return result
    
    def benchmark_sequence_length(self):
        """
        Benchmark performance scaling with sequence length
        """
        results = {"throughput": [], "latency": [], "accuracy": []}
        
        head_dim = 64  # Fixed head dimension
        num_heads = 8  # Fixed number of heads
        
        for seq_len in tqdm(self.seq_lengths, desc="Benchmarking sequence length"):
            # Generate random data
            query = np.random.randn(1, seq_len, head_dim)
            key = np.random.randn(1, seq_len, head_dim)
            value = np.random.randn(1, seq_len, head_dim)
            
            # Run reference implementation
            ref_output, _ = self._reference_attention(query, key, value)
            
            # Run simulated NVDLA implementation
            sim_output, perf = self._simulate_nvdla_attention(query, key, value, seq_len, head_dim, num_heads)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(ref_output, sim_output)
            
            # Store results
            results["throughput"].append({
                "seq_length": seq_len,
                "throughput_gops": perf["throughput_gops"]
            })
            
            results["latency"].append({
                "seq_length": seq_len,
                "latency_us": perf["latency_us"]
            })
            
            results["accuracy"].append({
                "seq_length": seq_len,
                "mean_relative_error": accuracy["mean_relative_error"]
            })
        
        self.seq_length_results = results
        
        # Save results
        with open(os.path.join(self.output_dir, "data", "seq_length_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate plots
        self._plot_seq_length_results(results)
        
        return results
    
    def benchmark_head_dimension(self):
        """
        Benchmark performance scaling with head dimension
        """
        results = {"throughput": [], "latency": [], "accuracy": []}
        
        seq_len = 64  # Fixed sequence length
        num_heads = 8  # Fixed number of heads
        
        for head_dim in tqdm(self.head_dims, desc="Benchmarking head dimension"):
            # Generate random data
            query = np.random.randn(1, seq_len, head_dim)
            key = np.random.randn(1, seq_len, head_dim)
            value = np.random.randn(1, seq_len, head_dim)
            
            # Run reference implementation
            ref_output, _ = self._reference_attention(query, key, value)
            
            # Run simulated NVDLA implementation
            sim_output, perf = self._simulate_nvdla_attention(query, key, value, seq_len, head_dim, num_heads)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(ref_output, sim_output)
            
            # Store results
            results["throughput"].append({
                "head_dim": head_dim,
                "throughput_gops": perf["throughput_gops"]
            })
            
            results["latency"].append({
                "head_dim": head_dim,
                "latency_us": perf["latency_us"]
            })
            
            results["accuracy"].append({
                "head_dim": head_dim,
                "mean_relative_error": accuracy["mean_relative_error"]
            })
        
        self.head_dim_results = results
        
        # Save results
        with open(os.path.join(self.output_dir, "data", "head_dim_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate plots
        self._plot_head_dim_results(results)
        
        return results
    
    def benchmark_num_heads(self):
        """
        Benchmark performance scaling with number of heads
        """
        results = {"throughput": [], "latency": [], "accuracy": []}
        
        seq_len = 64   # Fixed sequence length
        head_dim = 64  # Fixed head dimension
        
        for heads in tqdm(self.num_heads, desc="Benchmarking number of heads"):
            # Generate random data
            query = np.random.randn(1, seq_len, head_dim)
            key = np.random.randn(1, seq_len, head_dim)
            value = np.random.randn(1, seq_len, head_dim)
            
            # Run reference implementation
            ref_output, _ = self._reference_attention(query, key, value)
            
            # Run simulated NVDLA implementation
            sim_output, perf = self._simulate_nvdla_attention(query, key, value, seq_len, head_dim, heads)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(ref_output, sim_output)
            
            # Store results
            results["throughput"].append({
                "num_heads": heads,
                "throughput_gops": perf["throughput_gops"]
            })
            
            results["latency"].append({
                "num_heads": heads,
                "latency_us": perf["latency_us"]
            })
            
            results["accuracy"].append({
                "num_heads": heads,
                "mean_relative_error": accuracy["mean_relative_error"]
            })
        
        self.num_heads_results = results
        
        # Save results
        with open(os.path.join(self.output_dir, "data", "num_heads_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate plots
        self._plot_num_heads_results(results)
        
        return results
    
    def benchmark_comparison(self):
        """
        Benchmark comparison with CPU and theoretical GPU implementation
        """
        results = {"cpu": [], "gpu": [], "nvdla": []}
        
        seq_len = 128  # Fixed sequence length
        head_dim = 64  # Fixed head dimension
        num_heads = 8  # Fixed number of heads
        
        # Generate random data
        query = np.random.randn(1, seq_len, head_dim)
        key = np.random.randn(1, seq_len, head_dim)
        value = np.random.randn(1, seq_len, head_dim)
        
        # CPU implementation (actual measurement)
        start_time = time.time()
        self._reference_attention(query, key, value)
        cpu_time = time.time() - start_time
        
        # Calculate CPU ops per second
        qk_ops = seq_len * seq_len * head_dim
        softmax_ops = 5 * seq_len * seq_len
        out_ops = seq_len * seq_len * head_dim
        total_ops = qk_ops + softmax_ops + out_ops
        
        cpu_ops_per_second = total_ops / cpu_time
        cpu_latency_us = cpu_time * 1e6
        
        # NVDLA simulation
        _, nvdla_perf = self._simulate_nvdla_attention(query, key, value, seq_len, head_dim, num_heads)
        
        # GPU performance (estimated based on typical GPU performance)
        # Assuming GPU is ~20x faster than CPU for this workload
        gpu_ops_per_second = cpu_ops_per_second * 20
        gpu_latency_us = cpu_latency_us / 20
        
        # Store CPU results
        results["cpu"].append({
            "throughput_gops": cpu_ops_per_second / 1e9,
            "latency_us": cpu_latency_us,
            "power_mw": 15000  # Rough estimate for CPU power
        })
        
        # Store GPU results
        results["gpu"].append({
            "throughput_gops": gpu_ops_per_second / 1e9,
            "latency_us": gpu_latency_us,
            "power_mw": 250000  # Rough estimate for GPU power
        })
        
        # Store NVDLA results
        results["nvdla"].append({
            "throughput_gops": nvdla_perf["throughput_gops"],
            "latency_us": nvdla_perf["latency_us"],
            "power_mw": nvdla_perf["power_mw"]
        })
        
        self.comparison_results = results
        
        # Save results
        with open(os.path.join(self.output_dir, "data", "comparison_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate plots
        self._plot_comparison_results(results)
        
        return results
    
    def _plot_seq_length_results(self, results):
        """
        Generate plots for sequence length benchmarks
        """
        # Throughput vs Sequence Length
        plt.figure()
        seq_lens = [r["seq_length"] for r in results["throughput"]]
        throughputs = [r["throughput_gops"] for r in results["throughput"]]
        plt.plot(seq_lens, throughputs, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Sequence Length')
        plt.ylabel('Throughput (GOPS)')
        plt.title('NVDLA Attention Throughput vs Sequence Length')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "throughput_vs_seq_length.png"), dpi=300, bbox_inches='tight')
        
        # Latency vs Sequence Length
        plt.figure()
        seq_lens = [r["seq_length"] for r in results["latency"]]
        latencies = [r["latency_us"] for r in results["latency"]]
        plt.plot(seq_lens, latencies, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Sequence Length')
        plt.ylabel('Latency (μs)')
        plt.title('NVDLA Attention Latency vs Sequence Length')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "latency_vs_seq_length.png"), dpi=300, bbox_inches='tight')
        
        # Accuracy vs Sequence Length
        plt.figure()
        seq_lens = [r["seq_length"] for r in results["accuracy"]]
        errors = [r["mean_relative_error"] for r in results["accuracy"]]
        plt.plot(seq_lens, errors, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Sequence Length')
        plt.ylabel('Mean Relative Error')
        plt.title('NVDLA Attention Accuracy vs Sequence Length')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "accuracy_vs_seq_length.png"), dpi=300, bbox_inches='tight')
    
    def _plot_head_dim_results(self, results):
        """
        Generate plots for head dimension benchmarks
        """
        # Throughput vs Head Dimension
        plt.figure()
        head_dims = [r["head_dim"] for r in results["throughput"]]
        throughputs = [r["throughput_gops"] for r in results["throughput"]]
        plt.plot(head_dims, throughputs, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Head Dimension')
        plt.ylabel('Throughput (GOPS)')
        plt.title('NVDLA Attention Throughput vs Head Dimension')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "throughput_vs_head_dim.png"), dpi=300, bbox_inches='tight')
        
        # Latency vs Head Dimension
        plt.figure()
        head_dims = [r["head_dim"] for r in results["latency"]]
        latencies = [r["latency_us"] for r in results["latency"]]
        plt.plot(head_dims, latencies, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Head Dimension')
        plt.ylabel('Latency (μs)')
        plt.title('NVDLA Attention Latency vs Head Dimension')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "latency_vs_head_dim.png"), dpi=300, bbox_inches='tight')
    
    def _plot_num_heads_results(self, results):
        """
        Generate plots for number of heads benchmarks
        """
        # Throughput vs Number of Heads
        plt.figure()
        num_heads = [r["num_heads"] for r in results["throughput"]]
        throughputs = [r["throughput_gops"] for r in results["throughput"]]
        plt.plot(num_heads, throughputs, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Number of Heads')
        plt.ylabel('Throughput (GOPS)')
        plt.title('NVDLA Attention Throughput vs Number of Heads')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "throughput_vs_num_heads.png"), dpi=300, bbox_inches='tight')
        
        # Latency vs Number of Heads
        plt.figure()
        num_heads = [r["num_heads"] for r in results["latency"]]
        latencies = [r["latency_us"] for r in results["latency"]]
        plt.plot(num_heads, latencies, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Number of Heads')
        plt.ylabel('Latency (μs)')
        plt.title('NVDLA Attention Latency vs Number of Heads')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "latency_vs_num_heads.png"), dpi=300, bbox_inches='tight')
    
    def _plot_comparison_results(self, results):
        """
        Generate plots for platform comparison
        """
        # Throughput Comparison
        plt.figure()
        platforms = ["CPU", "GPU", "NVDLA+Attention"]
        throughputs = [
            results["cpu"][0]["throughput_gops"],
            results["gpu"][0]["throughput_gops"],
            results["nvdla"][0]["throughput_gops"]
        ]
        plt.bar(platforms, throughputs)
        plt.ylabel('Throughput (GOPS)')
        plt.title('Attention Mechanism Throughput Comparison')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "throughput_comparison.png"), dpi=300, bbox_inches='tight')
        
        # Latency Comparison
        plt.figure()
        latencies = [
            results["cpu"][0]["latency_us"],
            results["gpu"][0]["latency_us"],
            results["nvdla"][0]["latency_us"]
        ]
        plt.bar(platforms, latencies)
        plt.ylabel('Latency (μs)')
        plt.title('Attention Mechanism Latency Comparison')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "latency_comparison.png"), dpi=300, bbox_inches='tight')
        
        # Power Efficiency Comparison
        plt.figure()
        power_eff = [
            results["cpu"][0]["throughput_gops"] / results["cpu"][0]["power_mw"],
            results["gpu"][0]["throughput_gops"] / results["gpu"][0]["power_mw"],
            results["nvdla"][0]["throughput_gops"] / results["nvdla"][0]["power_mw"]
        ]
        # Normalize by max value
        power_eff = [p / max(power_eff) for p in power_eff]
        plt.bar(platforms, power_eff)
        plt.ylabel('Normalized Power Efficiency (GOPS/W)')
        plt.title('Attention Mechanism Power Efficiency Comparison')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "power_efficiency_comparison.png"), dpi=300, bbox_inches='tight')
    
    def generate_summary_report(self):
        """
        Generate summary of all benchmarks
        """
        report = {
            "sequence_length": self.seq_length_results if hasattr(self, 'seq_length_results') else None,
            "head_dimension": self.head_dim_results if hasattr(self, 'head_dim_results') else None,
            "num_heads": self.num_heads_results if hasattr(self, 'num_heads_results') else None,
            "platform_comparison": self.comparison_results if hasattr(self, 'comparison_results') else None
        }
        
        # Save full report
        with open(os.path.join(self.output_dir, "benchmark_summary.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_all_benchmarks(self):
        """
        Run all benchmarks
        """
        print("Running sequence length benchmark...")
        self.benchmark_sequence_length()
        
        print("Running head dimension benchmark...")
        self.benchmark_head_dimension()
        
        print("Running number of heads benchmark...")
        self.benchmark_num_heads()
        
        print("Running platform comparison benchmark...")
        self.benchmark_comparison()
        
        print("Generating summary report...")
        self.generate_summary_report()
        
        print(f"All benchmarks complete. Results saved to {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVDLA Attention Module Benchmark")
    parser.add_argument("--output-dir", default="results", help="Directory to save benchmark results")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[16, 32, 64, 128, 256], 
                       help="Sequence lengths to benchmark")
    parser.add_argument("--head-dims", nargs="+", type=int, default=[32, 64, 128], 
                       help="Head dimensions to benchmark")
    parser.add_argument("--num-heads", nargs="+", type=int, default=[1, 2, 4, 8, 16], 
                       help="Number of heads to benchmark")
    
    args = parser.parse_args()
    
    benchmark = AttentionBenchmark(output_dir=args.output_dir)
    benchmark.seq_lengths = args.seq_lengths
    benchmark.head_dims = args.head_dims
    benchmark.num_heads = args.num_heads
    
    benchmark.run_all_benchmarks()