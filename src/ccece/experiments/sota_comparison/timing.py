"""
Timing and Computational Efficiency Module for SOTA Comparison

Measures:
- Number of parameters
- FLOPs (floating point operations)
- Training time
- Inference time
- GPU memory usage
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class ComputationalMetrics:
    """Container for computational efficiency metrics."""
    parameters: int  # Number of trainable parameters
    flops: Optional[int]  # FLOPs per inference (may be None if can't compute)
    training_time: float  # Total training time in seconds
    inference_time: float  # Per-sample inference time in milliseconds
    gpu_memory: float  # Peak GPU memory in MB

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'parameters': self.parameters,
            'parameters_k': self.parameters / 1000,
            'flops': self.flops,
            'flops_m': self.flops / 1e6 if self.flops else None,
            'training_time': self.training_time,
            'training_time_min': self.training_time / 60,
            'inference_time': self.inference_time,
            'gpu_memory': self.gpu_memory,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
) -> Optional[int]:
    """
    Measure FLOPs for a PyTorch model.

    Uses fvcore or thop if available.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        dummy_input = torch.randn(1, *input_shape).to(device)
        flops = FlopCountAnalysis(model, dummy_input)
        return int(flops.total())
    except Exception:
        pass

    try:
        from thop import profile
        model.eval()
        dummy_input = torch.randn(1, *input_shape).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return int(flops)
    except Exception:
        pass

    return None


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    batch_size: int = 1,
    warmup: int = 100,
    iterations: int = 1000,
) -> float:
    """
    Measure per-sample inference time in milliseconds.

    Args:
        model: PyTorch model
        input_shape: Input shape (seq_len, input_dim)
        device: Device to run on
        batch_size: Batch size for inference
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Per-sample inference time in milliseconds
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(batch_size, *input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Synchronize before timing (for GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed iterations
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) / batch_size)  # Per-sample time

    return np.mean(times) * 1000  # Convert to milliseconds


def measure_gpu_memory(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """
    Measure peak GPU memory usage during forward pass.

    Args:
        model: PyTorch model
        input_shape: Input shape (seq_len, input_dim)
        device: Device to run on
        batch_size: Batch size

    Returns:
        Peak GPU memory in MB (0 if CPU)
    """
    if device.type != 'cuda':
        return 0.0

    model.eval()
    model.to(device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Forward pass
    dummy_input = torch.randn(batch_size, *input_shape).to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated(device)
    return peak_memory / (1024 ** 2)  # Convert to MB


def measure_inference_time_sklearn(
    predict_fn: Callable,
    X: np.ndarray,
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """
    Measure inference time for sklearn-style models.

    Args:
        predict_fn: Prediction function (e.g., model.predict)
        X: Input data (n_samples, seq_len, n_features)
        warmup: Number of warmup calls
        iterations: Number of timed calls

    Returns:
        Per-sample inference time in milliseconds
    """
    n_samples = X.shape[0]

    # Warmup
    for _ in range(warmup):
        _ = predict_fn(X)

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = predict_fn(X)
        end = time.perf_counter()
        times.append((end - start) / n_samples)

    return np.mean(times) * 1000


def compute_computational_metrics(
    model: Any,
    input_shape: tuple,
    device: torch.device,
    training_time: float,
    is_pytorch: bool = True,
    X_sample: Optional[np.ndarray] = None,
) -> ComputationalMetrics:
    """
    Compute all computational efficiency metrics.

    Args:
        model: Model (PyTorch or sklearn-style)
        input_shape: Input shape (seq_len, input_dim)
        device: Device
        training_time: Total training time in seconds
        is_pytorch: Whether model is PyTorch-based
        X_sample: Sample data for sklearn-style inference timing

    Returns:
        ComputationalMetrics object
    """
    if is_pytorch:
        parameters = count_parameters(model)
        flops = measure_flops(model, input_shape, device)
        inference_time = measure_inference_time(model, input_shape, device)
        gpu_memory = measure_gpu_memory(model, input_shape, device)
    else:
        # For sklearn-style models like ROCKET
        parameters = model.count_parameters() if hasattr(model, 'count_parameters') else 0
        flops = None
        if X_sample is not None and hasattr(model, 'predict'):
            inference_time = measure_inference_time_sklearn(model.predict, X_sample)
        else:
            inference_time = 0.0
        gpu_memory = 0.0

    return ComputationalMetrics(
        parameters=parameters,
        flops=flops,
        training_time=training_time,
        inference_time=inference_time,
        gpu_memory=gpu_memory,
    )
