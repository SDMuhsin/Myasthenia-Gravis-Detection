# Experiment 12: Advanced Neural Network for Myasthenia Gravis Detection

## Objective
Beat 60% accuracy using neural networks with less than 10GB VRAM for 5-way myasthenia gravis classification.

## Challenge
- Previous best: 61.92% (Gradient Boosting from Exp 11)
- Simple NN baseline: 35.50% (from Exp 11)
- **Target: >60% accuracy with <10GB VRAM**

## Strategy

### Iteration-Based Approach
1. **Quick iterations** with short epochs to test architectures rapidly
2. **Memory monitoring** to ensure <10GB VRAM usage
3. **Progressive enhancement** based on results

### Architecture Exploration
1. **Advanced LSTM-Attention Hybrid** (Iteration 1)
   - Memory-efficient multi-head attention
   - Bidirectional LSTM with residual connections
   - Global pooling strategies (avg + max)
   - Data augmentation with noise injection

2. **CNN-LSTM Hybrid** (Iteration 2)
   - 1D CNN for local pattern extraction
   - LSTM for temporal modeling
   - Batch normalization and dropout

3. **Lightweight Transformer** (Iteration 3)
   - Positional encoding
   - Multi-head self-attention
   - Layer normalization and GELU activation

4. **Ensemble Methods** (Iteration 4)
   - Multiple small networks
   - Voting/averaging strategies
   - Diverse architectures

### Memory Optimization Techniques
- **Small batch sizes** (8-16)
- **Gradient checkpointing**
- **Mixed precision training** (if needed)
- **Model pruning** and **quantization**
- **Efficient attention mechanisms**

### Data Enhancement
- **Augmentation**: Noise injection, scaling, temporal perturbations
- **Feature engineering**: Velocity, acceleration, error signals
- **Sequence preprocessing**: Normalization, outlier handling

## Architecture Details

### Advanced LSTM-Attention (v1)
```
Input (6 features) → Linear Projection (128) → 
Bidirectional LSTM (64x2) → Multi-Head Attention (4 heads) → 
Layer Norm + Residual → Feed-Forward → Layer Norm + Residual →
Global Pooling (Avg + Max) → Classification Head (128→64→5)
```

**Memory Estimate**: ~2-3GB for batch_size=16, hidden_dim=128

### CNN-LSTM Hybrid (v2)
```
Input (6 features) → 1D CNN (3,3,5 kernels, 64 channels) → 
Bidirectional LSTM (128) → Classification Head
```

**Memory Estimate**: ~1-2GB for batch_size=16

### Lightweight Transformer (v3)
```
Input → Linear Projection → Positional Encoding → 
Transformer Encoder (3 layers, 8 heads, 128 dim) → 
Global Average Pooling → Classification Head
```

**Memory Estimate**: ~3-4GB for batch_size=16

## Key Innovations

1. **Memory-Efficient Attention**: Custom implementation with reduced memory footprint
2. **Advanced Data Augmentation**: Sequence-aware noise injection and scaling
3. **Hybrid Architectures**: Combining CNN, LSTM, and Attention mechanisms
4. **Progressive Training**: Early stopping and learning rate scheduling
5. **Ensemble Strategies**: Multiple model combination for robust predictions

## Expected Outcomes

### Success Criteria
- **Primary**: >60% accuracy with <10GB VRAM
- **Secondary**: Understand which architectures work best for saccade data
- **Tertiary**: Establish memory-efficient training pipeline

### Iteration Timeline
- **Iteration 1**: Advanced LSTM-Attention baseline
- **Iteration 2**: CNN-LSTM hybrid if baseline insufficient
- **Iteration 3**: Transformer approach if needed
- **Iteration 4**: Ensemble methods for final push

## Files Structure
- `src/exp_12.py`: Main experiment with advanced architectures
- `src/support_12.py`: Additional architectures and analysis functions
- `exp_12_readme.md`: This documentation file
- `results/EXP_12/`: Results, plots, and model outputs

## Memory Management
- Monitor GPU usage with `nvidia-smi`
- Use `torch.cuda.empty_cache()` between folds
- Implement gradient accumulation if needed
- Consider mixed precision training for larger models

## Next Steps After Each Iteration
1. Analyze results and memory usage
2. Identify bottlenecks and improvement opportunities
3. Adjust architecture or hyperparameters
4. Document findings and iterate

---

**Goal**: Demonstrate that neural networks can compete with traditional ML methods for saccade-based myasthenia gravis detection while maintaining memory efficiency.
