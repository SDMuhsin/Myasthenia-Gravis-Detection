import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import re

# --- Configuration ---
RESULTS_DIR = './results/EXP_11'
EXP_PREFIX = 'EXP_11_'

def analyze_experiment_results():
    """Analyze the results from experiment 11."""
    print("="*60)
    print("Experiment 11 Results Analysis")
    print("="*60)
    
    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory {RESULTS_DIR} not found. Run exp_11.py first.")
        return
    
    # Read the summary file
    summary_file = os.path.join(RESULTS_DIR, f'{EXP_PREFIX}summary.txt')
    if os.path.exists(summary_file):
        print("\n--- Experiment Summary ---")
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract results section
        if "Model Performance Ranking:" in content:
            results_start = content.find("Model Performance Ranking:")
            results_end = content.find("Best performing model:")
            if results_start != -1 and results_end != -1:
                results_section = content[results_start:results_end]
                print(results_section)
                
                # Extract best model info
                best_start = content.find("Best performing model:")
                if best_start != -1:
                    best_line = content[best_start:content.find("\n", best_start)]
                    print(best_line)
        else:
            print("Results not found in summary file. Experiment may still be running.")
    else:
        print(f"Summary file {summary_file} not found. Experiment may not have completed.")
    
    # List all generated files
    print("\n--- Generated Files ---")
    if os.path.exists(RESULTS_DIR):
        files = os.listdir(RESULTS_DIR)
        for file in sorted(files):
            print(f"  {file}")
    
    return

def plot_model_comparison():
    """Create a comparison plot of model performances if results are available."""
    summary_file = os.path.join(RESULTS_DIR, f'{EXP_PREFIX}summary.txt')
    
    if not os.path.exists(summary_file):
        print("Summary file not found. Cannot create comparison plot.")
        return
    
    # Try to extract results from summary file
    with open(summary_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse results
    results = {}
    if "Model Performance Ranking:" in content:
        lines = content.split('\n')
        in_results = False
        for line in lines:
            if "Model Performance Ranking:" in line:
                in_results = True
                continue
            if in_results and line.strip() and not line.startswith("Best performing"):
                if '. ' in line and ':' in line:
                    try:
                        parts = line.split('. ', 1)[1].split(': ')
                        if len(parts) == 2:
                            model_name = parts[0].strip()
                            accuracy = float(parts[1].strip())
                            results[model_name] = accuracy
                    except:
                        continue
                elif "Best performing" in line:
                    break
    
    if results:
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        models = list(results.keys())
        accuracies = list(results.values())
        
        # Sort by accuracy
        sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
        models, accuracies = zip(*sorted_data)
        
        bars = plt.barh(range(len(models)), accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add baseline line
        plt.axvline(x=0.45, color='red', linestyle='--', linewidth=2, label='Baseline (45%)')
        
        # Customize plot
        plt.yticks(range(len(models)), models)
        plt.xlabel('Accuracy')
        plt.title('Experiment 11: Model Performance Comparison')
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        
        # Add accuracy labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f'{EXP_PREFIX}model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nComparison plot saved to: {plot_path}")
        
        # Print summary statistics
        print(f"\n--- Performance Statistics ---")
        print(f"Best accuracy: {max(accuracies):.4f}")
        print(f"Worst accuracy: {min(accuracies):.4f}")
        print(f"Mean accuracy: {np.mean(accuracies):.4f}")
        print(f"Models above baseline (45%): {sum(1 for acc in accuracies if acc > 0.45)}/{len(accuracies)}")
        
    else:
        print("Could not parse results from summary file.")

def analyze_confusion_matrices():
    """Analyze confusion matrices from the experiment."""
    print("\n--- Confusion Matrix Analysis ---")
    
    # Find all confusion matrix images
    if not os.path.exists(RESULTS_DIR):
        print("Results directory not found.")
        return
    
    confusion_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('confusion_matrix.png')]
    
    if confusion_files:
        print(f"Found {len(confusion_files)} confusion matrix plots:")
        for file in sorted(confusion_files):
            model_name = file.replace(EXP_PREFIX, '').replace('_confusion_matrix.png', '')
            print(f"  - {model_name}")
    else:
        print("No confusion matrix plots found.")

def suggest_next_steps():
    """Suggest next steps based on the results."""
    print("\n--- Suggested Next Steps ---")
    
    summary_file = os.path.join(RESULTS_DIR, f'{EXP_PREFIX}summary.txt')
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "SUCCESS: Achieved accuracy above 45%" in content:
            print("🎉 Great! You've exceeded the baseline. Consider:")
            print("  1. Fine-tuning the best performing model")
            print("  2. Ensemble methods combining top models")
            print("  3. Analyzing feature importance of the best model")
            print("  4. Testing on held-out validation data")
            print("  5. Investigating class-specific performance improvements")
            
        elif "Did not exceed 45% baseline" in content:
            print("⚠️  Baseline not exceeded. Consider:")
            print("  1. Feature engineering improvements:")
            print("     - More sophisticated temporal features")
            print("     - Frequency domain features")
            print("     - Patient-specific normalization")
            print("  2. Data augmentation techniques")
            print("  3. Different neural network architectures")
            print("  4. Hyperparameter optimization")
            print("  5. Class balancing techniques")
            print("  6. Binary classification approach (HC vs Disease, then subclassification)")
            
        else:
            print("Results unclear. Check if experiment completed successfully.")
    else:
        print("No summary file found. Run exp_11.py first.")

def create_feature_importance_analysis():
    """Placeholder for feature importance analysis."""
    print("\n--- Feature Importance Analysis ---")
    print("This would require access to trained models.")
    print("Consider implementing this in the main experiment file.")

def main():
    """Main analysis function."""
    print("Starting Experiment 11 Support Analysis...")
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all analyses
    analyze_experiment_results()
    plot_model_comparison()
    analyze_confusion_matrices()
    suggest_next_steps()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
