import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def load_data():
    """Load and parse the data using the same method as working experiments"""
    base_dir = "./data"
    
    # Use the same class definitions as the working experiments
    class_definitions = {
        'HC': {'path': 'Healthy control', 'label': 0},
        'MG': {'path': 'Definite MG', 'label': 1},
        'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
        'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
        'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
        'TAO': {'path': os.path.join('Non-MG diplopia (CNP, etc)', 'TAO'), 'label': 5},
    }
    
    # Use the same encoding and separator as working experiments
    csv_encoding = 'utf-16-le'
    csv_separator = ','
    required_cols = ['TargetH', 'TargetV']
    
    all_data = []
    
    for class_name, class_info in class_definitions.items():
        class_path = os.path.join(base_dir, class_info['path'])
        
        if not os.path.isdir(class_path):
            print(f"Warning: Path {class_path} does not exist")
            continue
            
        print(f"Processing {class_name} from {class_path}")
        
        # Look for patient directories (same as working experiments)
        patient_dirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        if not patient_dirs:
            print(f"INFO: No patient directories found in {class_path}")
            continue
        
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name}"):
            patient_id = patient_folder_name
            patient_dir_path = os.path.join(class_path, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            
            if not csv_files:
                continue
                
            for csv_file_path in csv_files:
                try:
                    # Load the data using same method as working experiments
                    df_full = pd.read_csv(csv_file_path, encoding=csv_encoding, sep=csv_separator)
                    original_columns = df_full.columns.tolist()
                    df_full.columns = [col.strip() for col in original_columns]
                    
                    # Check if we have the required columns
                    missing_cols = [col for col in required_cols if col not in df_full.columns]
                    if missing_cols:
                        continue
                    
                    # Extract target signals
                    target_h = df_full['TargetH'].values
                    target_v = df_full['TargetV'].values
                    
                    all_data.append({
                        'class': class_name,
                        'patient_id': patient_id,
                        'file': os.path.basename(csv_file_path),
                        'target_h': target_h,
                        'target_v': target_v,
                        'length': len(target_h)
                    })
                    
                except Exception as e:
                    print(f"Error loading {os.path.basename(csv_file_path)} (Patient: {patient_id}, Class: {class_name}): {e}")
                    continue
    
    return all_data

def analyze_target_patterns(data):
    """Analyze target signal patterns across classes"""
    
    print("\n" + "="*80)
    print("TARGET SIGNAL ANALYSIS")
    print("="*80)
    
    class_stats = defaultdict(list)
    
    for item in data:
        class_name = item['class']
        target_h = item['target_h']
        target_v = item['target_v']
        
        # Basic statistics
        h_unique = np.unique(target_h)
        v_unique = np.unique(target_v)
        
        h_range = np.max(target_h) - np.min(target_h)
        v_range = np.max(target_v) - np.min(target_v)
        
        h_std = np.std(target_h)
        v_std = np.std(target_v)
        
        # Count transitions (step changes)
        h_transitions = np.sum(np.abs(np.diff(target_h)) > 0.1)
        v_transitions = np.sum(np.abs(np.diff(target_v)) > 0.1)
        
        class_stats[class_name].append({
            'h_unique_values': h_unique,
            'v_unique_values': v_unique,
            'h_range': h_range,
            'v_range': v_range,
            'h_std': h_std,
            'v_std': v_std,
            'h_transitions': h_transitions,
            'v_transitions': v_transitions,
            'length': len(target_h)
        })
    
    # Summarize by class
    for class_name, stats_list in class_stats.items():
        print(f"\n--- CLASS: {class_name} ({len(stats_list)} files) ---")
        
        # Unique values across all files in this class
        all_h_values = set()
        all_v_values = set()
        
        h_ranges = []
        v_ranges = []
        h_stds = []
        v_stds = []
        h_trans = []
        v_trans = []
        
        for stats in stats_list:
            all_h_values.update(stats['h_unique_values'])
            all_v_values.update(stats['v_unique_values'])
            h_ranges.append(stats['h_range'])
            v_ranges.append(stats['v_range'])
            h_stds.append(stats['h_std'])
            v_stds.append(stats['v_std'])
            h_trans.append(stats['h_transitions'])
            v_trans.append(stats['v_transitions'])
        
        print(f"Horizontal Target Values: {sorted(all_h_values)}")
        print(f"Vertical Target Values: {sorted(all_v_values)}")
        print(f"H Range: {np.mean(h_ranges):.2f} ± {np.std(h_ranges):.2f}")
        print(f"V Range: {np.mean(v_ranges):.2f} ± {np.std(v_ranges):.2f}")
        print(f"H Std: {np.mean(h_stds):.2f} ± {np.std(h_stds):.2f}")
        print(f"V Std: {np.mean(v_stds):.2f} ± {np.std(v_stds):.2f}")
        print(f"H Transitions per trial: {np.mean(h_trans):.1f} ± {np.std(h_trans):.1f}")
        print(f"V Transitions per trial: {np.mean(v_trans):.1f} ± {np.std(v_trans):.1f}")

def compare_target_protocols(data):
    """Compare target protocols between classes"""
    
    print("\n" + "="*80)
    print("PROTOCOL COMPARISON")
    print("="*80)
    
    class_protocols = {}
    
    for item in data:
        class_name = item['class']
        target_h = item['target_h']
        target_v = item['target_v']
        
        # Get unique values and their frequencies
        h_unique, h_counts = np.unique(target_h, return_counts=True)
        v_unique, v_counts = np.unique(target_v, return_counts=True)
        
        # Create protocol signature
        h_signature = tuple(sorted(h_unique))
        v_signature = tuple(sorted(v_unique))
        
        if class_name not in class_protocols:
            class_protocols[class_name] = {
                'h_signatures': [],
                'v_signatures': [],
                'h_step_sizes': [],
                'v_step_sizes': []
            }
        
        class_protocols[class_name]['h_signatures'].append(h_signature)
        class_protocols[class_name]['v_signatures'].append(v_signature)
        
        # Calculate step sizes
        if len(h_unique) > 1:
            h_steps = np.diff(sorted(h_unique))
            class_protocols[class_name]['h_step_sizes'].extend(h_steps)
        
        if len(v_unique) > 1:
            v_steps = np.diff(sorted(v_unique))
            class_protocols[class_name]['v_step_sizes'].extend(v_steps)
    
    # Compare protocols
    print("\nProtocol Signatures by Class:")
    for class_name, protocols in class_protocols.items():
        print(f"\n{class_name}:")
        
        # Most common horizontal signature
        h_sigs = protocols['h_signatures']
        h_sig_counts = {}
        for sig in h_sigs:
            h_sig_counts[sig] = h_sig_counts.get(sig, 0) + 1
        
        most_common_h = max(h_sig_counts.items(), key=lambda x: x[1])
        print(f"  Most common H signature: {most_common_h[0]} ({most_common_h[1]}/{len(h_sigs)} files)")
        
        # Most common vertical signature
        v_sigs = protocols['v_signatures']
        v_sig_counts = {}
        for sig in v_sigs:
            v_sig_counts[sig] = v_sig_counts.get(sig, 0) + 1
        
        most_common_v = max(v_sig_counts.items(), key=lambda x: x[1])
        print(f"  Most common V signature: {most_common_v[0]} ({most_common_v[1]}/{len(v_sigs)} files)")
        
        # Step sizes
        if protocols['h_step_sizes']:
            h_steps = np.array(protocols['h_step_sizes'])
            print(f"  H step sizes: {np.unique(h_steps)}")
        
        if protocols['v_step_sizes']:
            v_steps = np.array(protocols['v_step_sizes'])
            print(f"  V step sizes: {np.unique(v_steps)}")

def main():
    print("Loading data...")
    data = load_data()
    
    if not data:
        print("No data loaded!")
        return
    
    print(f"Loaded {len(data)} files total")
    
    # Count by class
    class_counts = defaultdict(int)
    for item in data:
        class_counts[item['class']] += 1
    
    print("Files per class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Analyze target patterns
    analyze_target_patterns(data)
    
    # Compare protocols
    compare_target_protocols(data)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("If target signals differ systematically between classes,")
    print("this explains why TargetH_std and TargetV_std are top discriminators.")
    print("These features are essentially encoding the experimental protocol")
    print("rather than biological differences!")

if __name__ == "__main__":
    main()
