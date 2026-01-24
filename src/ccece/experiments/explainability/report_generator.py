"""
Report Generator for Explainability Analysis

Compiles all results from the 5 components into:
- full_report.json: All metrics in one file
- summary.txt: Human-readable summary
- Tables 9-11 in CSV and LaTeX formats
"""

import os
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


class ReportGenerator:
    """Generator for final explainability report and tables."""

    def __init__(
        self,
        results: Dict[str, Any],
        output_dir: str,
        tables_dir: str,
        channel_names: List[str],
        channel_categories: Dict[str, List[str]],
    ):
        """
        Args:
            results: Compiled results from all components
            output_dir: Base output directory
            tables_dir: Directory for table outputs
            channel_names: Names of channels
            channel_categories: Category to channel mapping
        """
        self.results = results
        self.output_dir = output_dir
        self.tables_dir = tables_dir
        self.channel_names = channel_names
        self.channel_categories = channel_categories

    def generate_full_report(self):
        """Generate the full JSON report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'summary': {},
        }

        # Add all component results
        for component in ['prototype_analysis', 'temporal_saliency', 'feature_importance',
                         'head_analysis', 'case_studies']:
            if component in self.results:
                report['components'][component] = self.results[component]

        # Generate summary
        report['summary'] = self._generate_summary()

        # Save
        with open(os.path.join(self.output_dir, 'full_report.json'), 'w') as f:
            json.dump(self._convert_to_serializable(report), f, indent=2)

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        return obj

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {}

        # Prototype analysis summary
        if 'prototype_analysis' in self.results:
            pa = self.results['prototype_analysis']
            summary['prototype_analysis'] = {
                'separability_ratio': pa.get('separability_ratio', 0),
                'inter_class_distance': pa.get('inter_class_distance', 0),
                'intra_class_distance': pa.get('intra_class_distance', 0),
                'conclusion': 'Prototypes are well-separated' if pa.get('separability_ratio', 0) > 1.5 else 'Prototypes need improvement',
            }

        # Temporal saliency summary
        if 'temporal_saliency' in self.results:
            ts = self.results['temporal_saliency']
            summary['temporal_saliency'] = {
                'mg_peak_location': ts.get('mg', {}).get('peak_location_mean', 0.5),
                'hc_peak_location': ts.get('hc', {}).get('peak_location_mean', 0.5),
                'early_late_significant': ts.get('statistical_tests', {}).get('early_late_ratio', {}).get('significant', False),
                'conclusion': 'MG shows later attention peaks (fatigability pattern)' if ts.get('mg', {}).get('peak_location_mean', 0.5) > ts.get('hc', {}).get('peak_location_mean', 0.5) else 'No clear fatigability pattern',
            }

        # Feature importance summary
        if 'feature_importance' in self.results:
            fi = self.results['feature_importance']
            ranking = fi.get('ranking', [])
            top3 = [r['channel'] for r in ranking[:3]]
            summary['feature_importance'] = {
                'top_3_features': top3,
                'velocity_most_important': any('Velocity' in ch for ch in top3),
                'category_ranking': fi.get('category_importance', {}),
                'conclusion': 'Velocity channels are most important (clinically expected)' if any('Velocity' in ch for ch in top3) else 'Unexpected feature ranking',
            }

        # Head analysis summary
        if 'head_analysis' in self.results:
            ha = self.results['head_analysis']
            summary['head_analysis'] = {
                'mean_agreement': ha.get('mean_pairwise_agreement', 0),
                'disagreement_rate': ha.get('disagreement_rate', 0),
                'conclusion': 'Heads show moderate specialization' if ha.get('disagreement_rate', 0) > 0.1 else 'Heads agree on most predictions',
            }

        # Case studies summary
        if 'case_studies' in self.results:
            cs = self.results['case_studies']
            summary['case_studies'] = {
                'total_samples': cs.get('total_samples', 0),
                'n_correct': cs.get('n_correct', 0),
                'n_incorrect': cs.get('n_incorrect', 0),
                'accuracy': cs.get('n_correct', 0) / max(cs.get('total_samples', 1), 1),
            }

        return summary

    def generate_summary_text(self):
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("EXPLAINABILITY ANALYSIS SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Prototype Analysis
        if 'prototype_analysis' in self.results:
            pa = self.results['prototype_analysis']
            lines.append("-" * 50)
            lines.append("COMPONENT 1: PROTOTYPE ANALYSIS")
            lines.append("-" * 50)
            lines.append(f"  Separability ratio: {pa.get('separability_ratio', 0):.3f}")
            lines.append(f"  Inter-class distance: {pa.get('inter_class_distance', 0):.3f}")
            lines.append(f"  Intra-class distance: {pa.get('intra_class_distance', 0):.3f}")

            per_head = pa.get('per_head_quality', {})
            if per_head:
                lines.append("  Per-head alignment:")
                for h in range(5):
                    if f'head_{h}' in per_head:
                        align = per_head[f'head_{h}']['alignment']
                        lines.append(f"    Head {h}: {align:.1%}")
            lines.append("")

        # Temporal Saliency
        if 'temporal_saliency' in self.results:
            ts = self.results['temporal_saliency']
            lines.append("-" * 50)
            lines.append("COMPONENT 2: TEMPORAL SALIENCY")
            lines.append("-" * 50)
            mg = ts.get('mg', {})
            hc = ts.get('hc', {})
            lines.append(f"  MG peak location: {mg.get('peak_location_mean', 0):.3f} +/- {mg.get('peak_location_std', 0):.3f}")
            lines.append(f"  HC peak location: {hc.get('peak_location_mean', 0):.3f} +/- {hc.get('peak_location_std', 0):.3f}")
            lines.append(f"  MG early/late ratio: {mg.get('early_vs_late_ratio', 0):.3f}")
            lines.append(f"  HC early/late ratio: {hc.get('early_vs_late_ratio', 0):.3f}")

            tests = ts.get('statistical_tests', {})
            for test_name, result in tests.items():
                sig = "***" if result.get('significant', False) else ""
                lines.append(f"  {test_name}: p={result.get('p_value', 1):.4f} {sig}")
            lines.append("")

        # Feature Importance
        if 'feature_importance' in self.results:
            fi = self.results['feature_importance']
            lines.append("-" * 50)
            lines.append("COMPONENT 3: FEATURE IMPORTANCE")
            lines.append("-" * 50)
            ranking = fi.get('ranking', [])
            lines.append("  Top 5 features:")
            for i, r in enumerate(ranking[:5]):
                lines.append(f"    {i+1}. {r['channel']}: {r['mean']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f})")

            cat_imp = fi.get('category_importance', {})
            lines.append("  Category importance:")
            for cat, imp in sorted(cat_imp.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {cat.capitalize()}: {imp:.4f}")

            vel_pos = fi.get('velocity_vs_position', {})
            lines.append(f"  Velocity vs Position: p={vel_pos.get('p_value', 1):.4f}")
            lines.append("")

        # Head Analysis
        if 'head_analysis' in self.results:
            ha = self.results['head_analysis']
            lines.append("-" * 50)
            lines.append("COMPONENT 4: HEAD ANALYSIS")
            lines.append("-" * 50)
            lines.append(f"  Mean pairwise agreement: {ha.get('mean_pairwise_agreement', 0):.3f}")
            lines.append(f"  Disagreement rate: {ha.get('disagreement_rate', 0):.3f}")

            perf = ha.get('head_performance', {})
            lines.append("  Per-head accuracy:")
            for h in range(5):
                if f'head_{h}' in perf:
                    acc = perf[f'head_{h}']['accuracy']
                    conf = perf[f'head_{h}']['confidence_mean']
                    lines.append(f"    Head {h}: {acc:.1%} (conf: {conf:.2f})")
            lines.append("")

        # Case Studies
        if 'case_studies' in self.results:
            cs = self.results['case_studies']
            lines.append("-" * 50)
            lines.append("COMPONENT 5: CASE STUDIES")
            lines.append("-" * 50)
            lines.append(f"  Total samples: {cs.get('total_samples', 0)}")
            lines.append(f"  Correct: {cs.get('n_correct', 0)}")
            lines.append(f"  Incorrect: {cs.get('n_incorrect', 0)}")

            # Category summaries
            cat_comp = cs.get('category_comparison', {})
            if 'confidence' in cat_comp:
                lines.append("  Confidence by category:")
                for cat, stats in cat_comp['confidence'].items():
                    lines.append(f"    {cat}: {stats['mean']:.3f} +/- {stats['std']:.3f}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("END OF SUMMARY")
        lines.append("=" * 70)

        # Write to file
        with open(os.path.join(self.output_dir, 'summary.txt'), 'w') as f:
            f.write('\n'.join(lines))

    def generate_table9_feature_importance(self):
        """Generate Table 9: Feature Importance."""
        if 'feature_importance' not in self.results:
            return

        fi = self.results['feature_importance']
        ranking = fi.get('ranking', [])

        # CSV
        csv_lines = ['Rank,Channel,Category,Importance,CI_Low,CI_High']
        for i, r in enumerate(ranking):
            cat = 'Unknown'
            for c, channels in self.channel_categories.items():
                if r['channel'] in channels:
                    cat = c.capitalize()
                    break
            csv_lines.append(f"{i+1},{r['channel']},{cat},{r['mean']:.4f},{r['ci_low']:.4f},{r['ci_high']:.4f}")

        with open(os.path.join(self.tables_dir, 'table9_feature_importance.csv'), 'w') as f:
            f.write('\n'.join(csv_lines))

        # LaTeX
        tex_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Feature Importance (Permutation)}',
            r'\label{tab:feature_importance}',
            r'\begin{tabular}{clccc}',
            r'\toprule',
            r'Rank & Channel & Category & Importance & 95\% CI \\',
            r'\midrule',
        ]

        for i, r in enumerate(ranking[:10]):  # Top 10
            cat = 'Unknown'
            for c, channels in self.channel_categories.items():
                if r['channel'] in channels:
                    cat = c.capitalize()
                    break
            channel_escaped = r['channel'].replace('_', r'\_')
            tex_lines.append(
                f"{i+1} & {channel_escaped} & {cat} & "
                f"{r['mean']:.3f} & [{r['ci_low']:.3f}, {r['ci_high']:.3f}] \\\\"
            )

        tex_lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ])

        with open(os.path.join(self.tables_dir, 'table9_feature_importance.tex'), 'w') as f:
            f.write('\n'.join(tex_lines))

    def generate_table10_prototype_analysis(self):
        """Generate Table 10: Prototype Analysis."""
        if 'prototype_analysis' not in self.results:
            return

        pa = self.results['prototype_analysis']
        per_head = pa.get('per_head_quality', {})

        # CSV
        csv_lines = ['Head,Alignment,HC_Alignment,MG_Alignment,Separability']
        for h in range(5):
            if f'head_{h}' in per_head:
                hq = per_head[f'head_{h}']
                csv_lines.append(
                    f"{h},{hq['alignment']:.4f},{hq['hc_alignment']:.4f},"
                    f"{hq['mg_alignment']:.4f},{hq['separability']:.4f}"
                )

        csv_lines.append(f"Overall,{pa.get('separability_ratio', 0):.4f},"
                        f"{pa.get('inter_class_distance', 0):.4f},"
                        f"{pa.get('intra_class_distance', 0):.4f},")

        with open(os.path.join(self.tables_dir, 'table10_prototype_analysis.csv'), 'w') as f:
            f.write('\n'.join(csv_lines))

        # LaTeX
        tex_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Prototype Analysis by Head}',
            r'\label{tab:prototype_analysis}',
            r'\begin{tabular}{lcccc}',
            r'\toprule',
            r'Head & Alignment & HC Align. & MG Align. & Separation \\',
            r'\midrule',
        ]

        for h in range(5):
            if f'head_{h}' in per_head:
                hq = per_head[f'head_{h}']
                tex_lines.append(
                    f"Head {h} & {hq['alignment']:.1%} & {hq['hc_alignment']:.1%} & "
                    f"{hq['mg_alignment']:.1%} & {hq['separability']:.2f} \\\\"
                )

        tex_lines.extend([
            r'\midrule',
            f"Overall Separability Ratio & \\multicolumn{{4}}{{c}}{{{pa.get('separability_ratio', 0):.2f}}} \\\\",
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ])

        with open(os.path.join(self.tables_dir, 'table10_prototype_analysis.tex'), 'w') as f:
            f.write('\n'.join(tex_lines))

    def generate_table11_head_analysis(self):
        """Generate Table 11: Head Analysis."""
        if 'head_analysis' not in self.results:
            return

        ha = self.results['head_analysis']
        perf = ha.get('head_performance', {})

        # CSV
        csv_lines = ['Head,Accuracy,Confidence,Correct_Conf,Wrong_Conf']
        for h in range(5):
            if f'head_{h}' in perf:
                hp = perf[f'head_{h}']
                csv_lines.append(
                    f"{h},{hp['accuracy']:.4f},{hp['confidence_mean']:.4f},"
                    f"{hp['correct_confidence']:.4f},{hp['wrong_confidence']:.4f}"
                )

        with open(os.path.join(self.tables_dir, 'table11_head_analysis.csv'), 'w') as f:
            f.write('\n'.join(csv_lines))

        # LaTeX
        tex_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Head Performance Analysis}',
            r'\label{tab:head_analysis}',
            r'\begin{tabular}{lcccc}',
            r'\toprule',
            r'Head & Accuracy & Mean Conf. & Correct Conf. & Wrong Conf. \\',
            r'\midrule',
        ]

        for h in range(5):
            if f'head_{h}' in perf:
                hp = perf[f'head_{h}']
                tex_lines.append(
                    f"Head {h} & {hp['accuracy']:.1%} & {hp['confidence_mean']:.2f} & "
                    f"{hp['correct_confidence']:.2f} & {hp['wrong_confidence']:.2f} \\\\"
                )

        tex_lines.extend([
            r'\midrule',
            f"Agreement Rate & \\multicolumn{{4}}{{c}}{{{ha.get('mean_pairwise_agreement', 0):.1%}}} \\\\",
            f"Disagreement Rate & \\multicolumn{{4}}{{c}}{{{ha.get('disagreement_rate', 0):.1%}}} \\\\",
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ])

        with open(os.path.join(self.tables_dir, 'table11_head_analysis.tex'), 'w') as f:
            f.write('\n'.join(tex_lines))

    def generate_all(self):
        """Generate all reports and tables."""
        print("  Generating full report...")
        self.generate_full_report()

        print("  Generating summary text...")
        self.generate_summary_text()

        print("  Generating Table 9 (Feature Importance)...")
        self.generate_table9_feature_importance()

        print("  Generating Table 10 (Prototype Analysis)...")
        self.generate_table10_prototype_analysis()

        print("  Generating Table 11 (Head Analysis)...")
        self.generate_table11_head_analysis()

        print("  Report generation complete!")
