#!/usr/bin/env python3
"""
Analyze Local Conversations

This script runs the same stratified quality analysis as run_quality_analysis.py
but on locally downloaded conversation data for much faster processing.

Usage:
    # Run analysis with default settings
    python data_quality/analyze_local_conversations.py
    
    # Use custom database location
    python data_quality/analyze_local_conversations.py --database /path/to/conversations.db
    
    # Run with different worker settings  
    QUALITY_ANALYZER_WORKERS=2 python data_quality/analyze_local_conversations.py
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import psutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_quality_analyzer_optimized import OptimizedDataQualityAnalyzer


def categorize_conversation_length(message_count):
    """Categorize conversation by length (same as original)"""
    if 1 <= message_count <= 2:
        return "one_off"
    elif 3 <= message_count <= 5:
        return "short"
    elif 6 <= message_count <= 20:
        return "medium"
    elif message_count >= 21:
        return "long"
    else:
        return "unknown"


def categorize_time_bucket(sent_at, start_date='2023-07-01'):
    """Categorize conversation by time bucket (same as original)"""
    quarter_months = [
        ('2023-Q3', '2023-07-01', '2023-09-30'),
        ('2023-Q4', '2023-10-01', '2023-12-31'),
        ('2024-Q1', '2024-01-01', '2024-03-31'),
        ('2024-Q2', '2024-04-01', '2024-06-30'),
        ('2024-Q3', '2024-07-01', '2024-09-30'),
        ('2024-Q4', '2024-10-01', '2024-12-31'),
        ('2025-Q1', '2025-01-01', '2025-01-31'),
    ]
    
    try:
        if isinstance(sent_at, str):
            conv_date = datetime.strptime(sent_at[:10], '%Y-%m-%d').date()
        else:
            conv_date = sent_at.date() if hasattr(sent_at, 'date') else sent_at
            
        for quarter, start_q, end_q in quarter_months:
            start_date_q = datetime.strptime(start_q, '%Y-%m-%d').date()
            end_date_q = datetime.strptime(end_q, '%Y-%m-%d').date()
            if start_date_q <= conv_date <= end_date_q:
                return quarter
    except Exception:
        pass
    return "unknown"


class LocalConversationAnalyzer:
    """
    Analyzes locally downloaded conversations using the same logic as run_quality_analysis.py
    but reading from SQLite instead of Snowflake
    """
    
    def __init__(self, database_path='conversation_data/conversations.db'):
        self.database_path = Path(database_path)
        
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")
        
        # Initialize quality analyzer with environment settings
        max_workers = int(os.environ.get('QUALITY_ANALYZER_WORKERS', '2'))
        use_multiprocessing = os.environ.get('QUALITY_ANALYZER_MULTIPROCESSING', 'false').lower() == 'true'
        
        self.quality_analyzer = OptimizedDataQualityAnalyzer(
            batch_size=20000,
            use_sampling=False,
            sample_rate=1.0,
            max_workers=max_workers,
            use_multiprocessing=use_multiprocessing
        )
        
        print(f"Using {max_workers} workers, "
              f"{'multiprocessing' if use_multiprocessing else 'threading'}")
        
        # Initialize tracking structures (same as original)
        self.stratified_metrics = defaultdict(lambda: defaultdict(lambda: {
            'total_conversations': 0,
            'empty': 0,
            'too_short': 0,
            'too_long': 0,
            'single_turn': 0,
            'gibberish': 0,
            'duplicate': 0,
            'spam': 0,
            'offensive': 0,
            'encoding_issues': 0,
            'language_issues': 0,
            'non_text_heavy': 0,
            'non_conversational': 0,
            'low_quality': 0,
            'repetitive_spam': 0,
            'usable': 0
        }))
        
        self.usable_conversation_ids = []
        self.usable_conversation_ids_by_segment = defaultdict(lambda: defaultdict(list))
        
        self.length_categories = ['one_off', 'short', 'medium', 'long']
        self.time_buckets = ['2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4', '2025-Q1']
        
    def analyze_conversations(self):
        """Main analysis function (same logic as original run_quality_analysis.py)"""
        print("="*80)
        print("STRATIFIED QUALITY ANALYSIS (LOCAL DATA)")
        print("="*80)
        print(f"Database: {self.database_path}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        start_time = time.time()
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Check data availability
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE downloaded = TRUE")
        available_conversations = cursor.fetchone()[0]
        
        if available_conversations == 0:
            raise ValueError("No downloaded conversations found. Run download first.")
        
        print(f"Found {available_conversations:,} downloaded conversations")
        
        # Get all downloaded conversations with their messages
        print("Loading conversations...")
        cursor.execute("""
            SELECT c.conversation_id, c.message_count, c.first_message_date
            FROM conversations c 
            WHERE c.downloaded = TRUE
            ORDER BY c.conversation_id
        """)
        
        conversation_metadata = cursor.fetchall()
        print(f"Processing {len(conversation_metadata):,} conversations...")
        
        # Process in batches
        batch_size = 5000
        processed = 0
        
        with tqdm(total=len(conversation_metadata), desc="Analyzing conversations") as pbar:
            for i in range(0, len(conversation_metadata), batch_size):
                batch = conversation_metadata[i:i + batch_size]
                batch_conv_ids = [row[0] for row in batch]
                
                                 # Get messages for this batch (only need text and sent_at for analysis)
                 conv_ids_placeholders = ', '.join(['?' for _ in batch_conv_ids])
                 cursor.execute(f"""
                     SELECT conversation_id, text, sent_at 
                     FROM messages 
                     WHERE conversation_id IN ({conv_ids_placeholders})
                     ORDER BY conversation_id, sent_at
                 """, batch_conv_ids)
                
                # Group messages by conversation
                conversations = defaultdict(list)
                for conv_id, text, sent_at in cursor.fetchall():
                    conversations[conv_id].append({
                        'text': text or '',
                        'sent_at': sent_at
                    })
                
                # Analyze each conversation
                for conv_id, msg_count, first_date in batch:
                    if conv_id in conversations:
                        # Determine categories (same logic as original)
                        length_category = categorize_conversation_length(msg_count)
                        time_bucket = categorize_time_bucket(first_date)
                        
                        self._analyze_conversation_stratified(
                            conversations[conv_id], conv_id, length_category, time_bucket
                        )
                    
                    processed += 1
                    pbar.update(1)
                
                # Memory management (same as original)
                if processed % 20000 == 0:
                    health_check = self.quality_analyzer.check_system_health()
                    stats = health_check['stats']
                    memory_stats = stats['memory_usage']
                    cpu_stats = stats['cpu_usage']
                    
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    
                    print(f"\n✓ Milestone: {processed:,} conversations processed ({rate:.0f} conv/s)")
                    print(f"  Memory: {memory_stats['used_gb']:.1f}GB ({memory_stats['percent_used']:.1f}%)")
                    print(f"  CPU: {cpu_stats['percent']:.1f}% | Workers: {stats['parallelization']['max_workers']}")
                    
                    if cpu_stats['temperature_c']:
                        print(f"  CPU Temp: {cpu_stats['temperature_c']:.1f}°C")
                    
                    if health_check['status'] != 'good':
                        print(f"  ⚠️  System Status: {health_check['status'].upper()}")
                        for rec in health_check['recommendations']:
                            print(f"     • {rec}")
                    
                    hash_count = self.quality_analyzer.clear_memory()
                    if memory_stats['percent_used'] > 80:
                        import gc
                        collected = gc.collect()
                        print(f"  Triggered garbage collection: collected {collected} objects")
        
        conn.close()
        
        duration = time.time() - start_time
        return self._compile_results(duration)
        
    def _analyze_conversation_stratified(self, messages, conversation_id, length_category, time_bucket):
        """Analyze conversation and update stratified metrics (same as original)"""
        # Use optimized quality analysis
        issues = self.quality_analyzer.analyze_conversation_optimized(messages)
        
        # Update stratified metrics
        metrics = self.stratified_metrics[length_category][time_bucket]
        metrics['total_conversations'] += 1
        
        # Map all quality issues to metrics (same as original)
        issue_mapping = {
            'empty': 'empty',
            'too_short': 'too_short', 
            'too_long': 'too_long',
            'single_turn': 'single_turn',
            'gibberish': 'gibberish',
            'duplicate': 'duplicate',
            'spam': 'spam',
            'offensive': 'offensive',
            'encoding_issues': 'encoding_issues',
            'language_issues': 'language_issues',
            'non_text_heavy': 'non_text_heavy',
            'non_conversational': 'non_conversational',
            'low_quality': 'low_quality',
            'repetitive_spam': 'repetitive_spam'
        }
        
        for issue_type, metric_name in issue_mapping.items():
            if issues.get(issue_type, False):
                metrics[metric_name] += 1
        
        # Calculate usable (same as original)
        major_issues = ['empty', 'too_short', 'gibberish', 'spam', 'offensive', 'encoding_issues', 'low_quality', 'repetitive_spam']
        is_usable = not any(issues.get(issue, False) for issue in major_issues)
        if is_usable:
            metrics['usable'] += 1
            self.usable_conversation_ids.append(conversation_id)
            self.usable_conversation_ids_by_segment[length_category][time_bucket].append(conversation_id)
    
    def _compile_results(self, duration):
        """Compile results (same as original)"""
        results = {
            'summary': {
                'total_conversations': sum(
                    metrics['total_conversations'] 
                    for length_cat in self.stratified_metrics.values()
                    for metrics in length_cat.values()
                ),
                'analysis_timestamp': datetime.now().isoformat(),
                'duration_minutes': round(duration / 60, 1),
                'data_source': str(self.database_path),
                'sampling_method': 'local_database_analysis',
                'quality_analyzer': 'OptimizedDataQualityAnalyzer',
                'total_usable_conversations': len(self.usable_conversation_ids)
            },
            'stratified_metrics': {},
            'usable_conversation_ids': self.usable_conversation_ids,
            'usable_conversation_ids_by_segment': dict(self.usable_conversation_ids_by_segment)
        }
        
        # Convert to regular dict and calculate percentages (same as original)
        for length_category, time_buckets in self.stratified_metrics.items():
            results['stratified_metrics'][length_category] = {}
            for time_bucket, metrics in time_buckets.items():
                if metrics['total_conversations'] > 0:
                    total = metrics['total_conversations']
                    quality_issues = {}
                    for issue_name in ['empty', 'too_short', 'too_long', 'single_turn', 'gibberish', 
                                     'duplicate', 'spam', 'offensive', 'encoding_issues', 'language_issues',
                                     'non_text_heavy', 'non_conversational', 'low_quality', 'repetitive_spam']:
                        count = metrics.get(issue_name, 0)
                        quality_issues[issue_name] = {
                            'count': count,
                            'percentage': round((count / total) * 100, 1)
                        }
                    
                    results['stratified_metrics'][length_category][time_bucket] = {
                        'total_conversations': total,
                        'usable_percentage': round((metrics['usable'] / total) * 100, 1),
                        'quality_issues': quality_issues
                    }
        
        return results
    
    def save_results(self, results):
        """Save results (same as original)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'local_stratified_analysis_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save usable conversation IDs (same as original)
        usable_ids_file = f'local_usable_conversation_ids_{timestamp}.txt'
        with open(usable_ids_file, 'w') as f:
            for conv_id in results['usable_conversation_ids']:
                f.write(f"{conv_id}\n")
        
        # Save by segment (same as original)
        segment_dir = f'local_usable_conversation_ids_by_segment_{timestamp}'
        os.makedirs(segment_dir, exist_ok=True)
        
        segment_summary = []
        for length_cat, time_buckets in results['usable_conversation_ids_by_segment'].items():
            for time_bucket, conv_ids in time_buckets.items():
                if conv_ids:
                    segment_file = f'{segment_dir}/{length_cat}_{time_bucket}_usable_ids.txt'
                    with open(segment_file, 'w') as f:
                        for conv_id in conv_ids:
                            f.write(f"{conv_id}\n")
                    
                    segment_summary.append({
                        'length_category': length_cat,
                        'time_bucket': time_bucket,
                        'usable_count': len(conv_ids),
                        'file': segment_file
                    })
        
        with open(f'{segment_dir}/segment_summary.json', 'w') as f:
            json.dump(segment_summary, f, indent=2)
        
        # Create visualizations (same as original)
        self.create_visualizations(results, f'local_analysis_plots_{timestamp}')
        
        return results_file, usable_ids_file, segment_dir
    
    def create_visualizations(self, results, output_dir):
        """Create visualizations (same as original)"""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Usability heatmap by length × time (same as original)
        length_categories = ['one_off', 'short', 'medium', 'long']
        time_buckets = ['2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4', '2025-Q1']
        
        usability_matrix = []
        for length_cat in length_categories:
            row = []
            for time_bucket in time_buckets:
                if (length_cat in results['stratified_metrics'] and 
                    time_bucket in results['stratified_metrics'][length_cat]):
                    usability = results['stratified_metrics'][length_cat][time_bucket]['usable_percentage']
                    row.append(usability)
                else:
                    row.append(0)
            usability_matrix.append(row)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(usability_matrix, 
                    xticklabels=time_buckets,
                    yticklabels=[cat.replace('_', ' ').title() for cat in length_categories],
                    annot=True, fmt='.1f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Usable Percentage (%)'})
        plt.title('Conversation Usability by Length Category and Time Period')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/usability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution by length category (same as original)
        length_totals = {}
        for length_cat in length_categories:
            if length_cat in results['stratified_metrics']:
                total = sum(metrics['total_conversations'] 
                           for metrics in results['stratified_metrics'][length_cat].values())
                length_totals[length_cat] = total
            else:
                length_totals[length_cat] = 0
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(length_categories)), 
                       [length_totals[cat] for cat in length_categories])
        plt.xlabel('Conversation Length Category')
        plt.ylabel('Number of Conversations')
        plt.title('Distribution of Conversations by Length Category')
        plt.xticks(range(len(length_categories)), 
                   [cat.replace('_', ' ').title() for cat in length_categories])
        
        for bar, cat in zip(bars, length_categories):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                     f'{length_totals[cat]:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_results(self, results):
        """Print results (same as original)"""
        print("\n" + "="*80)
        print("LOCAL STRATIFIED ANALYSIS RESULTS")
        print("="*80)
        print(f"Duration: {results['summary']['duration_minutes']:.1f} minutes")
        print(f"Total conversations analyzed: {results['summary']['total_conversations']:,}")
        print(f"Total usable conversations: {results['summary']['total_usable_conversations']:,}")
        print(f"Data source: {results['summary']['data_source']}")
        
        # Same detailed output as original...
        print("\n" + "="*50)
        print("QUALITY BY CONVERSATION LENGTH")
        print("="*50)
        
        length_order = ['one_off', 'short', 'medium', 'long']
        for length_cat in length_order:
            if length_cat in results['stratified_metrics']:
                print(f"\n{length_cat.replace('_', ' ').upper()} CONVERSATIONS:")
                
                total_convs = 0
                total_usable = 0
                
                for time_bucket, metrics in results['stratified_metrics'][length_cat].items():
                    total_convs += metrics['total_conversations']
                    total_usable += int(metrics['total_conversations'] * metrics['usable_percentage'] / 100)
                
                if total_convs > 0:
                    overall_usable_pct = (total_usable / total_convs) * 100
                    print(f"  Total conversations: {total_convs:,}")
                    print(f"  Overall usable percentage: {overall_usable_pct:.1f}%")
                    
                    print("  Quarterly breakdown:")
                    for time_bucket in ['2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4', '2025-Q1']:
                        if time_bucket in results['stratified_metrics'][length_cat]:
                            metrics = results['stratified_metrics'][length_cat][time_bucket]
                            print(f"    {time_bucket}: {metrics['total_conversations']:,} conversations, "
                                  f"{metrics['usable_percentage']:.1f}% usable")


def main():
    parser = argparse.ArgumentParser(description='Analyze locally downloaded conversations')
    parser.add_argument('--database', default='conversation_data/conversations.db',
                       help='Path to conversations database (default: conversation_data/conversations.db)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = LocalConversationAnalyzer(args.database)
        
        # Run analysis
        results = analyzer.analyze_conversations()
        
        # Save results
        results_file, usable_ids_file, segment_dir = analyzer.save_results(results)
        
        # Print results
        analyzer.print_results(results)
        
        print(f"\nResults saved:")
        print(f"  Detailed analysis: {results_file}")
        print(f"  Usable conversation IDs: {usable_ids_file}")
        print(f"  Segmented results: {segment_dir}/")
        print(f"  Visualizations: local_analysis_plots_*/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if "not found" in str(e):
            print("Run download first: python data_quality/download_conversations.py")
        sys.exit(1)


if __name__ == "__main__":
    main() 