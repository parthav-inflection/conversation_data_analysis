#!/usr/bin/env python3
"""
Streamlined script to run data quality analysis on 1 million random conversations 
from the date range 2023-07-01 to 2025-01-31.
Enhanced with stratified analysis by conversation length and time buckets.
"""

import os
import sys
import time
import json
import psutil
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_quality_analyzer_optimized import OptimizedDataQualityAnalyzer


def check_environment():
    """Check required environment variables"""
    required_vars = [
        'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 'SNOWFLAKE_ACCOUNT',
        'SNOWFLAKE_WAREHOUSE', 'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA',
        'SNOWFLAKE_CONVERSATION_TABLE'
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        return False
    return True


def categorize_conversation_length(message_count):
    """Categorize conversation by length"""
    if 1 <= message_count <= 2:
        return "one_off"  # 1-2 messages (one exchange)
    elif 3 <= message_count <= 5:
        return "short"    # 3-5 messages
    elif 6 <= message_count <= 20:
        return "medium"   # 6-20 messages
    elif message_count >= 21:
        return "long"     # 21+ messages
    else:
        return "unknown"  # 0 messages or invalid


def categorize_time_bucket(sent_at, start_date='2023-07-01'):
    """Categorize conversation by time bucket (quarterly)"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    quarter_months = [
        ('2023-Q3', '2023-07-01', '2023-09-30'),
        ('2023-Q4', '2023-10-01', '2023-12-31'),
        ('2024-Q1', '2024-01-01', '2024-03-31'),
        ('2024-Q2', '2024-04-01', '2024-06-30'),
        ('2024-Q3', '2024-07-01', '2024-09-30'),
        ('2024-Q4', '2024-10-01', '2024-12-31'),
        ('2025-Q1', '2025-01-01', '2025-01-31'),
    ]
    
    conv_date = sent_at.date() if hasattr(sent_at, 'date') else sent_at
    for quarter, start_q, end_q in quarter_months:
        start_date_q = datetime.strptime(start_q, '%Y-%m-%d').date()
        end_date_q = datetime.strptime(end_q, '%Y-%m-%d').date()
        if start_date_q <= conv_date <= end_date_q:
            return quarter
    return "unknown"


def run_stratified_analysis():
    """Run stratified quality analysis on 1M random conversations"""
    if not check_environment():
        sys.exit(1)
    
    print("="*80)
    print("COMPREHENSIVE STRATIFIED DATA QUALITY ANALYSIS")
    print("="*80)
    print(f"Date range: 2023-07-01 to 2025-01-31")
    print(f"Target: 1,000,000 conversations evenly sampled from all segments")
    print(f"Stratification: Length (one_off/short/medium/long) × Time (quarterly)")
    print(f"Sampling: ~35,714 conversations per segment (28 segments total)")
    print(f"Quality Analysis: Using OptimizedDataQualityAnalyzer (3-5x faster)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize custom analyzer for stratified analysis
    analyzer = StratifiedQualityAnalyzer(
        start_date='2023-07-01',
        end_date='2025-01-31'
    )
    
    try:
        start_time = time.time()
        
        # Run stratified analysis
        results = analyzer.analyze_stratified_sample()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save and display results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'stratified_quality_analysis_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save usable conversation IDs to a separate file for easy access
        usable_ids_file = f'usable_conversation_ids_{timestamp}.txt'
        with open(usable_ids_file, 'w') as f:
            for conv_id in results['usable_conversation_ids']:
                f.write(f"{conv_id}\n")
        
        # Save usable conversation IDs by segment to separate files
        segment_dir = f'usable_conversation_ids_by_segment_{timestamp}'
        os.makedirs(segment_dir, exist_ok=True)
        
        segment_summary = []
        for length_category, time_buckets in results['usable_conversation_ids_by_segment'].items():
            for time_bucket, conv_ids in time_buckets.items():
                if conv_ids:  # Only create files for segments with usable conversations
                    segment_file = f'{segment_dir}/{length_category}_{time_bucket}_usable_ids.txt'
                    with open(segment_file, 'w') as f:
                        for conv_id in conv_ids:
                            f.write(f"{conv_id}\n")
                    
                    segment_summary.append({
                        'length_category': length_category,
                        'time_bucket': time_bucket,
                        'usable_count': len(conv_ids),
                        'file': segment_file
                    })
        
        # Save segment summary
        segment_summary_file = f'{segment_dir}/segment_summary.json'
        with open(segment_summary_file, 'w') as f:
            json.dump(segment_summary, f, indent=2)
        
        # Create visualizations
        create_stratified_visualizations(results, f'stratified_plots_{timestamp}')
        
        # Print comprehensive results
        print_stratified_results(results, duration)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Usable conversation IDs saved to: {usable_ids_file}")
        print(f"Usable conversation IDs by segment saved to: {segment_dir}/")
        print(f"Segment summary saved to: {segment_summary_file}")
        print(f"Visualizations saved to: stratified_plots_{timestamp}/")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed - {str(e)}")
        sys.exit(1)


class StratifiedQualityAnalyzer:
    """Quality analyzer with stratification by length and time"""
    
    def __init__(self, start_date, end_date, target_total_conversations=1000000):
        self.start_date = start_date
        self.end_date = end_date
        self.target_total_conversations = target_total_conversations
        
        # Initialize the optimized quality analyzer with configurable parallel processing
        # Can be configured via environment variables
        max_workers = int(os.environ.get('QUALITY_ANALYZER_WORKERS', '2'))  # Default: 2 workers
        use_multiprocessing = os.environ.get('QUALITY_ANALYZER_MULTIPROCESSING', 'false').lower() == 'true'
        
        self.quality_analyzer = OptimizedDataQualityAnalyzer(
            batch_size=20000,  # Increased batch size for better throughput
            use_sampling=False,  # We're doing our own sampling
            sample_rate=1.0,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            use_multiprocessing=use_multiprocessing
        )
        
        # Optimize memory settings for your system (32GB RAM)
        self.quality_analyzer.memory_cleanup_threshold = 85  # Increased from 75%
        self.quality_analyzer.hash_cleanup_interval = 50000  # Increased from 25K
        
        print(f"Parallel processing: {max_workers} workers, "
              f"{'multiprocessing' if use_multiprocessing else 'threading'}")
        
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
        
        # Define all possible categories
        self.length_categories = ['one_off', 'short', 'medium', 'long']
        self.time_buckets = ['2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4', '2025-Q1']
        self.total_segments = len(self.length_categories) * len(self.time_buckets)
        self.conversations_per_segment = target_total_conversations // self.total_segments
        
        # Track usable conversation IDs for future analysis
        self.usable_conversation_ids = []
        
        # Track usable conversation IDs by segment for detailed analysis
        self.usable_conversation_ids_by_segment = defaultdict(lambda: defaultdict(list))
    
    def analyze_stratified_sample(self):
        """Analyze conversations with even sampling from each segment"""
        import snowflake.connector
        
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=os.environ.get('SNOWFLAKE_USER'),
            password=os.environ.get('SNOWFLAKE_PASSWORD'),
            account=os.environ.get('SNOWFLAKE_ACCOUNT'),
            warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE'),
            database=os.environ.get('SNOWFLAKE_DATABASE'),
            schema=os.environ.get('SNOWFLAKE_SCHEMA')
        )
        
        try:
            table_name = os.environ.get('SNOWFLAKE_CONVERSATION_TABLE')
            conversation_id_column = 'CONVERSATIONID'
            date_column = 'SENTAT'
            
            print(f"Sampling {self.conversations_per_segment:,} conversations from each of {self.total_segments} segments")
            print(f"Target total: {self.target_total_conversations:,} conversations")
            
            cursor = conn.cursor()
            all_conversation_ids = []
            
            # Create progress bar for segment sampling
            total_segments = len(self.time_buckets) * len(self.length_categories)
            with tqdm(total=total_segments, desc="Sampling segments", 
                     unit="segment", ncols=100, colour='blue') as segment_pbar:
                
                # Sample evenly from each time bucket and length category combination
                for time_bucket in self.time_buckets:
                    # Get date range for this time bucket
                    bucket_start_date, bucket_end_date = self._get_time_bucket_dates(time_bucket)
                    
                    for length_category in self.length_categories:
                        # Get message count range for this length category
                        min_messages, max_messages = self._get_length_category_range(length_category)
                        
                        segment_pbar.set_description(f"Sampling {length_category} in {time_bucket}")
                        
                        # Query to get conversation IDs for this specific segment
                        segment_query = f"""
                            WITH conversation_stats AS (
                                SELECT 
                                    {conversation_id_column},
                                    COUNT(*) as message_count,
                                    MIN({date_column}) as first_message_date,
                                    SUM(CASE WHEN TYPE = 'HUMAN_TO_AI' THEN 1 ELSE 0 END) as HUMAN_TO_AI_COUNT,
                                    SUM(CASE WHEN TYPE = 'AI_TO_HUMAN' THEN 1 ELSE 0 END) as AI_TO_HUMAN_COUNT
                                FROM {table_name}
                                WHERE {date_column} >= '{bucket_start_date}' 
                                AND {date_column} <= '{bucket_end_date}'
                                GROUP BY {conversation_id_column}
                            )
                            SELECT {conversation_id_column}
                            FROM conversation_stats
                            WHERE message_count >= {min_messages}
                            AND message_count <= {max_messages}
                            AND HUMAN_TO_AI_COUNT >= 1 
                            AND AI_TO_HUMAN_COUNT >= 1
                            ORDER BY RANDOM()
                            LIMIT {self.conversations_per_segment}
                        """
                        
                        cursor.execute(segment_query)
                        segment_conv_ids = [row[0] for row in cursor.fetchall()]
                        all_conversation_ids.extend(segment_conv_ids)
                        
                        # Update progress
                        segment_pbar.update(1)
                        segment_pbar.set_postfix({
                            'found': len(segment_conv_ids),
                            'total_collected': len(all_conversation_ids)
                        })
            
            print(f"Total conversation IDs collected: {len(all_conversation_ids):,}")
            
            # Process in batches with progress tracking  
            batch_size = 10000  # Reduced due to database performance issues
            processed = 0
            
            # Create progress bar for overall conversation processing
            with tqdm(total=len(all_conversation_ids), desc="Processing conversations", 
                     unit="conv", ncols=100, colour='green') as pbar:
                
                for i in range(0, len(all_conversation_ids), batch_size):
                    batch_ids = all_conversation_ids[i:i + batch_size]
                    conv_ids_str = "', '".join(str(cid) for cid in batch_ids)
                    
                    # Update progress bar description for data fetching
                    pbar.set_description(f"Fetching batch {i//batch_size + 1}")
                    
                    # Add timing for database queries
                    db_start = time.time()
                    
                    # Fetch messages with conversation metadata
                    messages_query = f"""
                        SELECT {conversation_id_column}, TEXT, {date_column}
                        FROM {table_name} 
                        WHERE {conversation_id_column} IN ('{conv_ids_str}')
                        ORDER BY {conversation_id_column}, {date_column}
                    """
                    
                    # Execute with retry logic for connection issues
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            cursor.execute(messages_query)
                            rows = cursor.fetchall()
                            break
                        except Exception as e:
                            if retry < max_retries - 1:
                                print(f"\n⚠️  Database error (retry {retry + 1}/{max_retries}): {e}")
                                time.sleep(5)  # Wait 5 seconds before retry
                            else:
                                raise
                    
                    db_time = time.time() - db_start
                    if db_time > 30:  # Warn if query takes >30 seconds
                        print(f"\n⚠️  Slow database query: {db_time:.1f}s for {len(batch_ids)} conversations")
                    
                    # Group by conversation
                    conversations = defaultdict(list)
                    for conv_id, text, sent_at in rows:
                        conversations[conv_id].append({
                            'text': text or '',
                            'sent_at': sent_at
                        })
                    
                    # Update progress bar description for analysis
                    pbar.set_description(f"Analyzing batch {i//batch_size + 1}")
                    
                    # Analyze each conversation with stratification
                    batch_processed = 0
                    for conv_id in batch_ids:
                        if conv_id in conversations:
                            self._analyze_conversation_stratified(conversations[conv_id], conv_id)
                        batch_processed += 1
                        processed += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Update description with processing rate every 500 conversations
                        if batch_processed % 500 == 0:
                            rate = pbar.n / (time.time() - pbar.start_t) if pbar.start_t else 0
                            pbar.set_description(f"Processing conversations ({rate:.0f} conv/s)")
                            
                            # Clear memory mid-batch if needed
                            if len(self.quality_analyzer.message_hashes) > 100000:
                                self.quality_analyzer.clear_memory()
                    
                    # Log milestone progress and manage memory
                    if processed % 20000 == 0:  # Less frequent memory management since you have plenty of RAM
                        elapsed = time.time() - pbar.start_t if pbar.start_t else 1
                        rate = processed / elapsed
                        
                        # Enhanced memory management with performance stats
                        hash_count = self.quality_analyzer.clear_memory()
                        health_check = self.quality_analyzer.check_system_health()
                        stats = health_check['stats']
                        memory_stats = stats['memory_usage']
                        cpu_stats = stats['cpu_usage']
                        
                        print(f"\n✓ Milestone: {processed:,} conversations processed ({rate:.0f} conv/s)")
                        print(f"  Memory: {memory_stats['used_gb']:.1f}GB ({memory_stats['percent_used']:.1f}%) | "
                              f"Cleared {hash_count:,} hashes")
                        print(f"  CPU: {cpu_stats['percent']:.1f}% | Workers: {stats['parallelization']['max_workers']}")
                        
                        # Show temperature if available
                        if cpu_stats['temperature_c']:
                            print(f"  CPU Temp: {cpu_stats['temperature_c']:.1f}°C")
                        
                        # Show health warnings
                        if health_check['status'] != 'good':
                            print(f"  ⚠️  System Status: {health_check['status'].upper()}")
                            for rec in health_check['recommendations']:
                                print(f"     • {rec}")
                        
                        # Force garbage collection if memory usage is high
                        if memory_stats['percent_used'] > 80:
                            import gc
                            collected = gc.collect()
                            print(f"  Triggered garbage collection: collected {collected} objects")
                        
                        pbar.set_description("Processing conversations")
            
            return self._compile_results()
            
        finally:
            conn.close()
    
    def _get_time_bucket_dates(self, time_bucket):
        """Get start and end dates for a time bucket"""
        quarter_dates = {
            '2023-Q3': ('2023-07-01', '2023-09-30'),
            '2023-Q4': ('2023-10-01', '2023-12-31'),
            '2024-Q1': ('2024-01-01', '2024-03-31'),
            '2024-Q2': ('2024-04-01', '2024-06-30'),
            '2024-Q3': ('2024-07-01', '2024-09-30'),
            '2024-Q4': ('2024-10-01', '2024-12-31'),
            '2025-Q1': ('2025-01-01', '2025-01-31'),
        }
        return quarter_dates[time_bucket]
    
    def _get_length_category_range(self, length_category):
        """Get message count range for a length category"""
        ranges = {
            'one_off': (1, 2),
            'short': (3, 5),
            'medium': (6, 20),
            'long': (21, 999999)  # Use large number for upper bound
        }
        return ranges[length_category]
    
    def _analyze_conversation_stratified(self, messages, conversation_id):
        """Analyze conversation and update stratified metrics using comprehensive analyzer"""
        # Determine length category
        message_count = len(messages)
        length_category = categorize_conversation_length(message_count)
        
        # Determine time bucket (use first message date)
        time_bucket = categorize_time_bucket(messages[0]['sent_at']) if messages else "unknown"
        
        # Use optimized quality analysis
        issues = self.quality_analyzer.analyze_conversation_optimized(messages)
        
        # Update stratified metrics
        metrics = self.stratified_metrics[length_category][time_bucket]
        metrics['total_conversations'] += 1
        
        # Map all quality issues to metrics
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
        
        # Calculate usable (no major quality issues)
        major_issues = ['empty', 'too_short', 'gibberish', 'spam', 'offensive', 'encoding_issues', 'low_quality', 'repetitive_spam']
        is_usable = not any(issues.get(issue, False) for issue in major_issues)
        if is_usable:
            metrics['usable'] += 1
            self.usable_conversation_ids.append(conversation_id)
            # Also track by segment
            self.usable_conversation_ids_by_segment[length_category][time_bucket].append(conversation_id)
    

    
    def _compile_results(self):
        """Compile stratified results"""
        results = {
            'summary': {
                'total_conversations': sum(
                    metrics['total_conversations'] 
                    for length_cat in self.stratified_metrics.values()
                    for metrics in length_cat.values()
                ),
                'analysis_timestamp': datetime.now().isoformat(),
                'date_range': f"{self.start_date} to {self.end_date}",
                'sampling_method': 'stratified_even_sampling_comprehensive',
                'quality_analyzer': 'OptimizedDataQualityAnalyzer',
                'target_conversations_per_segment': self.conversations_per_segment,
                'total_segments': self.total_segments,
                'target_total_conversations': self.target_total_conversations,
                'total_usable_conversations': len(self.usable_conversation_ids)
            },
            'stratified_metrics': {},
            'usable_conversation_ids': self.usable_conversation_ids,
            'usable_conversation_ids_by_segment': dict(self.usable_conversation_ids_by_segment)
        }
        
        # Convert to regular dict and calculate percentages
        for length_category, time_buckets in self.stratified_metrics.items():
            results['stratified_metrics'][length_category] = {}
            for time_bucket, metrics in time_buckets.items():
                if metrics['total_conversations'] > 0:
                    total = metrics['total_conversations']
                    # Calculate quality issues breakdown
                    quality_issues = {}
                    for issue_name in ['empty', 'too_short', 'too_long', 'single_turn', 'gibberish', 
                                     'duplicate', 'spam', 'offensive', 'encoding_issues', 'language_issues',
                                     'non_text_heavy', 'non_conversational', 'low_quality']:
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


def create_stratified_visualizations(results, output_dir):
    """Create visualizations for stratified analysis"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Usability heatmap by length × time
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
    
    # 2. Distribution by length category
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
    
    # Add value labels
    for bar, cat in zip(bars, length_categories):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f'{length_totals[cat]:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_stratified_results(results, duration):
    """Print comprehensive stratified results"""
    print("\n" + "="*80)
    print("STRATIFIED ANALYSIS RESULTS")
    print("="*80)
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Total conversations analyzed: {results['summary']['total_conversations']:,}")
    
    # Overall statistics by length category
    print("\n" + "="*50)
    print("QUALITY BY CONVERSATION LENGTH")
    print("="*50)
    
    length_order = ['one_off', 'short', 'medium', 'long']
    for length_cat in length_order:
        if length_cat in results['stratified_metrics']:
            print(f"\n{length_cat.replace('_', ' ').upper()} CONVERSATIONS:")
            
            # Aggregate across time buckets
            total_convs = 0
            total_usable = 0
            
            for time_bucket, metrics in results['stratified_metrics'][length_cat].items():
                total_convs += metrics['total_conversations']
                total_usable += int(metrics['total_conversations'] * metrics['usable_percentage'] / 100)
            
            if total_convs > 0:
                overall_usable_pct = (total_usable / total_convs) * 100
                print(f"  Total conversations: {total_convs:,}")
                print(f"  Overall usable percentage: {overall_usable_pct:.1f}%")
                
                # Show breakdown by quarter
                print("  Quarterly breakdown:")
                for time_bucket in ['2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4', '2025-Q1']:
                    if time_bucket in results['stratified_metrics'][length_cat]:
                        metrics = results['stratified_metrics'][length_cat][time_bucket]
                        print(f"    {time_bucket}: {metrics['total_conversations']:,} conversations, "
                              f"{metrics['usable_percentage']:.1f}% usable")
    
    # Identify best segments
    print("\n" + "="*50)
    print("HIGH-QUALITY SEGMENTS (Highest Usability)")
    print("="*50)
    
    segments = []
    for length_cat, time_buckets in results['stratified_metrics'].items():
        for time_bucket, metrics in time_buckets.items():
            if metrics['total_conversations'] >= 1000:  # Only consider segments with enough data
                segments.append({
                    'length': length_cat,
                    'time': time_bucket,
                    'total': metrics['total_conversations'],
                    'usable_pct': metrics['usable_percentage']
                })
    
    # Sort by usability percentage
    segments.sort(key=lambda x: x['usable_pct'], reverse=True)
    
    print("Top 10 highest quality segments:")
    for i, segment in enumerate(segments[:10], 1):
        print(f"{i:2d}. {segment['length'].replace('_', ' ').title()} conversations in {segment['time']}: "
              f"{segment['usable_pct']:.1f}% usable ({segment['total']:,} conversations)")
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS FOR HIGH-QUALITY SUBSET")
    print("="*50)
    
    # Find best combination for high-quality subset
    best_segments = [s for s in segments if s['usable_pct'] >= 70]  # 70%+ usable
    if best_segments:
        total_high_quality = sum(s['total'] for s in best_segments)
        print(f"Segments with ≥70% usability: {len(best_segments)} segments")
        print(f"Total conversations in high-quality segments: {total_high_quality:,}")
        print("Recommended focus areas:")
        for segment in best_segments[:5]:
            print(f"  • {segment['length'].replace('_', ' ').title()} conversations in {segment['time']}")
    else:
        print("No segments found with ≥70% usability. Consider relaxing quality thresholds.")
    
    # Show segment breakdown of usable conversation counts
    print("\n" + "="*50)
    print("USABLE CONVERSATION COUNTS BY SEGMENT")
    print("="*50)
    
    if 'usable_conversation_ids_by_segment' in results:
        segment_counts = []
        for length_cat, time_buckets in results['usable_conversation_ids_by_segment'].items():
            for time_bucket, conv_ids in time_buckets.items():
                if conv_ids:
                    segment_counts.append({
                        'length': length_cat,
                        'time': time_bucket,
                        'usable_count': len(conv_ids)
                    })
        
        # Sort by usable count descending
        segment_counts.sort(key=lambda x: x['usable_count'], reverse=True)
        
        print("Segments with usable conversations (sorted by count):")
        for i, segment in enumerate(segment_counts, 1):
            print(f"{i:2d}. {segment['length'].replace('_', ' ').title()} conversations in {segment['time']}: "
                  f"{segment['usable_count']:,} usable conversations")
        
        total_usable_by_segments = sum(s['usable_count'] for s in segment_counts)
        print(f"\nTotal usable conversations across all segments: {total_usable_by_segments:,}")
    else:
        print("No segment breakdown available.")


if __name__ == "__main__":
    run_stratified_analysis() 