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
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from data_quality_analyzer import FastDataQualityAnalyzer


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
    if message_count <= 2:
        return "one_off"  # 1-2 messages (one exchange)
    elif 3 <= message_count <= 5:
        return "short"    # 3-5 messages
    elif 6 <= message_count <= 20:
        return "medium"   # 6-20 messages
    else:
        return "long"     # 21+ messages


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
    print("STRATIFIED DATA QUALITY ANALYSIS - 1 MILLION CONVERSATIONS")
    print("="*80)
    print(f"Date range: 2023-07-01 to 2025-01-31")
    print(f"Target: 1,000,000 random conversations from filtered dataset")
    print(f"Stratification: Length (short/medium/long) × Time (quarterly)")
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
        
        # Create visualizations
        create_stratified_visualizations(results, f'stratified_plots_{timestamp}')
        
        # Print comprehensive results
        print_stratified_results(results, duration)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Visualizations saved to: stratified_plots_{timestamp}/")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed - {str(e)}")
        sys.exit(1)


class StratifiedQualityAnalyzer:
    """Quality analyzer with stratification by length and time"""
    
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.stratified_metrics = defaultdict(lambda: defaultdict(lambda: {
            'total_conversations': 0,
            'empty': 0,
            'single_turn': 0,
            'gibberish': 0,
            'duplicate': 0,
            'spam': 0,
            'offensive': 0,
            'encoding_issues': 0,
            'non_conversational': 0,
            'usable': 0
        }))
        
    def analyze_stratified_sample(self):
        """Analyze 1M conversations with stratification"""
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
            
            # Get 1M random conversation IDs with same filtering as conversation_length_analysis
            conv_ids_query = f"""
                WITH conversation_types AS (
                    SELECT 
                        {conversation_id_column},
                        SUM(CASE WHEN TYPE = 'HUMAN_TO_AI' THEN 1 ELSE 0 END) as HUMAN_TO_AI_COUNT,
                        SUM(CASE WHEN TYPE = 'AI_TO_HUMAN' THEN 1 ELSE 0 END) as AI_TO_HUMAN_COUNT
                    FROM {table_name}
                    WHERE {date_column} >= '{self.start_date}' 
                    AND {date_column} <= '{self.end_date}'
                    GROUP BY {conversation_id_column}
                )
                SELECT {conversation_id_column}
                FROM conversation_types
                WHERE HUMAN_TO_AI_COUNT >= 1 
                AND AI_TO_HUMAN_COUNT >= 1
                ORDER BY RANDOM()
                LIMIT 1000000
            """
            
            cursor = conn.cursor()
            cursor.execute(conv_ids_query)
            conversation_ids = [row[0] for row in cursor.fetchall()]
            
            print(f"Retrieved {len(conversation_ids):,} conversation IDs for analysis")
            
            # Process in batches
            batch_size = 1000
            processed = 0
            
            for i in range(0, len(conversation_ids), batch_size):
                batch_ids = conversation_ids[i:i + batch_size]
                conv_ids_str = "', '".join(str(cid) for cid in batch_ids)
                
                # Fetch messages with conversation metadata
                messages_query = f"""
                    SELECT {conversation_id_column}, TEXT, {date_column}
                    FROM {table_name} 
                    WHERE {conversation_id_column} IN ('{conv_ids_str}')
                    ORDER BY {conversation_id_column}, {date_column}
                """
                
                cursor.execute(messages_query)
                rows = cursor.fetchall()
                
                # Group by conversation
                conversations = defaultdict(list)
                for conv_id, text, sent_at in rows:
                    conversations[conv_id].append({
                        'text': text or '',
                        'sent_at': sent_at
                    })
                
                # Analyze each conversation with stratification
                for conv_id in batch_ids:
                    if conv_id in conversations:
                        self._analyze_conversation_stratified(conversations[conv_id])
                    processed += 1
                
                if processed % 10000 == 0:
                    print(f"Processed {processed:,} conversations...")
            
            return self._compile_results()
            
        finally:
            conn.close()
    
    def _analyze_conversation_stratified(self, messages):
        """Analyze conversation and update stratified metrics"""
        # Determine length category
        message_count = len(messages)
        length_category = categorize_conversation_length(message_count)
        
        # Determine time bucket (use first message date)
        time_bucket = categorize_time_bucket(messages[0]['sent_at']) if messages else "unknown"
        
        # Basic quality analysis (simplified version of the full analyzer)
        issues = self._assess_quality(messages)
        
        # Update stratified metrics
        metrics = self.stratified_metrics[length_category][time_bucket]
        metrics['total_conversations'] += 1
        
        for issue_type, has_issue in issues.items():
            if has_issue:
                metrics[issue_type] += 1
        
        # Calculate usable (no major quality issues)
        major_issues = ['empty', 'gibberish', 'spam', 'offensive', 'encoding_issues']
        is_usable = not any(issues.get(issue, False) for issue in major_issues)
        if is_usable:
            metrics['usable'] += 1
    
    def _assess_quality(self, messages):
        """Simplified quality assessment"""
        if not messages:
            return {'empty': True, 'single_turn': True, 'gibberish': True, 
                   'duplicate': False, 'spam': False, 'offensive': False, 
                   'encoding_issues': False, 'non_conversational': True}
        
        texts = [msg['text'] for msg in messages]
        combined_text = ' '.join(texts).strip()
        
        # Simple quality checks
        issues = {
            'empty': len(combined_text) < 10,
            'single_turn': len(messages) == 1,
            'gibberish': self._is_gibberish_simple(texts),
            'duplicate': False,  # Skip for simplicity in stratified analysis
            'spam': any('click here' in text.lower() or 'buy now' in text.lower() 
                       or 'free' in text.lower() for text in texts),
            'offensive': any(word in text.lower() 
                           for text in texts 
                           for word in ['fuck', 'shit', 'damn', 'hate']),
            'encoding_issues': any('�' in text or '\\u' in text for text in texts),
            'non_conversational': sum(1 for text in texts 
                                    if text.strip().lower() in ['ok', 'yes', 'no', 'k']) > len(texts) * 0.5
        }
        
        return issues
    
    def _is_gibberish_simple(self, texts):
        """Simple gibberish detection"""
        gibberish_count = 0
        for text in texts:
            if len(text) < 3:
                gibberish_count += 1
                continue
            # Very long words likely gibberish
            words = text.split()
            if any(len(word) > 15 for word in words):
                gibberish_count += 1
        return gibberish_count > len(texts) * 0.5
    
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
                'date_range': f"{self.start_date} to {self.end_date}"
            },
            'stratified_metrics': {}
        }
        
        # Convert to regular dict and calculate percentages
        for length_category, time_buckets in self.stratified_metrics.items():
            results['stratified_metrics'][length_category] = {}
            for time_bucket, metrics in time_buckets.items():
                if metrics['total_conversations'] > 0:
                    total = metrics['total_conversations']
                    results['stratified_metrics'][length_category][time_bucket] = {
                        'total_conversations': total,
                        'usable_percentage': round((metrics['usable'] / total) * 100, 1),
                        'quality_issues': {
                            issue: {
                                'count': count,
                                'percentage': round((count / total) * 100, 1)
                            }
                            for issue, count in metrics.items()
                            if issue not in ['total_conversations', 'usable']
                        }
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


if __name__ == "__main__":
    run_stratified_analysis() 