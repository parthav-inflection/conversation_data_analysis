#!/usr/bin/env python3
"""
Scope and Cost Estimation Script for Filtered Dataset

Estimates token costs for the filtered conversation dataset using accurate tokenization:
- Date range: 2023-07-01 to 2025-01-31  
- Only conversations with ≥1 HUMAN_TO_AI and ≥1 AI_TO_HUMAN message
- Categorized by length: short (3-5), medium (6-20), long (21+)
- Uses HuggingFace tokenizer for accurate token counting
"""

import os
import pandas as pd
import snowflake.connector
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import gc


def get_snowflake_connection():
    """
    Establish connection to Snowflake using environment variables.
    
    Returns:
        snowflake.connector.connection: Active Snowflake connection
    """
    required_env_vars = [
        'SNOWFLAKE_USER',
        'SNOWFLAKE_PASSWORD',
        'SNOWFLAKE_ACCOUNT',
        'SNOWFLAKE_WAREHOUSE',
        'SNOWFLAKE_DATABASE',
        'SNOWFLAKE_SCHEMA',
        'SNOWFLAKE_CONVERSATION_TABLE'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )
    
    return conn


def get_all_filtered_conversations_by_category(conn, category: str) -> pd.DataFrame:
    """
    Get ALL filtered conversations for a specific category for full tokenization.
    
    Args:
        conn: Active Snowflake connection
        category: Length category ('short', 'medium', 'long')
        
    Returns:
        pd.DataFrame: DataFrame containing all conversation data for the category
    """
    table_name = os.getenv('SNOWFLAKE_CONVERSATION_TABLE')
    if not table_name:
        raise ValueError("SNOWFLAKE_CONVERSATION_TABLE environment variable is required")
    
    # Define message count ranges for categories
    category_ranges = {
        'short': (3, 5),
        'medium': (6, 20),
        'long': (21, 999)  # Upper bound for practical purposes
    }
    
    if category not in category_ranges:
        raise ValueError(f"Invalid category: {category}")
    
    min_msgs, max_msgs = category_ranges[category]
    
    # Get all conversation IDs for this category
    conv_ids_query = f"""
    WITH conversation_types AS (
        SELECT 
            CONVERSATIONID,
            COUNT(*) as MESSAGE_COUNT,
            SUM(CASE WHEN TYPE = 'HUMAN_TO_AI' THEN 1 ELSE 0 END) as HUMAN_TO_AI_COUNT,
            SUM(CASE WHEN TYPE = 'AI_TO_HUMAN' THEN 1 ELSE 0 END) as AI_TO_HUMAN_COUNT,
            DATE_TRUNC('QUARTER', MIN(SENTAT)) as time_quarter
        FROM {table_name}
        WHERE SENTAT >= '2023-07-01' 
        AND SENTAT <= '2025-01-31'
        GROUP BY CONVERSATIONID
    )
    SELECT CONVERSATIONID, time_quarter
    FROM conversation_types
    WHERE HUMAN_TO_AI_COUNT >= 1 
    AND AI_TO_HUMAN_COUNT >= 1
    AND MESSAGE_COUNT BETWEEN {min_msgs} AND {max_msgs}
    ORDER BY CONVERSATIONID
    """
    
    cursor = conn.cursor()
    try:
        cursor.execute(conv_ids_query)
        conv_results = cursor.fetchall()
        
        if not conv_results:
            return pd.DataFrame()
        
        print(f"Found {len(conv_results):,} conversations in {category} category")
        
        # Process conversations in large batches to utilize 28GB RAM
        batch_size = 5000  # Process 5K conversations at a time
        all_conversation_data = []
        
        for i in tqdm(range(0, len(conv_results), batch_size), desc=f"Processing {category} conversations"):
            batch_conv_results = conv_results[i:i + batch_size]
            conv_ids = [str(row[0]) for row in batch_conv_results]
            conv_metadata = {row[0]: row[1] for row in batch_conv_results}
            
            conv_ids_str = "', '".join(conv_ids)
            messages_query = f"""
            SELECT CONVERSATIONID, TEXT, SENTAT
            FROM {table_name}
            WHERE CONVERSATIONID IN ('{conv_ids_str}')
            ORDER BY CONVERSATIONID, SENTAT
            """
            
            cursor.execute(messages_query)
            message_results = cursor.fetchall()
            
            # Group messages by conversation
            conversations = defaultdict(list)
            for conv_id, text, sent_at in message_results:
                conversations[conv_id].append({
                    'text': text or '',
                    'sent_at': sent_at
                })
            
            # Convert to list format for this batch
            for conv_id in conv_ids:
                if conv_id in conversations:
                    messages = conversations[conv_id]
                    combined_text = ' '.join([msg['text'] for msg in messages])
                    all_conversation_data.append({
                        'conversation_id': conv_id,
                        'combined_text': combined_text,
                        'message_count': len(messages),
                        'length_category': category,
                        'time_quarter': conv_metadata[conv_id]
                    })
        
        return pd.DataFrame(all_conversation_data)
        
    finally:
        cursor.close()


def setup_tokenizer_with_acceleration(tokenizer_name: str = "gpt2"):
    """
    Setup tokenizer with MPS acceleration if available.
    
    Args:
        tokenizer_name: HuggingFace tokenizer to use
        
    Returns:
        tokenizer: Configured tokenizer
        device: Device being used
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check for MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA acceleration")
    else:
        device = "cpu"
        print("Using CPU")
    
    return tokenizer, device


def tokenize_conversations_batch(conversations_df: pd.DataFrame, tokenizer, device: str, batch_size: int = 1000) -> pd.DataFrame:
    """
    Tokenize conversations in large batches with hardware acceleration.
    
    Args:
        conversations_df: DataFrame with conversation data
        tokenizer: HuggingFace tokenizer
        device: Device to use for acceleration
        batch_size: Number of conversations to process at once
        
    Returns:
        pd.DataFrame: DataFrame with accurate token counts
    """
    print(f"Tokenizing {len(conversations_df):,} conversations with batch size {batch_size}")
    
    token_results = []
    
    for i in tqdm(range(0, len(conversations_df), batch_size), desc="Tokenizing batches"):
        batch_df = conversations_df.iloc[i:i + batch_size].copy()
        batch_texts = batch_df['combined_text'].tolist()
        
        try:
            # Tokenize batch
            batch_tokens = tokenizer(
                batch_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_tensors=None
            )
            
            # Extract token counts
            for j, tokens in enumerate(batch_tokens['input_ids']):
                row_idx = i + j
                row = batch_df.iloc[j]
                
                token_results.append({
                    'conversation_id': row['conversation_id'],
                    'length_category': row['length_category'],
                    'time_quarter': row['time_quarter'],
                    'message_count': row['message_count'],
                    'char_count': len(row['combined_text']),
                    'token_count': len(tokens)
                })
                
        except Exception as e:
            print(f"Error tokenizing batch starting at {i}: {e}")
            # Fallback for this batch
            for j in range(len(batch_df)):
                row = batch_df.iloc[j]
                token_results.append({
                    'conversation_id': row['conversation_id'],
                    'length_category': row['length_category'],
                    'time_quarter': row['time_quarter'],
                    'message_count': row['message_count'],
                    'char_count': len(row['combined_text']),
                    'token_count': len(row['combined_text']) // 4  # Fallback estimate
                })
        
        # Clean up memory periodically
        if i % (batch_size * 10) == 0:  # Every 10 batches
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
    
    return pd.DataFrame(token_results)


def calculate_category_statistics(token_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for a category.
    
    Args:
        token_df: DataFrame with token counts
        
    Returns:
        pd.DataFrame: Statistics summary
    """
    stats = token_df.groupby('length_category').agg({
        'token_count': ['count', 'mean', 'std', 'min', 'max', 'sum'],
        'char_count': ['mean', 'std'],
        'message_count': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    stats = stats.reset_index()
    
    return stats


def get_total_conversations_by_category(conn) -> pd.DataFrame:
    """
    Get total conversation counts by category for the full filtered dataset.
    
    Args:
        conn: Active Snowflake connection
        
    Returns:
        pd.DataFrame: Total counts by category
    """
    table_name = os.getenv('SNOWFLAKE_CONVERSATION_TABLE')
    
    query = f"""
    WITH conversation_types AS (
        SELECT 
            CONVERSATIONID,
            COUNT(*) as MESSAGE_COUNT,
            SUM(CASE WHEN TYPE = 'HUMAN_TO_AI' THEN 1 ELSE 0 END) as HUMAN_TO_AI_COUNT,
            SUM(CASE WHEN TYPE = 'AI_TO_HUMAN' THEN 1 ELSE 0 END) as AI_TO_HUMAN_COUNT
        FROM {table_name}
        WHERE SENTAT >= '2023-07-01' 
        AND SENTAT <= '2025-01-31'
        GROUP BY CONVERSATIONID
    ),
    filtered_conversations AS (
        SELECT 
            CASE 
                WHEN MESSAGE_COUNT BETWEEN 3 AND 5 THEN 'short'
                WHEN MESSAGE_COUNT BETWEEN 6 AND 20 THEN 'medium'
                WHEN MESSAGE_COUNT >= 21 THEN 'long'
                ELSE 'one_off'
            END as length_category
        FROM conversation_types
        WHERE HUMAN_TO_AI_COUNT >= 1 
        AND AI_TO_HUMAN_COUNT >= 1
    )
    SELECT 
        length_category,
        COUNT(*) as total_conversations
    FROM filtered_conversations
    WHERE length_category IN ('short', 'medium', 'long')
    GROUP BY length_category
    ORDER BY 
        CASE length_category 
            WHEN 'short' THEN 1 
            WHEN 'medium' THEN 2 
            WHEN 'long' THEN 3 
        END;
    """
    
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        return pd.DataFrame(results, columns=['length_category', 'total_conversations'])
    finally:
        cursor.close()


def get_filtered_temporal_statistics(conn) -> pd.DataFrame:
    """
    Get quarterly temporal statistics for the filtered dataset.
    
    Args:
        conn: Active Snowflake connection
        
    Returns:
        pd.DataFrame: DataFrame containing filtered temporal statistics
    """
    table_name = os.getenv('SNOWFLAKE_CONVERSATION_TABLE')
    if not table_name:
        raise ValueError("SNOWFLAKE_CONVERSATION_TABLE environment variable is required")
    
    query = f"""
    WITH conversation_types AS (
        SELECT 
            CONVERSATIONID,
            COUNT(*) as MESSAGE_COUNT,
            SUM(LENGTH(TEXT)) as total_chars,
            SUM(CASE WHEN TYPE = 'HUMAN_TO_AI' THEN 1 ELSE 0 END) as HUMAN_TO_AI_COUNT,
            SUM(CASE WHEN TYPE = 'AI_TO_HUMAN' THEN 1 ELSE 0 END) as AI_TO_HUMAN_COUNT,
            DATE_TRUNC('QUARTER', MIN(SENTAT)) as time_quarter
        FROM {table_name}
        WHERE SENTAT >= '2023-07-01' 
        AND SENTAT <= '2025-01-31'
        GROUP BY CONVERSATIONID
    ),
    filtered_conversations AS (
        SELECT 
            CONVERSATIONID,
            MESSAGE_COUNT,
            total_chars,
            time_quarter,
            CASE 
                WHEN MESSAGE_COUNT BETWEEN 3 AND 5 THEN 'short'
                WHEN MESSAGE_COUNT BETWEEN 6 AND 20 THEN 'medium'
                WHEN MESSAGE_COUNT >= 21 THEN 'long'
                ELSE 'one_off'
            END as length_category
        FROM conversation_types
        WHERE HUMAN_TO_AI_COUNT >= 1 
        AND AI_TO_HUMAN_COUNT >= 1
    )
    SELECT 
        time_quarter,
        length_category,
        COUNT(*) as total_conversations,
        AVG(total_chars) as avg_chars_per_convo,
        SUM(total_chars) as total_chars_in_bucket
    FROM filtered_conversations
    WHERE length_category IN ('short', 'medium', 'long')
    GROUP BY time_quarter, length_category
    ORDER BY time_quarter, 
        CASE length_category 
            WHEN 'short' THEN 1 
            WHEN 'medium' THEN 2 
            WHEN 'long' THEN 3 
        END;
    """
    
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        
        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        
        # Convert numeric columns from Decimal to float for compatibility
        numeric_columns = ['total_conversations', 'avg_chars_per_convo', 'total_chars_in_bucket']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    finally:
        cursor.close()


def calculate_costs_from_actual_tokens(all_token_stats: pd.DataFrame) -> tuple:
    """
    Calculate costs from actual token counts (not projections).
    
    Args:
        all_token_stats: Combined token statistics from all categories
        
    Returns:
        tuple: (summary_df, breakdown_df)
    """
    # Calculate total tokens by category
    category_totals = all_token_stats.groupby('length_category')['token_count_sum'].sum().reset_index()
    category_totals.columns = ['length_category', 'total_tokens']
    
    # Add conversation counts
    category_counts = all_token_stats[['length_category', 'token_count_count']].copy()
    category_counts.columns = ['length_category', 'total_conversations']
    
    # Merge
    breakdown = category_totals.merge(category_counts, on='length_category')
    
    # Calculate total tokens across all categories
    total_tokens = breakdown['total_tokens'].sum()
    total_conversations = breakdown['total_conversations'].sum()
    
    # DeepSeek Reasoner pricing with buffer
    buffer = 1.20  # 20% conservative buffer
    input_cost_per_million = 0.55 * buffer
    output_cost_per_million = 2.19 * buffer
    
    # Assume 90% input, 10% output for analysis tasks
    input_tokens = total_tokens * 0.9
    output_tokens = total_tokens * 0.1
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost
    
    # Create detailed breakdown by category
    breakdown['tokens_millions'] = breakdown['total_tokens'] / 1_000_000
    breakdown['estimated_cost_usd'] = (breakdown['total_tokens'] / 1_000_000) * (input_cost_per_million * 0.9 + output_cost_per_million * 0.1)
    breakdown['avg_tokens_per_conversation'] = breakdown['total_tokens'] / breakdown['total_conversations']
    
    # Summary
    summary = pd.DataFrame([{
        'total_conversations': total_conversations,
        'total_tokens': total_tokens,
        'total_tokens_millions': total_tokens / 1_000_000,
        'input_tokens_millions': input_tokens / 1_000_000,
        'output_tokens_millions': output_tokens / 1_000_000,
        'input_cost_usd': input_cost,
        'output_cost_usd': output_cost,
        'total_cost_usd': total_cost,
        'avg_cost_per_conversation': total_cost / total_conversations,
        'avg_tokens_per_conversation': total_tokens / total_conversations
    }])
    
    return summary, breakdown


def main():
    """
    Main execution function with full dataset tokenization using MPS acceleration.
    """
    try:
        print("=" * 80)
        print("FULL DATASET TOKENIZATION WITH MPS ACCELERATION")
        print("=" * 80)
        print("Filters: 2023-07-01 to 2025-01-31, ≥1 human-AI message pair")
        print("Categories: short (3-5), medium (6-20), long (21+ messages)")
        print("Processing: ALL conversations (not sampling)")
        print("Hardware: 28GB RAM with MPS acceleration")
        print("=" * 80)
        
        # Setup tokenizer with acceleration
        tokenizer_name = "gpt2"  # Fast and reliable
        tokenizer, device = setup_tokenizer_with_acceleration(tokenizer_name)
        
        # Connect to Snowflake
        conn = get_snowflake_connection()
        
        # Get total conversation counts first
        print("\n1. Getting overview of dataset...")
        total_counts = get_total_conversations_by_category(conn)
        print("Total conversations by category:")
        print(total_counts.to_string(index=False))
        
        # Process each category separately to manage memory
        categories = ['short', 'medium', 'long']
        all_token_stats = []
        all_detailed_results = []
        
        # Large batch size to utilize 28GB RAM efficiently
        batch_size = 2000  # Adjust based on conversation length
        
        for category in categories:
            print(f"\n2.{categories.index(category)+1} Processing {category} conversations...")
            
            # Get all conversations for this category
            category_df = get_all_filtered_conversations_by_category(conn, category)
            
            if category_df.empty:
                print(f"No {category} conversations found!")
                continue
            
            print(f"Tokenizing {len(category_df):,} {category} conversations...")
            
            # Adjust batch size based on category (longer conversations = smaller batches)
            if category == 'long':
                category_batch_size = max(500, batch_size // 4)  # Smaller batches for long conversations
            elif category == 'medium':
                category_batch_size = max(1000, batch_size // 2)
            else:
                category_batch_size = batch_size
            
            # Tokenize all conversations in this category
            token_results = tokenize_conversations_batch(
                category_df, tokenizer, device, category_batch_size
            )
            
            # Calculate statistics for this category
            category_stats = calculate_category_statistics(token_results)
            all_token_stats.append(category_stats)
            
            # Keep detailed results
            all_detailed_results.append(token_results)
            
            print(f"Completed {category} category:")
            print(f"  - Conversations: {len(token_results):,}")
            print(f"  - Total tokens: {token_results['token_count'].sum():,}")
            print(f"  - Avg tokens/conversation: {token_results['token_count'].mean():.1f}")
            
            # Save intermediate results for this category
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            token_results.to_csv(f'tokens_{category}_{timestamp}.csv', index=False)
            print(f"  - Saved: tokens_{category}_{timestamp}.csv")
            
            # Clean up memory
            del category_df, token_results
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        
        # Close Snowflake connection
        conn.close()
        
        print(f"\n3. Calculating final costs and statistics...")
        
        # Combine all statistics
        combined_stats = pd.concat(all_token_stats, ignore_index=True)
        
        # Calculate final costs
        cost_summary, cost_breakdown = calculate_costs_from_actual_tokens(combined_stats)
        
        # Display results
        print("\n" + "=" * 80)
        print("FINAL RESULTS - ACTUAL TOKEN COUNTS")
        print("=" * 80)
        print("Cost Summary:")
        print(cost_summary.to_string(index=False, float_format='%.2f'))
        
        print(f"\nCost Breakdown by Category:")
        print("-" * 60)
        print(cost_breakdown.to_string(index=False, float_format='%.2f'))
        
        print(f"\nDetailed Statistics by Category:")
        print("-" * 60)
        print(combined_stats.to_string(index=False, float_format='%.2f'))
        
        # Save final results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cost_summary.to_csv(f'final_cost_summary_{timestamp}.csv', index=False)
        cost_breakdown.to_csv(f'final_cost_breakdown_{timestamp}.csv', index=False)
        combined_stats.to_csv(f'final_token_statistics_{timestamp}.csv', index=False)
        
        # Combine and save all detailed results
        all_detailed = pd.concat(all_detailed_results, ignore_index=True)
        all_detailed.to_csv(f'all_conversation_tokens_{timestamp}.csv', index=False)
        
        print(f"\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        
        total_convs = cost_summary['total_conversations'].iloc[0]
        total_tokens = cost_summary['total_tokens'].iloc[0]
        total_cost = cost_summary['total_cost_usd'].iloc[0]
        avg_tokens = cost_summary['avg_tokens_per_conversation'].iloc[0]
        avg_cost = cost_summary['avg_cost_per_conversation'].iloc[0]
        
        print(f"Conversations processed: {total_convs:,.0f}")
        print(f"Total tokens counted: {total_tokens:,.0f} ({total_tokens/1_000_000:.1f}M)")
        print(f"Total estimated cost: ${total_cost:,.2f}")
        print(f"Average tokens per conversation: {avg_tokens:.1f}")
        print(f"Average cost per conversation: ${avg_cost:.6f}")
        print(f"Tokenizer: {tokenizer_name}")
        print(f"Device: {device}")
        print(f"Processing: Complete dataset (no sampling)")
        
        print(f"\nFiles saved:")
        print(f"- Final cost summary: final_cost_summary_{timestamp}.csv")
        print(f"- Final cost breakdown: final_cost_breakdown_{timestamp}.csv") 
        print(f"- Final statistics: final_token_statistics_{timestamp}.csv")
        print(f"- All conversation tokens: all_conversation_tokens_{timestamp}.csv")
        print(f"- Individual category files: tokens_[category]_{timestamp}.csv")
        
        print(f"\n✓ Successfully tokenized and analyzed the complete filtered dataset!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            print(f"\nSnowflake connection closed.")


if __name__ == "__main__":
    main() 