#!/usr/bin/env python3
"""
Simple Conversation Length Distribution Plot

One-off script to generate a plot showing the distribution of conversation lengths
from the same Snowflake table used by the data quality analyzer.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
def get_conversation_lengths():
    """Get conversation lengths from Snowflake table"""
    print("Connecting to Snowflake...")
    
    # Use same table and columns as data quality analyzer
    table_name = os.environ.get('SNOWFLAKE_CONVERSATION_TABLE', 'CONVERSATIONS')
    conversation_id_column = os.environ.get('SNOWFLAKE_CONVERSATION_ID_COLUMN', 'CONVERSATIONID')
    date_column = os.environ.get('SNOWFLAKE_DATE_COLUMN', 'SENTAT')
    
    # Snowflake connection
    conn = snowflake.connector.connect(
        user=os.environ.get('SNOWFLAKE_USER'),
        password=os.environ.get('SNOWFLAKE_PASSWORD'),
        account=os.environ.get('SNOWFLAKE_ACCOUNT'),
        warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE'),
        database=os.environ.get('SNOWFLAKE_DATABASE'),
        schema=os.environ.get('SNOWFLAKE_SCHEMA')
    )
    
    try:
        # Query to get conversation lengths with sampling for faster results
        query = f"""
            WITH conversation_types AS (
                SELECT 
                    {conversation_id_column},
                    COUNT(*) as MESSAGE_COUNT,
                    SUM(CASE WHEN TYPE = 'HUMAN_TO_AI' THEN 1 ELSE 0 END) as HUMAN_TO_AI_COUNT,
                    SUM(CASE WHEN TYPE = 'AI_TO_HUMAN' THEN 1 ELSE 0 END) as AI_TO_HUMAN_COUNT
                FROM {table_name}
                WHERE {date_column} >= '2023-07-01' 
                AND {date_column} <= '2025-01-31'
                GROUP BY {conversation_id_column}
            )
            SELECT 
                {conversation_id_column} as CONVERSATION_ID,
                MESSAGE_COUNT
            FROM conversation_types
            WHERE HUMAN_TO_AI_COUNT >= 1 
            AND AI_TO_HUMAN_COUNT >= 1
            ORDER BY MESSAGE_COUNT DESC
        """
        
        print("Executing query...")
        df = pd.read_sql(query, conn)
        print(f"Retrieved {len(df):,} conversations")
        print(f"Columns in DataFrame: {list(df.columns)}")
        
        return df
                
    finally:
        conn.close()


def create_plot(df):
    """Create a simple histogram plot of conversation lengths"""
    message_counts = df['MESSAGE_COUNT']
    
    # Calculate basic stats
    mean_length = message_counts.mean()
    median_length = message_counts.median()
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    plt.hist(message_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_length, color='red', linestyle='--', 
               label=f'Mean: {mean_length:.1f}')
    plt.axvline(median_length, color='green', linestyle='--', 
               label=f'Median: {median_length:.1f}')
    
    plt.xlabel('Number of Messages per Conversation')
    plt.ylabel('Number of Conversations')
    plt.title('Distribution of Conversation Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('conversation_length_distribution.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'conversation_length_distribution.png'")
    
    # Show basic stats
    print(f"\nBasic Statistics:")
    print(f"Total conversations: {len(df):,}")
    print(f"Min length: {message_counts.min()} messages")
    print(f"Max length: {message_counts.max()} messages")
    print(f"Mean length: {mean_length:.2f} messages")
    print(f"Median length: {median_length:.1f} messages")
    
    # Show distribution by categories
    one_off = sum(1 for count in message_counts if count == 2)  # 1 human + 1 AI response
    short_convs = sum(1 for count in message_counts if 3 <= count <= 5)
    medium_convs = sum(1 for count in message_counts if 6 <= count <= 20)
    long_convs = sum(1 for count in message_counts if count >= 21)
    
    print(f"\nConversation Categories:")
    print(f"One-off queries (2 messages - 1 exchange pair): {one_off:,} ({one_off/len(df)*100:.1f}%)")
    print(f"Short (3-5 messages): {short_convs:,} ({short_convs/len(df)*100:.1f}%)")
    print(f"Medium (6-20 messages): {medium_convs:,} ({medium_convs/len(df)*100:.1f}%)")
    print(f"Long (21+ messages): {long_convs:,} ({long_convs/len(df)*100:.1f}%)")


if __name__ == "__main__":
    print("Analyzing conversation length distribution...")
    
    # Get data
    df = get_conversation_lengths()
    
    # Create plot
    create_plot(df)
    
    print("Analysis complete!") 