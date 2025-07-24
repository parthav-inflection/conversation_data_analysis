#!/usr/bin/env python3
"""
Conversation filter script for analyzing conversation data from Snowflake.
"""

import os
import sys
import json
import logging
import pandas as pd
import snowflake.connector
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Configuration variables loaded from environment
SNOWFLAKE_USER = os.environ.get('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.environ.get('SNOWFLAKE_PASSWORD')
SNOWFLAKE_ACCOUNT = os.environ.get('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_WAREHOUSE = os.environ.get('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.environ.get('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.environ.get('SNOWFLAKE_SCHEMA')
SNOWFLAKE_CONVERSATION_TABLE = os.environ.get('SNOWFLAKE_CONVERSATION_TABLE')
JUDGE_MODEL_API_KEY = os.environ.get('JUDGE_MODEL_API_KEY')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def sample_and_process_conversations(limit=10000):
    """
    Sample conversations from Snowflake and process them into a DataFrame.
    
    Args:
        limit (int): Maximum number of conversations to sample (default: 10,000)
        
    Returns:
        pd.DataFrame: DataFrame with conversation_id and full_conversation columns
    """
    logger.info(f"Starting sampling process for {limit} conversations...")
    
    conn = None
    try:
        # Establish connection to Snowflake
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        
        # Execute SQL query to sample entire conversations
        query = f"""
        WITH sampled_conversations AS (
            SELECT CONVERSATIONID
            FROM {SNOWFLAKE_CONVERSATION_TABLE}
            GROUP BY 1
            ORDER BY RANDOM()
            LIMIT {limit}
        )
        SELECT 
            c.CONVERSATIONID as conversation_id,
            c.TEXT as message_text,
            c.TYPE as author,
            c.SENTAT as timestamp
        FROM {SNOWFLAKE_CONVERSATION_TABLE} c
        JOIN sampled_conversations s ON c.CONVERSATIONID = s.CONVERSATIONID
        ORDER BY c.CONVERSATIONID, c.SENTAT
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Load results directly into DataFrame
        df_raw = cursor.fetch_pandas_all()
        df_raw.columns = df_raw.columns.str.lower()
        logger.info(f"Fetched {len(df_raw)} messages from Snowflake")
        
        # Process df_raw to combine messages into single document per conversation
        conversations = {}
        
        for _, row in df_raw.iterrows():
            conversation_id = row['conversation_id']
            message_text = row['message_text']
            author = row['author']
            
            if conversation_id not in conversations:
                conversations[conversation_id] = []
            
            # Limit to maximum of 15 messages per conversation
            if len(conversations[conversation_id]) < 15:
                conversations[conversation_id].append(f"{author}: {message_text}")
        
        # Create new DataFrame with combined conversations
        conversation_data = []
        for conversation_id, messages in conversations.items():
            full_conversation = "\n".join(messages)
            conversation_data.append({
                'conversation_id': conversation_id,
                'full_conversation': full_conversation
            })
        
        df_conversations = pd.DataFrame(conversation_data)
        
        unique_conversations = len(df_conversations)
        logger.info(f"Successfully processed {unique_conversations} unique conversations")
        
        return df_conversations
        
    finally:
        if conn:
            conn.close()
            logger.info("Snowflake connection closed")


def is_gibberish(text):
    """
    Helper function to detect gibberish text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        bool: True if the ratio of alphabetic characters to total characters is less than 0.7
    """
    if not text or len(text) == 0:
        return True
    
    alphabetic_count = sum(1 for char in text if char.isalpha())
    total_count = len(text)
    
    return (alphabetic_count / total_count) < 0.7


def prefilter_conversations(df):
    """
    Apply pre-filtering to remove short and gibberish conversations.
    
    Args:
        df (pd.DataFrame): DataFrame with conversation data
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    initial_count = len(df)
    logger.info(f"Initial number of conversations before filtering: {initial_count}")
    
    # Filter out conversations with less than 50 characters
    df_filtered = df[df['full_conversation'].str.len() >= 50].copy()
    short_removed = initial_count - len(df_filtered)
    logger.info(f"Removed {short_removed} conversations with less than 50 characters")
    
    # Apply gibberish filter
    df_filtered = df_filtered[~df_filtered['full_conversation'].apply(is_gibberish)].copy()
    gibberish_removed = len(df) - short_removed - len(df_filtered)
    logger.info(f"Removed {gibberish_removed} gibberish conversations")
    
    final_count = len(df_filtered)
    logger.info(f"Final number of conversations remaining after filtering: {final_count}")
    
    return df_filtered


def call_judge_model(conversation_text):
    """
    Helper function to call the judge model API with the specified prompt template.
    
    Args:
        conversation_text (str): The conversation text to evaluate
        
    Returns:
        dict: Parsed JSON response from the judge model
    """
    prompt = f"""You are an expert conversation evaluator for large language model training. Your task is to analyze conversations between users and AI assistants to determine their suitability for LLM fine-tuning.

IMPORTANT: Your response MUST be a valid JSON object. Do not include backticks, markdown code block markers, or any other text outside the JSON object.

[CONVERSATION]
{conversation_text}
[END CONVERSATION]

Evaluate this conversation and return a JSON object with these fields:

- quality (integer 1-10): The overall quality of the conversation for fine-tuning.
- topic (string): The main subject from the provided list.
- summary (string): A concise 10-20 word description.
- classification_reasoning (string): Brief reasoning for the quality and safety classifications.
- contains_pii (boolean): True if PII is present.
- contains_sensitive_content (boolean): True if PHI or other sensitive topics are present.
- is_safe (boolean): True if the conversation contains NO harmful material.
- response_accuracy (string): "high" | "medium" | "low".
- ai_assistant_is_wrong (boolean): True if the assistant is clearly incorrect.
- language (string): The primary language used."""

    if not JUDGE_MODEL_API_KEY:
        raise ValueError("JUDGE_MODEL_API_KEY not set")
    
    # Configure for OpenAI-compatible API
    headers = {
        "Authorization": f"Bearer {JUDGE_MODEL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4",  # Adjust model name as needed
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",  # Adjust endpoint as needed
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    
    # Clean the response by stripping whitespace and markdown
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    
    return json.loads(content)


def evaluate_conversations_with_llm(df, output_filepath='llm_results.jsonl'):
    """
    Evaluate conversations using LLM and save results to JSONL file.
    
    Args:
        df (pd.DataFrame): Pre-filtered DataFrame with conversation data
        output_filepath (str): Path to save JSONL results
        
    Returns:
        pd.DataFrame: DataFrame containing all evaluation results
    """
    logger.info(f"Starting LLM evaluation of {len(df)} conversations...")
    
    results = []
    
    # Open output file in append mode to save progress
    with open(output_filepath, 'a') as f:
        # Iterate through DataFrame with progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating conversations"):
            conversation_id = row['conversation_id']
            full_conversation = row['full_conversation']
            
            try:
                # Call judge model
                evaluation_result = call_judge_model(full_conversation)
                
                # Add conversation_id to the result
                evaluation_result['conversation_id'] = conversation_id
                
                # Append to results list
                results.append(evaluation_result)
                
                # Write to JSONL file for progress saving
                f.write(json.dumps(evaluation_result) + '\n')
                f.flush()
                
            except Exception as e:
                logger.error(f"Failed to evaluate conversation {conversation_id}: {str(e)}")
                continue
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    logger.info(f"Successfully evaluated {len(df_results)} conversations")
    
    return df_results


def analyze_and_save_results(df_results, output_filepath='analysis_results.json'):
    """
    Analyze the LLM evaluation results and save summary statistics.
    
    Args:
        df_results (pd.DataFrame): DataFrame with LLM evaluation results
        output_filepath (str): Path to save analysis results
        
    Returns:
        dict: Analysis summary statistics
    """
    logger.info("Starting analysis of evaluation results...")
    
    if df_results.empty:
        logger.warning("No results to analyze")
        return {}
    
    # Calculate summary statistics
    analysis = {
        'total_conversations': len(df_results),
        'quality_stats': {
            'mean': float(df_results['quality'].mean()) if 'quality' in df_results.columns else 0,
            'median': float(df_results['quality'].median()) if 'quality' in df_results.columns else 0,
            'std': float(df_results['quality'].std()) if 'quality' in df_results.columns else 0
        },
        'safety_stats': {
            'safe_conversations': int(df_results['is_safe'].sum()) if 'is_safe' in df_results.columns else 0,
            'contains_pii': int(df_results['contains_pii'].sum()) if 'contains_pii' in df_results.columns else 0,
            'contains_sensitive': int(df_results['contains_sensitive_content'].sum()) if 'contains_sensitive_content' in df_results.columns else 0
        },
        'language_distribution': df_results['language'].value_counts().to_dict() if 'language' in df_results.columns else {},
        'accuracy_distribution': df_results['response_accuracy'].value_counts().to_dict() if 'response_accuracy' in df_results.columns else {}
    }
    
    # Save analysis to file
    with open(output_filepath, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Analysis completed and saved to {output_filepath}")
    logger.info(f"Total conversations analyzed: {analysis['total_conversations']}")
    logger.info(f"Average quality score: {analysis['quality_stats']['mean']:.2f}")
    logger.info(f"Safe conversations: {analysis['safety_stats']['safe_conversations']}")
    
    return analysis


def main():
    """Main execution function."""
    logger.info("Starting conversation filter script...")
    
    # Define constants
    SAMPLE_SIZE = 10000
    LLM_RESULTS_FILE = 'llm_results.jsonl'
    ANALYSIS_RESULTS_FILE = 'analysis_results.json'
    
    try:
        # Step 1: Sample and process conversations
        logger.info("Starting conversation sampling and processing...")
        df_raw = sample_and_process_conversations(limit=SAMPLE_SIZE)
        logger.info(f"Completed sampling. Retrieved {len(df_raw)} conversations.")
        
        # Step 2: Pre-filter conversations
        logger.info("Starting pre-filtering...")
        df_filtered = prefilter_conversations(df_raw)
        logger.info(f"Completed pre-filtering. {len(df_filtered)} conversations remaining.")
        
        # Step 3: LLM evaluation
        logger.info("Starting LLM evaluation...")
        df_evaluated = evaluate_conversations_with_llm(df_filtered, LLM_RESULTS_FILE)
        logger.info(f"Completed LLM evaluation. {len(df_evaluated)} conversations evaluated.")
        
        # Step 4: Analysis and results saving
        logger.info("Starting analysis and results saving...")
        analysis_results = analyze_and_save_results(df_evaluated, ANALYSIS_RESULTS_FILE)
        logger.info("Completed analysis and results saving.")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed at step: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1) 