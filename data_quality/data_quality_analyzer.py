#!/usr/bin/env python3
"""
Data Quality Analyzer for Large-Scale Conversation Data

This script analyzes conversation data quality using conversation-based analysis
suitable for processing large-scale conversation datasets from Snowflake. It identifies:
1. Empty or incomplete conversations
2. Single-turn conversations (not truly conversational)
3. Conversations with predominantly gibberish content
4. Duplicate conversations
5. Conversations containing vulgar or offensive content
6. Spam-like conversations
7. Non-conversational exchanges
8. Conversations with encoding/language issues
9. Low-quality conversations based on multiple criteria

The script processes complete conversations rather than individual messages,
enabling context-aware quality assessment and more accurate filtering decisions.
"""

import os
import sys
import re
import json
import logging
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import snowflake.connector
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Fast text processing libraries
import regex  # Faster than re for complex patterns
from unicodedata import category, normalize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_quality_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Container for data quality metrics - now tracking conversations"""
    total_conversations: int = 0
    total_messages: int = 0  # Keep for detailed stats
    empty_conversations: int = 0
    too_short_conversations: int = 0  # Conversations with too few messages
    too_long_conversations: int = 0   # Conversations with too many messages
    duplicate_conversations: int = 0
    gibberish_conversations: int = 0  # Conversations with mostly gibberish
    non_text_heavy_conversations: int = 0
    offensive_conversations: int = 0  # Conversations containing offensive content
    spam_conversations: int = 0
    non_conversational_conversations: int = 0  # Conversations that aren't conversational
    encoding_issues_conversations: int = 0
    language_issues_conversations: int = 0
    single_turn_conversations: int = 0  # Conversations with only one message
    low_quality_conversations: int = 0  # Conversations with poor overall quality
    
    def usable_percentage(self) -> float:
        """Calculate percentage of usable conversations"""
        unusable = (self.empty_conversations + self.too_short_conversations + 
                   self.duplicate_conversations + self.gibberish_conversations + 
                   self.non_text_heavy_conversations + self.offensive_conversations + 
                   self.spam_conversations + self.non_conversational_conversations + 
                   self.encoding_issues_conversations + self.language_issues_conversations +
                   self.single_turn_conversations + self.low_quality_conversations)
        return ((self.total_conversations - unusable) / self.total_conversations) * 100 if self.total_conversations > 0 else 0


class FastDataQualityAnalyzer:
    """
    High-performance conversation-based data quality analyzer
    
    This analyzer processes complete conversations rather than individual messages,
    enabling context-aware quality assessment. It groups messages by conversation ID
    and applies quality checks at the conversation level for more meaningful results.
    """
    
    def __init__(self, batch_size: int = 10000, use_sampling: bool = True, sample_rate: float = 0.001,
                 start_date: str = '2023-07-01', end_date: str = '2025-01-31'):
        self.batch_size = batch_size
        self.use_sampling = use_sampling
        self.sample_rate = sample_rate
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize metrics
        self.metrics = DataQualityMetrics()
        self.detailed_stats = defaultdict(Counter)
        
        # Load profanity list (lightweight approach)
        self.profanity_patterns = self._load_profanity_patterns()
        
        # Precompile regex patterns for speed
        self._compile_patterns()
        
        # Hash set for duplicate detection (memory efficient for sampling)
        self.message_hashes: Set[str] = set()
        
    def _load_profanity_patterns(self) -> List[regex.Pattern]:
        """Load basic profanity patterns (lightweight approach)"""
        # Basic profanity patterns - you should extend this with a proper list
        basic_patterns = [
            r'\b(fuck|shit|damn|bitch|ass|crap|hell)\b',
            r'\b(stupid|idiot|moron|retard)\b',
            r'\b(hate|kill|die|death)\b',
            r'\b(nazi|hitler|terrorist)\b'
        ]
        return [regex.compile(pattern, regex.IGNORECASE) for pattern in basic_patterns]
    
    def _compile_patterns(self):
        """Precompile regex patterns for performance"""
        # Gibberish detection patterns
        self.gibberish_patterns = [
            regex.compile(r'[a-zA-Z]{15,}'),  # Very long sequences of letters
            regex.compile(r'(.)\1{4,}'),      # Repeated characters
            regex.compile(r'[0-9]{10,}'),     # Long number sequences
            regex.compile(r'[^a-zA-Z0-9\s]{5,}'),  # Long special character sequences
        ]
        
        # Non-conversational patterns
        self.non_conversational_patterns = [
            regex.compile(r'^(ok|okay|yes|no|k|lol|haha|hmm)$', regex.IGNORECASE),
            regex.compile(r'^[.!?]+$'),
            regex.compile(r'^[\d\s\-\(\)]+$'),  # Only numbers, spaces, dashes, parentheses
            regex.compile(r'^[^\w\s]+$'),       # Only special characters
        ]
        
        # Spam patterns
        self.spam_patterns = [
            regex.compile(r'(click here|visit|buy now|free|offer|deal)', regex.IGNORECASE),
            regex.compile(r'(http|www\.|\.com|\.org)', regex.IGNORECASE),
            regex.compile(r'(\$\d+|\d+%|discount|sale)', regex.IGNORECASE),
        ]
        
        # Encoding issue patterns
        self.encoding_patterns = [
            regex.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'),  # Control characters
            regex.compile(r'[��]+'),  # Common encoding error characters
            regex.compile(r'\\u[0-9a-fA-F]{4}'),  # Unescaped unicode
        ]
    
    def _fast_hash(self, text: str) -> str:
        """Fast hash function for duplicate detection"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    def _is_empty_or_whitespace(self, text: str) -> bool:
        """Check if message is empty or only whitespace"""
        return not text or not text.strip()
    
    def _is_too_short(self, text: str, min_length: int = 3) -> bool:
        """Check if message is too short to be meaningful"""
        return len(text.strip()) < min_length
    
    def _is_too_long(self, text: str, max_length: int = 10000) -> bool:
        """Check if message is suspiciously long"""
        return len(text) > max_length
    
    def _is_gibberish(self, text: str) -> bool:
        """
        Fast gibberish detection using multiple heuristics
        """
        if not text or len(text) < 3:
            return True
            
        # Check character distribution
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.3:  # Less than 30% alphabetic characters
            return True
        
        # Check for very long words (likely gibberish)
        words = text.split()
        if any(len(word) > 20 for word in words):
            return True
        
        # Check regex patterns
        for pattern in self.gibberish_patterns:
            if pattern.search(text):
                return True
        
        # Check vowel-consonant ratio (basic linguistic heuristic)
        vowels = sum(1 for c in text.lower() if c in 'aeiou')
        consonants = sum(1 for c in text.lower() if c.isalpha() and c not in 'aeiou')
        if consonants > 0 and vowels / consonants < 0.1:  # Very few vowels
            return True
        
        return False
    
    def _has_encoding_issues(self, text: str) -> bool:
        """Detect encoding/character issues"""
        try:
            # Check for encoding patterns
            for pattern in self.encoding_patterns:
                if pattern.search(text):
                    return True
            
            # Check for mixed scripts (potential encoding issues)
            scripts = set()
            for char in text:
                if char.isalpha():
                    scripts.add(category(char)[0])
            
            # If more than 2 different script categories, might be encoding issue
            return len(scripts) > 2
            
        except Exception:
            return True  # If we can't process it, consider it problematic
    
    def _is_non_text_heavy(self, text: str) -> bool:
        """Check if message is heavily non-text (numbers, symbols, etc.)"""
        if not text:
            return True
        
        text_chars = sum(1 for c in text if c.isalpha() or c.isspace())
        return (text_chars / len(text)) < 0.5
    
    def _contains_profanity(self, text: str) -> bool:
        """Fast profanity detection using precompiled patterns"""
        for pattern in self.profanity_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _is_spam_like(self, text: str) -> bool:
        """Detect spam-like content"""
        spam_indicators = 0
        for pattern in self.spam_patterns:
            if pattern.search(text):
                spam_indicators += 1
        
        # Consider spam if multiple indicators or very promotional
        return spam_indicators >= 2 or 'free' in text.lower() and 'click' in text.lower()
    
    def _is_non_conversational(self, text: str) -> bool:
        """Detect non-conversational content"""
        text_clean = text.strip().lower()
        
        # Check against non-conversational patterns
        for pattern in self.non_conversational_patterns:
            if pattern.match(text_clean):
                return True
        
        # Single character responses
        if len(text_clean) == 1:
            return True
        
        # Only punctuation
        if all(not c.isalnum() for c in text_clean):
            return True
        
        return False
    
    def _detect_language_issues(self, text: str) -> bool:
        """Detect language-related issues using simple ASCII-based approach"""
        try:
            if not text:
                return True
            
            # Simple ASCII-based detection - most English text should be primarily ASCII
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
            return ascii_ratio < 0.8  # Less than 80% ASCII characters likely indicates non-English
                
        except Exception:
            return True  # If detection fails, consider it problematic
    
    def _is_duplicate(self, text: str) -> bool:
        """Fast duplicate detection using hashing"""
        text_hash = self._fast_hash(text.strip().lower())
        if text_hash in self.message_hashes:
            return True
        self.message_hashes.add(text_hash)
        return False
    
    def analyze_conversation(self, conversation: List[Dict]) -> Dict[str, bool]:
        """
        Analyze a complete conversation for various quality issues
        
        Args:
            conversation: List of message dictionaries with 'text' and other fields
            
        Returns:
            Dictionary with boolean flags for each conversation-level quality issue
        """
        if not conversation:
            return {
                'empty': True,
                'too_short': True,
                'too_long': False,
                'duplicate': False,
                'gibberish': True,
                'non_text_heavy': True,
                'offensive': False,
                'spam': False,
                'non_conversational': True,
                'encoding_issues': False,
                'language_issues': False,
                'single_turn': True,
                'low_quality': True
            }
        
        # Extract texts from conversation
        texts = [msg.get('text', '') or '' for msg in conversation]
        conversation_text = ' '.join(texts).strip()
        
        # Basic conversation metrics
        num_messages = len(conversation)
        total_length = len(conversation_text)
        
        # Check for single turn conversations
        is_single_turn = num_messages == 1
        
        # Check if conversation is too short (less than 2 meaningful messages)
        meaningful_messages = sum(1 for text in texts if len(text.strip()) >= 3)
        is_too_short = meaningful_messages < 2
        
        # Check if conversation is too long (more than 50 messages)
        is_too_long = num_messages > 50
        
        # Check if conversation is empty or mostly empty
        is_empty = total_length < 10 or all(len(text.strip()) < 3 for text in texts)
        
        # Check for duplicate conversation (based on combined text)
        is_duplicate = self._is_duplicate_conversation(conversation_text)
        
        # Analyze message-level issues across the conversation
        message_issues = [self.analyze_message(text) for text in texts]
        
        # Conversation-level quality assessments
        gibberish_ratio = sum(1 for issues in message_issues if issues['gibberish']) / len(message_issues)
        is_gibberish = gibberish_ratio > 0.5  # More than 50% gibberish messages
        
        non_text_ratio = sum(1 for issues in message_issues if issues['non_text_heavy']) / len(message_issues)
        is_non_text_heavy = non_text_ratio > 0.5
        
        has_offensive = any(issues['offensive'] for issues in message_issues)
        has_spam = any(issues['spam'] for issues in message_issues)
        
        encoding_ratio = sum(1 for issues in message_issues if issues['encoding_issues']) / len(message_issues)
        has_encoding_issues = encoding_ratio > 0.3  # More than 30% with encoding issues
        
        language_ratio = sum(1 for issues in message_issues if issues['language_issues']) / len(message_issues)
        has_language_issues = language_ratio > 0.5  # More than 50% with language issues
        
        # Check if conversation is non-conversational
        non_conv_ratio = sum(1 for issues in message_issues if issues['non_conversational']) / len(message_issues)
        is_non_conversational = non_conv_ratio > 0.7  # More than 70% non-conversational
        
        # Overall quality assessment
        quality_issues = sum([
            is_empty, is_too_short, is_gibberish, is_non_text_heavy,
            has_offensive, has_spam, has_encoding_issues, has_language_issues,
            is_non_conversational
        ])
        is_low_quality = quality_issues >= 3  # 3 or more quality issues
        
        return {
            'empty': is_empty,
            'too_short': is_too_short,
            'too_long': is_too_long,
            'duplicate': is_duplicate,
            'gibberish': is_gibberish,
            'non_text_heavy': is_non_text_heavy,
            'offensive': has_offensive,
            'spam': has_spam,
            'non_conversational': is_non_conversational,
            'encoding_issues': has_encoding_issues,
            'language_issues': has_language_issues,
            'single_turn': is_single_turn,
            'low_quality': is_low_quality
        }
    
    def _is_duplicate_conversation(self, conversation_text: str) -> bool:
        """Check if this conversation is a duplicate based on combined text"""
        conv_hash = self._fast_hash(conversation_text.strip().lower())
        if conv_hash in self.message_hashes:  # Reusing the same hash set
            return True
        self.message_hashes.add(conv_hash)
        return False
    
    def analyze_message(self, text: str) -> Dict[str, bool]:
        """
        Analyze a single message for various quality issues
        
        Returns:
            Dictionary with boolean flags for each quality issue
        """
        if text is None:
            text = ""
        
        issues = {
            'empty': self._is_empty_or_whitespace(text),
            'too_short': self._is_too_short(text),
            'too_long': self._is_too_long(text),
            'duplicate': self._is_duplicate(text),
            'gibberish': self._is_gibberish(text),
            'non_text_heavy': self._is_non_text_heavy(text),
            'offensive': self._contains_profanity(text),
            'spam': self._is_spam_like(text),
            'non_conversational': self._is_non_conversational(text),
            'encoding_issues': self._has_encoding_issues(text),
            'language_issues': self._detect_language_issues(text),
        }
        
        return issues
    
    def analyze_batch(self, conversations: List[List[Dict]]) -> Tuple[DataQualityMetrics, Dict]:
        """
        Analyze a batch of conversations
        
        Args:
            conversations: List of conversations, where each conversation is a list of message dictionaries
            
        Returns:
            Tuple of (metrics, detailed_stats)
        """
        batch_metrics = DataQualityMetrics()
        batch_stats = defaultdict(Counter)
        
        for conversation in conversations:
            batch_metrics.total_conversations += 1
            batch_metrics.total_messages += len(conversation)  # Track total messages for stats
            
            # Analyze the conversation
            issues = self.analyze_conversation(conversation)
            
            # Update conversation-level metrics
            if issues['empty']:
                batch_metrics.empty_conversations += 1
            if issues['too_short']:
                batch_metrics.too_short_conversations += 1
            if issues['too_long']:
                batch_metrics.too_long_conversations += 1
            if issues['duplicate']:
                batch_metrics.duplicate_conversations += 1
            if issues['gibberish']:
                batch_metrics.gibberish_conversations += 1
            if issues['non_text_heavy']:
                batch_metrics.non_text_heavy_conversations += 1
            if issues['offensive']:
                batch_metrics.offensive_conversations += 1
            if issues['spam']:
                batch_metrics.spam_conversations += 1
            if issues['non_conversational']:
                batch_metrics.non_conversational_conversations += 1
            if issues['encoding_issues']:
                batch_metrics.encoding_issues_conversations += 1
            if issues['language_issues']:
                batch_metrics.language_issues_conversations += 1
            if issues['single_turn']:
                batch_metrics.single_turn_conversations += 1
            if issues['low_quality']:
                batch_metrics.low_quality_conversations += 1
            
            # Update detailed stats
            for issue_type, has_issue in issues.items():
                batch_stats[issue_type][has_issue] += 1
        
        return batch_metrics, batch_stats
    
    def process_snowflake_data(self, 
                              table_name: str, 
                              text_column: str = 'TEXT',
                              date_column: str = 'SENTAT',
                              conversation_id_column: str = 'CONVERSATIONID',
                              type_column: str = 'TYPE',
                              limit: Optional[int] = None) -> Dict:
        """
        Process data directly from Snowflake in batches of complete conversations
        Only includes conversations with at least one HUMAN_TO_AI and one AI_TO_HUMAN message.
        
        Args:
            table_name: Name of the Snowflake table
            text_column: Name of the column containing text data
            date_column: Name of the column containing date data
            conversation_id_column: Name of the column containing conversation ID
            type_column: Name of the column containing message type
            limit: Optional limit for testing (None for full dataset)
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting Snowflake conversation-based data processing for table: {table_name}")
        
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
            cursor = conn.cursor()
            
            # First, get count of conversations in the date range
            if limit:
                # For exact limit, we'll use the limit directly
                total_conversations = limit
                logger.info(f"Using exact limit of {limit:,} conversations")
            else:
                # For sampling or full dataset, get actual count
                sample_clause = f" SAMPLE ({self.sample_rate * 100})" if self.use_sampling else ""
                count_query = f"""
                    SELECT COUNT(DISTINCT {conversation_id_column}) 
                    FROM {table_name}{sample_clause}
                    WHERE {date_column} >= '{self.start_date}' 
                    AND {date_column} <= '{self.end_date}'
                """
                cursor.execute(count_query)
                total_conversations = cursor.fetchone()[0]
            
            logger.info(f"Processing {total_conversations:,} conversations from {self.start_date} to {self.end_date}")
            
            # Get list of conversation IDs to process
            if limit:
                # For exact limit, use random sampling with ORDER BY RANDOM()
                conv_ids_query = f"""
                    SELECT DISTINCT {conversation_id_column} 
                    FROM {table_name}
                    WHERE {date_column} >= '{self.start_date}' 
                    AND {date_column} <= '{self.end_date}'
                    ORDER BY RANDOM()
                    LIMIT {limit}
                """
            else:
                # For full dataset or sampling mode
                conv_ids_query = f"""
                    SELECT DISTINCT {conversation_id_column} 
                    FROM {table_name}{sample_clause}
                    WHERE {date_column} >= '{self.start_date}' 
                    AND {date_column} <= '{self.end_date}'
                """
            
            cursor.execute(conv_ids_query)
            conversation_ids = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Retrieved {len(conversation_ids):,} conversation IDs")
            
            # Process conversations in batches
            processed_conversations = 0
            start_time = time.time()
            
            # Process conversation IDs in chunks for batch queries
            conv_batch_size = min(100, self.batch_size // 10)  # Smaller batches for conversation queries
            
            with tqdm(total=len(conversation_ids), desc="Processing conversations") as pbar:
                for i in range(0, len(conversation_ids), conv_batch_size):
                    batch_conv_ids = conversation_ids[i:i + conv_batch_size]
                    
                    # Create IN clause for batch of conversation IDs
                    conv_ids_str = "', '".join(str(cid) for cid in batch_conv_ids)
                    
                    # Fetch all messages for this batch of conversations
                    messages_query = f"""
                        SELECT {conversation_id_column}, {text_column}, {date_column}
                        FROM {table_name} 
                        WHERE {conversation_id_column} IN ('{conv_ids_str}')
                        AND {date_column} >= '{self.start_date}' 
                        AND {date_column} <= '{self.end_date}'
                        ORDER BY {conversation_id_column}, {date_column}
                    """
                    
                    cursor.execute(messages_query)
                    rows = cursor.fetchall()
                    
                    # Group messages by conversation ID
                    conversations = defaultdict(list)
                    for row in rows:
                        conv_id, text, sent_at = row
                        conversations[conv_id].append({
                            'conversation_id': conv_id,
                            'text': text if text is not None else "",
                            'sent_at': sent_at
                        })
                    
                    # Convert to list of conversation lists
                    conversation_batch = list(conversations.values())
                    
                    if conversation_batch:
                        # Analyze batch of conversations
                        batch_metrics, batch_stats = self.analyze_batch(conversation_batch)
                        
                        # Merge with overall metrics
                        self._merge_metrics(batch_metrics)
                        self._merge_stats(batch_stats)
                    
                    processed_conversations += len(conversation_batch)
                    pbar.update(len(conversation_batch))
                    
                    # Log progress every 1000 conversations
                    if processed_conversations % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_conversations / elapsed
                        logger.info(f"Processed {processed_conversations:,} conversations at {rate:.0f} conv/sec")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            return self._generate_report()
            
        finally:
            conn.close()
    
    def _merge_metrics(self, batch_metrics: DataQualityMetrics):
        """Merge batch metrics with overall metrics"""
        self.metrics.total_conversations += batch_metrics.total_conversations
        self.metrics.total_messages += batch_metrics.total_messages
        self.metrics.empty_conversations += batch_metrics.empty_conversations
        self.metrics.too_short_conversations += batch_metrics.too_short_conversations
        self.metrics.too_long_conversations += batch_metrics.too_long_conversations
        self.metrics.duplicate_conversations += batch_metrics.duplicate_conversations
        self.metrics.gibberish_conversations += batch_metrics.gibberish_conversations
        self.metrics.non_text_heavy_conversations += batch_metrics.non_text_heavy_conversations
        self.metrics.offensive_conversations += batch_metrics.offensive_conversations
        self.metrics.spam_conversations += batch_metrics.spam_conversations
        self.metrics.non_conversational_conversations += batch_metrics.non_conversational_conversations
        self.metrics.encoding_issues_conversations += batch_metrics.encoding_issues_conversations
        self.metrics.language_issues_conversations += batch_metrics.language_issues_conversations
        self.metrics.single_turn_conversations += batch_metrics.single_turn_conversations
        self.metrics.low_quality_conversations += batch_metrics.low_quality_conversations
    
    def _merge_stats(self, batch_stats: Dict):
        """Merge batch detailed stats with overall stats"""
        for category, counter in batch_stats.items():
            for key, value in counter.items():
                self.detailed_stats[category][key] += value
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'total_conversations_analyzed': self.metrics.total_conversations,
                'total_messages_analyzed': self.metrics.total_messages,
                'usable_conversation_percentage': round(self.metrics.usable_percentage(), 2),
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'sampling_used': self.use_sampling,
                'sample_rate': self.sample_rate if self.use_sampling else 1.0,
                'date_range': f"{self.start_date} to {self.end_date}"
            },
            'quality_issues': {
                'empty_conversations': {
                    'count': self.metrics.empty_conversations,
                    'percentage': round((self.metrics.empty_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'too_short_conversations': {
                    'count': self.metrics.too_short_conversations,
                    'percentage': round((self.metrics.too_short_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'too_long_conversations': {
                    'count': self.metrics.too_long_conversations,
                    'percentage': round((self.metrics.too_long_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'duplicate_conversations': {
                    'count': self.metrics.duplicate_conversations,
                    'percentage': round((self.metrics.duplicate_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'gibberish_conversations': {
                    'count': self.metrics.gibberish_conversations,
                    'percentage': round((self.metrics.gibberish_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'non_text_heavy_conversations': {
                    'count': self.metrics.non_text_heavy_conversations,
                    'percentage': round((self.metrics.non_text_heavy_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'offensive_conversations': {
                    'count': self.metrics.offensive_conversations,
                    'percentage': round((self.metrics.offensive_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'spam_conversations': {
                    'count': self.metrics.spam_conversations,
                    'percentage': round((self.metrics.spam_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'non_conversational_conversations': {
                    'count': self.metrics.non_conversational_conversations,
                    'percentage': round((self.metrics.non_conversational_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'encoding_issues_conversations': {
                    'count': self.metrics.encoding_issues_conversations,
                    'percentage': round((self.metrics.encoding_issues_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'language_issues_conversations': {
                    'count': self.metrics.language_issues_conversations,
                    'percentage': round((self.metrics.language_issues_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'single_turn_conversations': {
                    'count': self.metrics.single_turn_conversations,
                    'percentage': round((self.metrics.single_turn_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                },
                'low_quality_conversations': {
                    'count': self.metrics.low_quality_conversations,
                    'percentage': round((self.metrics.low_quality_conversations / self.metrics.total_conversations) * 100, 2) if self.metrics.total_conversations > 0 else 0
                }
            },
            'projected_full_dataset': self._project_to_full_dataset() if self.use_sampling else None
        }
        
        return report
    
    def _project_to_full_dataset(self) -> Dict:
        """Project sampling results to full dataset based on date range and conversation analysis"""
        scale_factor = 1 / self.sample_rate
        
        # Estimate total conversations in date range
        estimated_conversations_in_range = int(self.metrics.total_conversations * scale_factor)
        
        # Calculate usable vs unusable conversations
        usable_percentage = self.metrics.usable_percentage()
        estimated_usable_conversations = int(estimated_conversations_in_range * (usable_percentage / 100))
        estimated_unusable_conversations = estimated_conversations_in_range - estimated_usable_conversations
        
        # Estimate total messages (assuming average messages per conversation from sample)
        avg_messages_per_conversation = self.metrics.total_messages / self.metrics.total_conversations if self.metrics.total_conversations > 0 else 0
        estimated_total_messages = int(estimated_conversations_in_range * avg_messages_per_conversation)
        
        return {
            'estimated_total_conversations': estimated_conversations_in_range,
            'estimated_total_messages': estimated_total_messages,
            'average_messages_per_conversation': round(avg_messages_per_conversation, 2),
            'estimated_usable_conversations': estimated_usable_conversations,
            'estimated_unusable_conversations': estimated_unusable_conversations,
            'projected_conversation_issues': {
                'empty_conversations': int(self.metrics.empty_conversations * scale_factor),
                'too_short_conversations': int(self.metrics.too_short_conversations * scale_factor),
                'too_long_conversations': int(self.metrics.too_long_conversations * scale_factor),
                'duplicate_conversations': int(self.metrics.duplicate_conversations * scale_factor),
                'gibberish_conversations': int(self.metrics.gibberish_conversations * scale_factor),
                'non_text_heavy_conversations': int(self.metrics.non_text_heavy_conversations * scale_factor),
                'offensive_conversations': int(self.metrics.offensive_conversations * scale_factor),
                'spam_conversations': int(self.metrics.spam_conversations * scale_factor),
                'non_conversational_conversations': int(self.metrics.non_conversational_conversations * scale_factor),
                'encoding_issues_conversations': int(self.metrics.encoding_issues_conversations * scale_factor),
                'language_issues_conversations': int(self.metrics.language_issues_conversations * scale_factor),
                'single_turn_conversations': int(self.metrics.single_turn_conversations * scale_factor),
                'low_quality_conversations': int(self.metrics.low_quality_conversations * scale_factor)
            }
        }
    
    def save_report(self, report: Dict, filename: str = 'data_quality_report.json'):
        """Save analysis report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {filename}")
    
    def create_visualizations(self, report: Dict, output_dir: str = 'quality_analysis_plots'):
        """Create visualization plots for the analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Quality Issues Distribution
        issues = report['quality_issues']
        issue_names = list(issues.keys())
        issue_percentages = [issues[name]['percentage'] for name in issue_names]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(issue_names)), issue_percentages)
        plt.xlabel('Quality Issues')
        plt.ylabel('Percentage of Messages')
        plt.title('Distribution of Data Quality Issues')
        plt.xticks(range(len(issue_names)), [name.replace('_', ' ').title() for name in issue_names], 
                   rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, pct in zip(bars, issue_percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/quality_issues_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Usability Pie Chart
        usable_pct = report['summary']['usable_conversation_percentage']
        unusable_pct = 100 - usable_pct
        
        plt.figure(figsize=(10, 8))
        plt.pie([usable_pct, unusable_pct], 
                labels=[f'Usable ({usable_pct:.1f}%)', f'Unusable ({unusable_pct:.1f}%)'],
                colors=['#2ecc71', '#e74c3c'],
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Overall Data Usability')
        plt.savefig(f'{output_dir}/usability_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")


def main():
    """Main execution function"""
    logger.info("Starting Conversation-Based Data Quality Analysis for Conversation Dataset (2023-07 to 2025-01)")
    
    # Configuration
    table_name = os.environ.get('SNOWFLAKE_CONVERSATION_TABLE', 'CONVERSATIONS')
    text_column = 'TEXT'
    date_column = os.environ.get('SNOWFLAKE_DATE_COLUMN', 'SENTAT')
    conversation_id_column = os.environ.get('SNOWFLAKE_CONVERSATION_ID_COLUMN', 'CONVERSATIONID')
    
    # For testing - use a smaller sample
    use_full_analysis = os.environ.get('USE_FULL_ANALYSIS', 'false').lower() == 'true'
    test_limit = None if use_full_analysis else 1000  # 1000 conversations for testing (instead of 100k messages)
    
    # Initialize analyzer
    # For large dataset, use sampling for initial analysis
    # Date range filtering: 2023-07-01 to 2025-01-31
    analyzer = FastDataQualityAnalyzer(
        batch_size=50000,  # Larger batches for efficiency
        use_sampling=not use_full_analysis,  # Sample for initial analysis
        sample_rate=0.001,  # 0.1% sample
        start_date='2023-07-01',
        end_date='2025-01-31'
    )
    
    try:
        # Run analysis
        report = analyzer.process_snowflake_data(
            table_name=table_name,
            text_column=text_column,
            date_column=date_column,
            conversation_id_column=conversation_id_column,
            limit=test_limit
        )
        
        # Save report
        analyzer.save_report(report, 'data_quality_analysis_report.json')
        
        # Create visualizations
        analyzer.create_visualizations(report)
        
        # Print summary
        print("\n" + "="*60)
        print("DATA QUALITY ANALYSIS SUMMARY (CONVERSATION-BASED)")
        print("="*60)
        print(f"Total conversations analyzed: {report['summary']['total_conversations_analyzed']:,}")
        print(f"Total messages analyzed: {report['summary']['total_messages_analyzed']:,}")
        print(f"Usable conversation percentage: {report['summary']['usable_conversation_percentage']:.2f}%")
        print(f"Date range: {report['summary']['date_range']}")
        
        if report['projected_full_dataset']:
            proj = report['projected_full_dataset']
            print(f"\nProjected for full dataset in date range:")
            print(f"Estimated total conversations in range: {proj['estimated_total_conversations']:,}")
            print(f"Estimated total messages in range: {proj['estimated_total_messages']:,}")
            print(f"Average messages per conversation: {proj['average_messages_per_conversation']:.1f}")
            print(f"Estimated usable conversations: {proj['estimated_usable_conversations']:,}")
            print(f"Estimated unusable conversations: {proj['estimated_unusable_conversations']:,}")
        
        print("\nTop conversation quality issues:")
        issues = report['quality_issues']
        sorted_issues = sorted(issues.items(), key=lambda x: x[1]['percentage'], reverse=True)
        for issue_name, issue_data in sorted_issues[:5]:
            print(f"  {issue_name.replace('_', ' ').title()}: {issue_data['percentage']:.2f}%")
        
        logger.info("Conversation-based analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 