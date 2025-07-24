#!/usr/bin/env python3
"""
Download Conversations for Local Analysis

This script downloads conversations from Snowflake and stores them locally
for fast, repeatable analysis. Includes automatic resume functionality.

Usage:
    # Download with default settings (recommended)
    python data_quality/download_conversations.py
    
    # Download with custom chunk size
    python data_quality/download_conversations.py --chunk-size 5000
    
    # Check download status
    python data_quality/download_conversations.py --status
"""

import os
import sys
import time
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import snowflake.connector
from tqdm import tqdm


class ConversationDownloader:
    """Downloads conversations from Snowflake to local SQLite database"""
    
    def __init__(self, data_dir='conversation_data', chunk_size=10000, 
                 start_date='2023-07-01', end_date='2025-01-31'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.start_date = start_date
        self.end_date = end_date
        self.sqlite_db = self.data_dir / 'conversations.db'
        
    def setup_database(self):
        """Create SQLite database with required tables"""
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                message_count INTEGER,
                first_message_date TEXT,
                downloaded BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create messages table (matching Snowflake schema)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                conversation_id TEXT,
                text TEXT,
                sent_at TEXT,
                message_type TEXT,
                channel TEXT,
                message_id INTEGER,
                mode TEXT,
                sharing_restricted BOOLEAN,
                sid TEXT,
                user_id TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_id ON messages (conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_downloaded ON conversations (downloaded)')
        
        conn.commit()
        conn.close()
        
    def get_conversation_metadata(self):
        """Download conversation IDs and metadata (Phase 1)"""
        print("Phase 1: Getting conversation metadata from Snowflake...")
        
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        # Check if we already have metadata
        cursor.execute("SELECT COUNT(*) FROM conversations")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"Found existing metadata for {existing_count:,} conversations")
            conn.close()
            return existing_count
        
        # Connect to Snowflake
        sf_conn = snowflake.connector.connect(
            user=os.environ.get('SNOWFLAKE_USER'),
            password=os.environ.get('SNOWFLAKE_PASSWORD'),
            account=os.environ.get('SNOWFLAKE_ACCOUNT'),
            warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE'),
            database=os.environ.get('SNOWFLAKE_DATABASE'),
            schema=os.environ.get('SNOWFLAKE_SCHEMA')
        )
        
        table_name = os.environ.get('SNOWFLAKE_CONVERSATION_TABLE')
        sf_cursor = sf_conn.cursor()
        
        # Get conversation metadata in one query
        metadata_query = f"""
            SELECT 
                CONVERSATIONID,
                COUNT(*) as message_count,
                MIN(SENTAT) as first_message_date
            FROM {table_name}
            WHERE SENTAT >= '{self.start_date}' 
            AND SENTAT <= '{self.end_date}'
            AND CONVERSATIONID IN (
                SELECT CONVERSATIONID
                FROM {table_name}
                WHERE SENTAT >= '{self.start_date}' AND SENTAT <= '{self.end_date}'
                GROUP BY CONVERSATIONID
                HAVING SUM(CASE WHEN TYPE = 'HUMAN_TO_AI' THEN 1 ELSE 0 END) >= 1
                AND SUM(CASE WHEN TYPE = 'AI_TO_HUMAN' THEN 1 ELSE 0 END) >= 1
            )
            GROUP BY CONVERSATIONID
            ORDER BY CONVERSATIONID
        """
        
        print("Executing metadata query...")
        sf_cursor.execute(metadata_query)
        
        # Store metadata locally
        print("Storing metadata locally...")
        batch = []
        batch_size = 20000
        total_stored = 0
        
        for row in tqdm(sf_cursor, desc="Processing metadata"):
            conv_id, msg_count, first_date = row
            batch.append((conv_id, msg_count, str(first_date), False))
            
            if len(batch) >= batch_size:
                cursor.executemany(
                    'INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?)',
                    batch
                )
                conn.commit()
                total_stored += len(batch)
                batch = []
        
        # Insert remaining batch
        if batch:
            cursor.executemany(
                'INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?)',
                batch
            )
            conn.commit()
            total_stored += len(batch)
        
        sf_conn.close()
        conn.close()
        
        print(f"Stored metadata for {total_stored:,} conversations")
        return total_stored
        
    def download_messages(self):
        """Download message content (Phase 2)"""
        print("\nPhase 2: Downloading conversation messages...")
        
        # Get conversations that need downloading
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT conversation_id FROM conversations 
            WHERE downloaded = FALSE 
            ORDER BY conversation_id
        ''')
        
        conv_ids_to_download = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not conv_ids_to_download:
            print("✅ All conversations already downloaded!")
            return
        
        print(f"Need to download {len(conv_ids_to_download):,} conversations")
        
        # Connect to Snowflake
        sf_conn = snowflake.connector.connect(
            user=os.environ.get('SNOWFLAKE_USER'),
            password=os.environ.get('SNOWFLAKE_PASSWORD'),
            account=os.environ.get('SNOWFLAKE_ACCOUNT'),
            warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE'),
            database=os.environ.get('SNOWFLAKE_DATABASE'),
            schema=os.environ.get('SNOWFLAKE_SCHEMA')
        )
        
        table_name = os.environ.get('SNOWFLAKE_CONVERSATION_TABLE')
        sf_cursor = sf_conn.cursor()
        
        # Download in chunks
        total_downloaded = 0
        
        with tqdm(total=len(conv_ids_to_download), desc="Downloading messages") as pbar:
            for i in range(0, len(conv_ids_to_download), self.chunk_size):
                batch_conv_ids = conv_ids_to_download[i:i + self.chunk_size]
                
                try:
                    # Download messages for this batch (get all columns to match schema)
                    conv_ids_str = "', '".join(batch_conv_ids)
                    messages_query = f"""
                        SELECT CONVERSATIONID, TEXT, SENTAT, TYPE, 
                               CHANNEL, ID, MODE, SHARINGRESTRICTED, SID, USERID
                        FROM {table_name} 
                        WHERE CONVERSATIONID IN ('{conv_ids_str}')
                        AND SENTAT >= '{self.start_date}' 
                        AND SENTAT <= '{self.end_date}'
                        ORDER BY CONVERSATIONID, SENTAT
                    """
                    
                    sf_cursor.execute(messages_query)
                    rows = sf_cursor.fetchall()
                    
                    # Store locally with proper transaction handling
                    self._store_batch(batch_conv_ids, rows)
                    
                    total_downloaded += len(batch_conv_ids)
                    pbar.update(len(batch_conv_ids))
                    pbar.set_postfix({
                        'messages': len(rows),
                        'downloaded': total_downloaded
                    })
                    
                except Exception as e:
                    if "too many SQL variables" in str(e):
                        print(f"\n⚠️  SQL error with {len(batch_conv_ids)} conversations. Reducing batch size...")
                        # Split batch in half and retry
                        mid = len(batch_conv_ids) // 2
                        for sub_batch in [batch_conv_ids[:mid], batch_conv_ids[mid:]]:
                            if sub_batch:
                                self._download_small_batch(sf_cursor, table_name, sub_batch)
                        total_downloaded += len(batch_conv_ids)
                        pbar.update(len(batch_conv_ids))
                    else:
                        print(f"\n❌ Error downloading batch: {e}")
                        raise
        
        sf_conn.close()
        print(f"\n✅ Download complete! Downloaded {total_downloaded:,} conversations")
        
    def _download_small_batch(self, sf_cursor, table_name, batch_conv_ids):
        """Download a small batch when hitting SQL limits"""
        conv_ids_str = "', '".join(batch_conv_ids)
        messages_query = f"""
            SELECT CONVERSATIONID, TEXT, SENTAT, TYPE, 
                   CHANNEL, ID, MODE, SHARINGRESTRICTED, SID, USERID
            FROM {table_name} 
            WHERE CONVERSATIONID IN ('{conv_ids_str}')
            AND SENTAT >= '{self.start_date}' 
            AND SENTAT <= '{self.end_date}'
            ORDER BY CONVERSATIONID, SENTAT
        """
        sf_cursor.execute(messages_query)
        rows = sf_cursor.fetchall()
        self._store_batch(batch_conv_ids, rows)
        
    def _store_batch(self, batch_conv_ids, rows):
        """Store a batch of messages with proper transaction handling"""
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # Insert messages (all 10 columns to match schema)
            cursor.executemany(
                'INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                rows
            )
            
            # Mark conversations as downloaded
            for conv_id in batch_conv_ids:
                cursor.execute(
                    'UPDATE conversations SET downloaded = TRUE WHERE conversation_id = ?',
                    (conv_id,)
                )
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Error storing batch: {e}")
            raise
        finally:
            conn.close()
            
    def show_status(self):
        """Show current download status"""
        if not self.sqlite_db.exists():
            print("❌ No database found. Run download first.")
            return
            
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE downloaded = TRUE")
        downloaded = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        remaining = total - downloaded
        
        print("="*60)
        print("DOWNLOAD STATUS")
        print("="*60)
        print(f"Database: {self.sqlite_db}")
        print(f"Total conversations: {total:,}")
        print(f"Downloaded: {downloaded:,}")
        print(f"Remaining: {remaining:,}")
        print(f"Total messages: {total_messages:,}")
        print(f"Progress: {(downloaded/total*100):.1f}%" if total > 0 else "0%")
        print()
        
        if remaining > 0:
            print("🎯 Next step: Resume download")
            print(f"   python data_quality/download_conversations.py")
        else:
            print("✅ Download complete! Ready for analysis")
            print(f"   python data_quality/analyze_local_conversations.py")
        
        conn.close()
        
    def run(self):
        """Run the complete download process"""
        print("="*60)
        print("CONVERSATION DOWNLOAD")
        print("="*60)
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Chunk size: {self.chunk_size:,}")
        print(f"Database: {self.sqlite_db}")
        print("="*60)
        
        start_time = time.time()
        
        # Setup database
        self.setup_database()
        
        # Phase 1: Get metadata
        total_conversations = self.get_conversation_metadata()
        
        # Phase 2: Download messages
        self.download_messages()
        
        duration = time.time() - start_time
        print(f"\n✅ Download pipeline complete!")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Ready for analysis with: python data_quality/analyze_local_conversations.py")


def main():
    parser = argparse.ArgumentParser(description='Download conversations for local analysis')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Number of conversations to download per batch (default: 10000)')
    parser.add_argument('--data-dir', default='conversation_data',
                       help='Directory for local data storage (default: conversation_data)')
    parser.add_argument('--start-date', default='2023-07-01',
                       help='Start date (default: 2023-07-01)')
    parser.add_argument('--end-date', default='2025-01-31',
                       help='End date (default: 2025-01-31)')
    parser.add_argument('--status', action='store_true',
                       help='Show download status and exit')
    
    args = parser.parse_args()
    
    downloader = ConversationDownloader(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if args.status:
        downloader.show_status()
    else:
        try:
            downloader.run()
        except KeyboardInterrupt:
            print("\n\n🛑 Download interrupted")
            print("Progress has been saved. Resume with:")
            print(f"python data_quality/download_conversations.py")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Check your Snowflake environment variables")
            sys.exit(1)


if __name__ == "__main__":
    main() 