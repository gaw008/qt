"""
Cold data archival: export old data to Parquet, delete from Supabase.

This job runs daily via cron to:
1. Export records older than retention period to Parquet files
2. Delete archived records from Supabase cloud database
3. Log all archival operations for audit trail

Parquet files are stored on Vultr temporarily, then synced to Windows local storage.
"""
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available - archival will use JSON fallback")

try:
    import pyarrow
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logger.warning("pyarrow not available - archival will use JSON fallback")

# Import Supabase client
try:
    from supabase_client import supabase_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("supabase_client not available")


# Archive directory configuration
# On Vultr: /root/quant_system_full/archives/
# On Windows: C:\quant_system_v2\archives\
if sys.platform == 'win32':
    ARCHIVE_DIR = Path(r"C:\quant_system_v2\archives")
else:
    ARCHIVE_DIR = Path("/root/quant_system_full/archives")

# Retention policies in days
# Core tables - shorter retention for frequently updated data
# Analysis tables - longer retention for historical analysis
RETENTION_DAYS = {
    # Core tables
    "runs": 30,
    "orders": 90,
    "fills": 90,
    "positions": 7,
    "metrics_snapshots": 7,
    "selection_results": 30,
    # Analysis tables
    "trade_signals": 90,
    "execution_analysis": 90,
    "daily_performance": 365,
    "market_regimes": 365,
    "strategy_performance": 365,
    "factor_crowding_history": 90,
    # Note: ai_training_history and compliance_events are permanent (no archival)
}

# Table configurations: (table_name, date_column)
TABLE_CONFIGS = [
    # Core tables
    ("runs", "started_at"),
    ("orders", "created_at"),
    ("fills", "filled_at"),
    ("positions", "snapshot_at"),
    ("metrics_snapshots", "recorded_at"),
    ("selection_results", "selected_at"),
    # Analysis tables (except permanent ones)
    ("trade_signals", "created_at"),
    ("execution_analysis", "analyzed_at"),
    ("daily_performance", "date"),
    ("market_regimes", "detected_at"),
    ("strategy_performance", "date"),
    ("factor_crowding_history", "recorded_at"),
]


class ArchivalJob:
    """Manages cold data archival from Supabase to local Parquet files."""

    def __init__(self, archive_dir: Path = ARCHIVE_DIR):
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "tables_processed": 0,
            "total_records_archived": 0,
            "total_records_deleted": 0,
            "errors": [],
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None
        }

    def get_client(self):
        """Get Supabase client."""
        if not SUPABASE_AVAILABLE:
            raise RuntimeError("Supabase client not available")
        return supabase_client.get_client()

    def fetch_old_records(
        self,
        table_name: str,
        date_column: str,
        cutoff: datetime,
        batch_size: int = 1000
    ) -> List[Dict]:
        """Fetch records older than cutoff date."""
        all_records = []
        offset = 0

        cutoff_str = cutoff.isoformat()
        client = self.get_client()

        while True:
            try:
                result = client.table(table_name)\
                    .select("*")\
                    .lt(date_column, cutoff_str)\
                    .range(offset, offset + batch_size - 1)\
                    .execute()

                if not result.data:
                    break

                all_records.extend(result.data)
                offset += batch_size

                # Safety limit to prevent infinite loops
                if offset > 100000:
                    logger.warning(f"Reached safety limit for {table_name}")
                    break

            except Exception as e:
                logger.error(f"Error fetching from {table_name}: {e}")
                break

        return all_records

    def export_to_parquet(
        self,
        records: List[Dict],
        table_name: str,
        cutoff: datetime
    ) -> Optional[Path]:
        """Export records to Parquet file."""
        if not records:
            return None

        # Create archive filename with date stamp
        date_stamp = cutoff.strftime('%Y%m%d')

        if PANDAS_AVAILABLE and PYARROW_AVAILABLE:
            # Use Parquet format (preferred)
            archive_file = self.archive_dir / f"{table_name}_{date_stamp}.parquet"

            # If file exists, append to it
            if archive_file.exists():
                existing_df = pd.read_parquet(archive_file)
                new_df = pd.DataFrame(records)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_parquet(archive_file, index=False, compression="snappy")
            else:
                df = pd.DataFrame(records)
                df.to_parquet(archive_file, index=False, compression="snappy")
        else:
            # Fallback to JSON format
            archive_file = self.archive_dir / f"{table_name}_{date_stamp}.json"

            existing_records = []
            if archive_file.exists():
                with open(archive_file, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)

            combined_records = existing_records + records
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(combined_records, f, default=str, indent=2)

        return archive_file

    def delete_archived_records(
        self,
        table_name: str,
        date_column: str,
        cutoff: datetime
    ) -> int:
        """Delete records older than cutoff from Supabase."""
        cutoff_str = cutoff.isoformat()
        client = self.get_client()

        try:
            # Delete in batches to avoid timeout
            deleted_count = 0
            batch_size = 500

            while True:
                # Get IDs of records to delete
                result = client.table(table_name)\
                    .select("id")\
                    .lt(date_column, cutoff_str)\
                    .limit(batch_size)\
                    .execute()

                if not result.data:
                    break

                ids_to_delete = [r["id"] for r in result.data]

                # Delete by IDs
                for record_id in ids_to_delete:
                    client.table(table_name).delete().eq("id", record_id).execute()
                    deleted_count += 1

                # Safety check
                if deleted_count > 50000:
                    logger.warning(f"Safety limit reached for {table_name} deletion")
                    break

            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting from {table_name}: {e}")
            return 0

    def log_archival(
        self,
        table_name: str,
        records_archived: int,
        archive_file: Optional[Path],
        cutoff: datetime
    ):
        """Log archival operation to archive_log table."""
        try:
            client = self.get_client()
            client.table("archive_log").insert({
                "table_name": table_name,
                "records_archived": records_archived,
                "archive_file": str(archive_file) if archive_file else None,
                "archived_before": cutoff.isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to log archival for {table_name}: {e}")

    def archive_table(
        self,
        table_name: str,
        date_column: str,
        retention_days: int
    ) -> Tuple[int, int]:
        """
        Archive a single table.

        Returns:
            Tuple of (records_archived, records_deleted)
        """
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        logger.info(f"Archiving {table_name} - records before {cutoff.date()}")

        # 1. Fetch old records
        records = self.fetch_old_records(table_name, date_column, cutoff)

        if not records:
            logger.info(f"No records to archive in {table_name}")
            return 0, 0

        records_count = len(records)
        logger.info(f"Found {records_count} records to archive in {table_name}")

        # 2. Export to Parquet/JSON
        archive_file = self.export_to_parquet(records, table_name, cutoff)
        if archive_file:
            logger.info(f"Exported to {archive_file}")

        # 3. Delete from Supabase
        deleted_count = self.delete_archived_records(table_name, date_column, cutoff)
        logger.info(f"Deleted {deleted_count} records from {table_name}")

        # 4. Log archival
        self.log_archival(table_name, records_count, archive_file, cutoff)

        return records_count, deleted_count

    def run(self) -> Dict:
        """
        Run archival for all tables with retention policies.

        Returns:
            Statistics dictionary with archival results.
        """
        logger.info("=" * 60)
        logger.info("Starting archival job...")
        logger.info(f"Archive directory: {self.archive_dir}")
        logger.info("=" * 60)

        if not SUPABASE_AVAILABLE:
            self.stats["errors"].append("Supabase client not available")
            logger.error("Supabase client not available - aborting")
            return self.stats

        for table_name, date_column in TABLE_CONFIGS:
            retention = RETENTION_DAYS.get(table_name)
            if retention is None:
                logger.info(f"Skipping {table_name} - no retention policy (permanent)")
                continue

            try:
                archived, deleted = self.archive_table(table_name, date_column, retention)
                self.stats["tables_processed"] += 1
                self.stats["total_records_archived"] += archived
                self.stats["total_records_deleted"] += deleted
            except Exception as e:
                error_msg = f"Failed to archive {table_name}: {str(e)}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)

        self.stats["ended_at"] = datetime.utcnow().isoformat()

        # Summary
        logger.info("=" * 60)
        logger.info("Archival job completed!")
        logger.info(f"Tables processed: {self.stats['tables_processed']}")
        logger.info(f"Total records archived: {self.stats['total_records_archived']}")
        logger.info(f"Total records deleted: {self.stats['total_records_deleted']}")
        if self.stats["errors"]:
            logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats["errors"]:
                logger.warning(f"  - {error}")
        logger.info("=" * 60)

        return self.stats


def run_archival_job() -> Dict:
    """Entry point for running the archival job."""
    job = ArchivalJob()
    return job.run()


# Cron entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cold data archival job")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without actually archiving"
    )
    parser.add_argument(
        "--table",
        type=str,
        help="Archive specific table only"
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No data will be modified")
        # In dry run, just show what would be archived
        if SUPABASE_AVAILABLE:
            client = supabase_client.get_client()
            for table_name, date_column in TABLE_CONFIGS:
                retention = RETENTION_DAYS.get(table_name)
                if retention is None:
                    continue
                if args.table and args.table != table_name:
                    continue

                cutoff = datetime.utcnow() - timedelta(days=retention)
                cutoff_str = cutoff.isoformat()

                try:
                    result = client.table(table_name)\
                        .select("id", count="exact")\
                        .lt(date_column, cutoff_str)\
                        .execute()
                    count = result.count if result.count else 0
                    logger.info(f"{table_name}: {count} records would be archived (before {cutoff.date()})")
                except Exception as e:
                    logger.error(f"{table_name}: Error - {e}")
    else:
        # Run actual archival
        if args.table:
            # Archive specific table
            job = ArchivalJob()
            for table_name, date_column in TABLE_CONFIGS:
                if table_name == args.table:
                    retention = RETENTION_DAYS.get(table_name, 30)
                    job.archive_table(table_name, date_column, retention)
                    break
            else:
                logger.error(f"Unknown table: {args.table}")
        else:
            # Archive all tables
            stats = run_archival_job()

            # Exit with error code if there were failures
            if stats.get("errors"):
                sys.exit(1)
