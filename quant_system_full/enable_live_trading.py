#!/usr/bin/env python3
"""
Enable Live Trading Configuration Script

This script safely transitions the system from DRY_RUN mode to live trading.
It includes validation, backup, and rollback capabilities.

CRITICAL: Only run this script after successful validation with validate_tiger_live_trading.py
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradingEnabler:
    """Safe live trading enablement with backup and rollback"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / '.env'
        self.backup_dir = self.project_root / 'backups'
        self.backup_dir.mkdir(exist_ok=True)

    def backup_current_config(self) -> str:
        """Backup current configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f'config_backup_{timestamp}'
        backup_path.mkdir(exist_ok=True)

        # Backup .env file
        if self.env_file.exists():
            shutil.copy2(self.env_file, backup_path / '.env')

        # Backup live trading config
        live_config = self.project_root / 'live_trading_config.json'
        if live_config.exists():
            shutil.copy2(live_config, backup_path / 'live_trading_config.json')

        # Backup Tiger properties
        props_file = self.project_root / 'props' / 'tiger_openapi_config.properties'
        if props_file.exists():
            shutil.copy2(props_file, backup_path / 'tiger_openapi_config.properties')

        logger.info(f"Configuration backed up to: {backup_path}")
        return str(backup_path)

    def validate_prerequisites(self) -> bool:
        """Validate prerequisites for live trading"""
        logger.info("Validating prerequisites...")

        # Check if validation report exists and is recent
        validation_reports = list(self.project_root.glob('tiger_validation_report_*.json'))
        if not validation_reports:
            logger.error("❌ No validation report found. Run validate_tiger_live_trading.py first")
            return False

        # Get most recent validation report
        latest_report = max(validation_reports, key=os.path.getctime)

        # Check if report is recent (within last 24 hours)
        report_age = datetime.now().timestamp() - os.path.getctime(latest_report)
        if report_age > 86400:  # 24 hours
            logger.warning("⚠️  Validation report is older than 24 hours. Consider re-running validation.")

        # Check validation status
        try:
            with open(latest_report, 'r') as f:
                report_data = json.load(f)

            readiness_status = report_data.get('readiness_status')
            if readiness_status not in ['READY', 'READY_WITH_WARNINGS']:
                logger.error(f"❌ System not ready for live trading. Status: {readiness_status}")
                return False

            logger.info(f"✅ Prerequisites validated. Status: {readiness_status}")
            return True

        except Exception as e:
            logger.error(f"❌ Could not read validation report: {e}")
            return False

    def enable_live_trading(self) -> bool:
        """Enable live trading configuration"""
        logger.info("Enabling live trading configuration...")

        try:
            # Read current .env file
            env_lines = []
            if self.env_file.exists():
                with open(self.env_file, 'r') as f:
                    env_lines = f.readlines()

            # Update DRY_RUN setting
            updated_lines = []
            dry_run_found = False

            for line in env_lines:
                if line.strip().startswith('DRY_RUN='):
                    updated_lines.append('DRY_RUN=false\n')
                    dry_run_found = True
                    logger.info("✅ Updated DRY_RUN=false")
                else:
                    updated_lines.append(line)

            # Add DRY_RUN if not found
            if not dry_run_found:
                updated_lines.append('DRY_RUN=false\n')
                logger.info("✅ Added DRY_RUN=false")

            # Write updated .env file
            with open(self.env_file, 'w') as f:
                f.writelines(updated_lines)

            # Update live trading config JSON
            self._update_live_trading_config()

            # Add live trading timestamp
            self._add_live_trading_metadata()

            logger.info("✅ Live trading configuration enabled successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to enable live trading: {e}")
            return False

    def _update_live_trading_config(self):
        """Update live trading configuration JSON"""
        config_file = self.project_root / 'live_trading_config.json'

        # Load existing config or create new
        config = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing config: {e}")

        # Update configuration
        config.update({
            "timestamp": datetime.now().isoformat(),
            "status": "LIVE_TRADING",
            "mode": "PRODUCTION",
            "enabled_at": datetime.now().isoformat(),
            "enabled_by": "enable_live_trading.py"
        })

        # Write updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("✅ Updated live_trading_config.json")

    def _add_live_trading_metadata(self):
        """Add live trading metadata file"""
        metadata = {
            "live_trading_enabled": True,
            "enabled_timestamp": datetime.now().isoformat(),
            "enabled_by": "enable_live_trading.py",
            "version": "1.0",
            "warnings": [
                "LIVE TRADING IS ACTIVE",
                "Real money will be traded",
                "Monitor system closely",
                "Emergency stop available"
            ]
        }

        metadata_file = self.project_root / 'LIVE_TRADING_ACTIVE.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("✅ Created LIVE_TRADING_ACTIVE.json metadata file")

    def display_live_trading_warnings(self):
        """Display critical warnings about live trading"""
        logger.info("\n" + "="*80)
        logger.info("🚨 CRITICAL: LIVE TRADING ENABLED 🚨")
        logger.info("="*80)
        logger.info("⚠️  REAL MONEY WILL BE TRADED")
        logger.info("⚠️  MONITOR THE SYSTEM CLOSELY")
        logger.info("⚠️  HAVE EMERGENCY PROCEDURES READY")
        logger.info("⚠️  VERIFY ALL TRADES BEFORE CONFIRMATION")
        logger.info("="*80)

        logger.info("\n📋 NEXT STEPS:")
        logger.info("1. Restart all trading system components")
        logger.info("2. Monitor the first few trades manually")
        logger.info("3. Verify emergency stop procedures work")
        logger.info("4. Check account balance and positions regularly")
        logger.info("5. Review trade logs for any anomalies")

        logger.info("\n🛑 EMERGENCY PROCEDURES:")
        logger.info("- Emergency stop token configured in .env")
        logger.info("- Kill switch available via dashboard")
        logger.info("- Manual position closure via Tiger interface")
        logger.info("- System restart will stop all trading")

        logger.info("\n📊 MONITORING CHECKLIST:")
        logger.info("□ Account balance tracking")
        logger.info("□ Position monitoring")
        logger.info("□ Trade execution logs")
        logger.info("□ Risk metrics (ES@97.5%)")
        logger.info("□ Performance attribution")
        logger.info("□ Alert system responsiveness")

        logger.info("="*80)

    def create_rollback_script(self, backup_path: str):
        """Create rollback script for emergency reversion"""
        rollback_script = f"""#!/usr/bin/env python3
'''
Emergency Rollback Script
Auto-generated on {datetime.now().isoformat()}

This script reverts the system back to DRY_RUN mode.
Use in case of emergency or if live trading needs to be disabled.
'''

import shutil
from pathlib import Path

def emergency_rollback():
    project_root = Path(__file__).parent
    backup_path = Path("{backup_path}")

    print("🚨 EMERGENCY ROLLBACK INITIATED")

    # Restore .env file
    if (backup_path / '.env').exists():
        shutil.copy2(backup_path / '.env', project_root / '.env')
        print("✅ Restored .env file")

    # Restore live trading config
    if (backup_path / 'live_trading_config.json').exists():
        shutil.copy2(backup_path / 'live_trading_config.json', project_root / 'live_trading_config.json')
        print("✅ Restored live trading config")

    # Remove live trading metadata
    metadata_file = project_root / 'LIVE_TRADING_ACTIVE.json'
    if metadata_file.exists():
        metadata_file.unlink()
        print("✅ Removed live trading metadata")

    print("🎯 ROLLBACK COMPLETE - SYSTEM REVERTED TO DRY_RUN")
    print("🔄 RESTART ALL TRADING COMPONENTS")

if __name__ == "__main__":
    emergency_rollback()
"""

        rollback_file = self.project_root / 'emergency_rollback.py'
        with open(rollback_file, 'w') as f:
            f.write(rollback_script)

        logger.info(f"✅ Created emergency rollback script: {rollback_file}")

def main():
    """Main execution"""
    print("="*80)
    print("🚨 TIGER API LIVE TRADING ENABLEMENT 🚨")
    print("="*80)
    print("This script will enable LIVE TRADING with REAL MONEY")
    print("Make sure you have completed validation first!")
    print("="*80)

    # Confirmation prompt
    confirmation = input("\nAre you sure you want to enable LIVE TRADING? (type 'ENABLE LIVE TRADING'): ")
    if confirmation != 'ENABLE LIVE TRADING':
        print("❌ Live trading enablement cancelled")
        return 1

    enabler = LiveTradingEnabler()

    try:
        # Step 1: Validate prerequisites
        if not enabler.validate_prerequisites():
            print("❌ Prerequisites not met. Cannot enable live trading.")
            return 1

        # Step 2: Backup current configuration
        backup_path = enabler.backup_current_config()

        # Step 3: Enable live trading
        if not enabler.enable_live_trading():
            print("❌ Failed to enable live trading")
            return 1

        # Step 4: Create rollback script
        enabler.create_rollback_script(backup_path)

        # Step 5: Display warnings and instructions
        enabler.display_live_trading_warnings()

        print("\n✅ LIVE TRADING ENABLED SUCCESSFULLY")
        print("🔄 Please restart the trading system to activate live trading")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Live trading enablement cancelled by user")
        return 2
    except Exception as e:
        print(f"\n🚨 Live trading enablement failed: {e}")
        return 3

if __name__ == "__main__":
    exit(main())