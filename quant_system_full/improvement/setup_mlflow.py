#!/usr/bin/env python3
"""
MLflow Experiment Tracking Setup

This script initializes MLflow for tracking experiments in the quantitative trading system.
It sets up experiments, creates baseline runs, and validates the tracking infrastructure.

Usage:
    python setup_mlflow.py [--experiment-name NAME] [--tracking-uri URI]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class MLflowManager:
    """Manage MLflow experiment tracking setup and operations"""

    def __init__(self, tracking_uri: str = "http://localhost:5000",
                 experiment_name: str = "quant_trading_improvements"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = None

    def setup_tracking(self) -> bool:
        """Initialize MLflow tracking server and experiment"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            print(f"MLflow tracking URI set to: {self.tracking_uri}")

            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(
                    name=self.experiment_name
                )
                print(f"Created new experiment: {self.experiment_name} (ID: {self.experiment_id})")
            except mlflow.exceptions.MlflowException as e:
                if "already exists" in str(e):
                    experiment = mlflow.get_experiment_by_name(self.experiment_name)
                    self.experiment_id = experiment.experiment_id
                    print(f"Using existing experiment: {self.experiment_name} (ID: {self.experiment_id})")
                else:
                    raise e

            # Set the experiment as active
            mlflow.set_experiment(self.experiment_name)

            return True

        except Exception as e:
            print(f"Error setting up MLflow tracking: {e}")
            return False

    def create_baseline_run(self) -> Optional[str]:
        """Create a baseline run to test MLflow functionality"""
        try:
            with mlflow.start_run(run_name="baseline_setup_test") as run:
                # Log baseline parameters
                mlflow.log_param("setup_version", "1.0")
                mlflow.log_param("system_type", "quantitative_trading")
                mlflow.log_param("timestamp", datetime.now().isoformat())

                # Log baseline metrics
                mlflow.log_metric("setup_success", 1.0)
                mlflow.log_metric("components_installed", 5.0)  # Number of components

                # Log tags
                mlflow.set_tag("experiment_type", "system_setup")
                mlflow.set_tag("phase", "baseline")
                mlflow.set_tag("environment", "development")

                # Create and log a simple artifact
                artifact_path = project_root / "improvement" / "reports" / "mlflow_setup_log.txt"
                artifact_path.parent.mkdir(exist_ok=True)

                with open(artifact_path, 'w') as f:
                    f.write(f"MLflow Setup Log\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Experiment: {self.experiment_name}\n")
                    f.write(f"Run ID: {run.info.run_id}\n")
                    f.write(f"Tracking URI: {self.tracking_uri}\n")
                    f.write("Setup completed successfully.\n")

                mlflow.log_artifact(str(artifact_path), "setup_logs")

                print(f"Baseline run created successfully: {run.info.run_id}")
                return run.info.run_id

        except Exception as e:
            print(f"Error creating baseline run: {e}")
            return None

    def test_experiment_operations(self) -> bool:
        """Test various MLflow experiment operations"""
        try:
            print("Testing MLflow experiment operations...")

            # Test 1: Parameter logging
            with mlflow.start_run(run_name="parameter_test") as run:
                test_params = {
                    "algorithm": "value_momentum",
                    "universe_size": 500,
                    "selection_size": 20,
                    "risk_model": "ledoit_wolf"
                }

                for key, value in test_params.items():
                    mlflow.log_param(key, value)

                print("✓ Parameter logging test passed")

            # Test 2: Metrics logging
            with mlflow.start_run(run_name="metrics_test") as run:
                test_metrics = {
                    "sharpe_ratio": 1.25,
                    "max_drawdown": 0.08,
                    "win_rate": 0.65,
                    "profit_factor": 1.85
                }

                for key, value in test_metrics.items():
                    mlflow.log_metric(key, value)

                print("✓ Metrics logging test passed")

            # Test 3: Tags and artifacts
            with mlflow.start_run(run_name="tags_artifacts_test") as run:
                mlflow.set_tag("strategy_type", "multi_factor")
                mlflow.set_tag("market", "US")

                # Create a test artifact
                test_file = project_root / "improvement" / "reports" / "test_artifact.json"
                with open(test_file, 'w') as f:
                    import json
                    json.dump({"test": "artifact", "timestamp": datetime.now().isoformat()}, f)

                mlflow.log_artifact(str(test_file), "test_artifacts")

                print("✓ Tags and artifacts test passed")

            print("All MLflow experiment operations tested successfully!")
            return True

        except Exception as e:
            print(f"Error testing experiment operations: {e}")
            return False

    def cleanup_test_runs(self):
        """Clean up test runs (optional)"""
        try:
            # Get all runs for the experiment
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string="tags.experiment_type = 'system_setup'"
            )

            print(f"Found {len(runs)} test runs to clean up")

            # Note: In MLflow, runs are typically not deleted but marked as deleted
            # This is just a placeholder for cleanup logic if needed

        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Set up MLflow experiment tracking')
    parser.add_argument('--experiment-name', default='quant_trading_improvements',
                        help='MLflow experiment name')
    parser.add_argument('--tracking-uri', default='http://localhost:5000',
                        help='MLflow tracking server URI')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run tests, skip setup')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up test runs after setup')

    args = parser.parse_args()

    # Initialize MLflow manager
    manager = MLflowManager(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name
    )

    if not args.test_only:
        print("=== MLflow Experiment Tracking Setup ===")

        # Step 1: Setup tracking
        if not manager.setup_tracking():
            print("Failed to set up MLflow tracking")
            sys.exit(1)

        # Step 2: Create baseline run
        baseline_run_id = manager.create_baseline_run()
        if not baseline_run_id:
            print("Failed to create baseline run")
            sys.exit(1)

        print(f"✓ Baseline run created: {baseline_run_id}")

    # Step 3: Test operations
    if not manager.test_experiment_operations():
        print("Failed experiment operations tests")
        sys.exit(1)

    # Step 4: Cleanup if requested
    if args.cleanup:
        manager.cleanup_test_runs()

    print("\n=== MLflow Setup Complete ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"Tracking URI: {args.tracking_uri}")
    print("You can now use MLflow to track your quantitative trading experiments!")
    print(f"Access the MLflow UI at: {args.tracking_uri}")


if __name__ == "__main__":
    main()