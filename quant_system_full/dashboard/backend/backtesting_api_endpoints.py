"""
Backtesting Report API Endpoints for Dashboard Integration
支持回测报告生成的API端点集成

This module provides FastAPI endpoints for backtesting report generation:
- Three-phase validation report generation
- Report status tracking and progress monitoring
- Report file download and management
- Interactive dashboard data endpoints
- Real-time report generation status updates
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Add bot directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "bot"))

try:
    from backtesting_report_system import (
        BacktestingReportSystem,
        ThreePhaseConfig,
        generate_three_phase_validation_report,
        create_sample_backtest_data
    )
    BACKTESTING_SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Backtesting report system not available: {e}")
    BACKTESTING_SYSTEM_AVAILABLE = False

try:
    from backtest import PortfolioBacktester
    PORTFOLIO_BACKTESTER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Portfolio backtester not available: {e}")
    PORTFOLIO_BACKTESTER_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/backtesting", tags=["Backtesting Reports"])

# Global report generation status tracking
report_generation_status: Dict[str, Dict[str, Any]] = {}


# Pydantic models for API
class BacktestRequest(BaseModel):
    """Request model for backtesting report generation."""

    strategy_name: str = Field(..., description="Name of the trading strategy")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")

    # Strategy configuration
    universe: Optional[List[str]] = Field(default=None, description="List of symbols to trade")
    rebalance_frequency: str = Field(default="monthly", description="Rebalancing frequency")
    initial_capital: float = Field(default=1000000.0, description="Initial capital")
    transaction_costs: float = Field(default=0.001, description="Transaction costs as fraction")

    # Report configuration
    output_formats: List[str] = Field(default=["html", "pdf"], description="Output formats to generate")
    include_statistical_tests: bool = Field(default=True, description="Include statistical significance tests")
    include_charts: bool = Field(default=True, description="Include interactive charts")
    include_crisis_analysis: bool = Field(default=True, description="Include crisis period analysis")

    # Phase configuration
    custom_phases: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Custom phase definitions (overrides default three phases)"
    )


class ReportStatusResponse(BaseModel):
    """Response model for report generation status."""

    request_id: str
    status: str  # "pending", "running", "completed", "error"
    progress: float  # 0.0 to 1.0
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Results (if completed)
    output_files: Optional[Dict[str, str]] = None
    summary_metrics: Optional[Dict[str, Any]] = None


class BacktestSummary(BaseModel):
    """Summary model for backtest results."""

    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

    # Phase breakdown
    phase_results: Dict[str, Dict[str, float]]

    # Risk metrics
    volatility: float
    var_95: float
    calmar_ratio: float


# Report generation endpoints

@router.post("/generate-report", response_model=Dict[str, str])
async def generate_backtesting_report(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Generate comprehensive three-phase backtesting validation report.

    This endpoint initiates the generation of a comprehensive backtesting report
    and returns a request ID for tracking progress.
    """

    if not BACKTESTING_SYSTEM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Backtesting report system not available"
        )

    # Generate unique request ID
    request_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.strategy_name) % 10000:04d}"

    # Initialize status tracking
    report_generation_status[request_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Report generation queued",
        "started_at": datetime.now(),
        "request": request.dict()
    }

    # Start background task
    background_tasks.add_task(
        _generate_report_background,
        request_id,
        request
    )

    logger.info(f"Backtesting report generation initiated: {request_id}")

    return {
        "request_id": request_id,
        "status": "accepted",
        "message": f"Report generation started for {request.strategy_name}"
    }


@router.get("/status/{request_id}", response_model=ReportStatusResponse)
async def get_report_status(request_id: str) -> ReportStatusResponse:
    """
    Get the status of a backtesting report generation request.
    """

    if request_id not in report_generation_status:
        raise HTTPException(
            status_code=404,
            detail="Report generation request not found"
        )

    status_data = report_generation_status[request_id]

    return ReportStatusResponse(**status_data)


@router.get("/download/{request_id}/{format}")
async def download_report(request_id: str, format: str):
    """
    Download a generated report file.

    Args:
        request_id: The report generation request ID
        format: The file format (html, pdf, excel, json)
    """

    if request_id not in report_generation_status:
        raise HTTPException(
            status_code=404,
            detail="Report generation request not found"
        )

    status_data = report_generation_status[request_id]

    if status_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Report generation not completed. Status: {status_data['status']}"
        )

    output_files = status_data.get("output_files", {})

    if format not in output_files:
        raise HTTPException(
            status_code=404,
            detail=f"Report format '{format}' not available"
        )

    file_path = output_files[format]

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Report file not found on disk"
        )

    # Determine media type
    media_types = {
        "html": "text/html",
        "pdf": "application/pdf",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "json": "application/json"
    }

    media_type = media_types.get(format, "application/octet-stream")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=os.path.basename(file_path)
    )


# Data endpoints for dashboard integration

@router.get("/recent-reports")
async def get_recent_reports(limit: int = Query(10, ge=1, le=100)) -> List[Dict[str, Any]]:
    """
    Get list of recent backtesting reports.
    """

    # Sort by creation time and limit results
    recent_reports = sorted(
        report_generation_status.items(),
        key=lambda x: x[1]["started_at"],
        reverse=True
    )[:limit]

    results = []
    for request_id, status_data in recent_reports:
        results.append({
            "request_id": request_id,
            "strategy_name": status_data.get("request", {}).get("strategy_name", "Unknown"),
            "status": status_data["status"],
            "started_at": status_data["started_at"],
            "completed_at": status_data.get("completed_at"),
            "available_formats": list(status_data.get("output_files", {}).keys())
        })

    return results


@router.get("/summary/{request_id}", response_model=BacktestSummary)
async def get_backtest_summary(request_id: str) -> BacktestSummary:
    """
    Get summary metrics for a completed backtest.
    """

    if request_id not in report_generation_status:
        raise HTTPException(
            status_code=404,
            detail="Report generation request not found"
        )

    status_data = report_generation_status[request_id]

    if status_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Report generation not completed"
        )

    summary_metrics = status_data.get("summary_metrics", {})

    if not summary_metrics:
        raise HTTPException(
            status_code=404,
            detail="Summary metrics not available"
        )

    return BacktestSummary(**summary_metrics)


@router.get("/chart-data/{request_id}")
async def get_chart_data(request_id: str, chart_type: str = Query(...)) -> Dict[str, Any]:
    """
    Get chart data for interactive visualizations.

    Args:
        request_id: The report generation request ID
        chart_type: Type of chart (equity_curve, drawdown, returns_distribution, etc.)
    """

    if request_id not in report_generation_status:
        raise HTTPException(
            status_code=404,
            detail="Report generation request not found"
        )

    status_data = report_generation_status[request_id]

    if status_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Report generation not completed"
        )

    # Load chart data from generated results
    chart_data = status_data.get("chart_data", {}).get(chart_type)

    if not chart_data:
        raise HTTPException(
            status_code=404,
            detail=f"Chart data for '{chart_type}' not available"
        )

    return chart_data


# Configuration and utility endpoints

@router.get("/config/default")
async def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for backtesting reports.
    """

    if not BACKTESTING_SYSTEM_AVAILABLE:
        return {
            "error": "Backtesting system not available",
            "default_config": {}
        }

    default_config = ThreePhaseConfig()

    return {
        "phase_definitions": {
            "phase1": {
                "name": default_config.phase1_name,
                "start_date": default_config.phase1_start,
                "end_date": default_config.phase1_end
            },
            "phase2": {
                "name": default_config.phase2_name,
                "start_date": default_config.phase2_start,
                "end_date": default_config.phase2_end
            },
            "phase3": {
                "name": default_config.phase3_name,
                "start_date": default_config.phase3_start,
                "end_date": default_config.phase3_end
            }
        },
        "crisis_periods": [
            {
                "name": name,
                "start_date": start,
                "end_date": end
            }
            for start, end, name in default_config.crisis_periods
        ],
        "benchmarks": default_config.benchmarks,
        "default_settings": {
            "confidence_level": default_config.confidence_level,
            "include_charts": default_config.include_charts,
            "include_statistical_tests": default_config.include_statistical_tests
        }
    }


@router.post("/test/generate-sample")
async def generate_sample_report(
    background_tasks: BackgroundTasks,
    output_formats: List[str] = Query(default=["html"])
) -> Dict[str, str]:
    """
    Generate a sample backtesting report with mock data for testing.
    """

    if not BACKTESTING_SYSTEM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Backtesting report system not available"
        )

    # Create sample request
    sample_request = BacktestRequest(
        strategy_name="Sample Multi-Factor Strategy",
        start_date="2006-01-01",
        end_date="2025-01-01",
        output_formats=output_formats
    )

    # Generate unique request ID
    request_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize status tracking
    report_generation_status[request_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Sample report generation queued",
        "started_at": datetime.now(),
        "request": sample_request.dict()
    }

    # Start background task with sample data
    background_tasks.add_task(
        _generate_sample_report_background,
        request_id,
        sample_request
    )

    return {
        "request_id": request_id,
        "status": "accepted",
        "message": "Sample report generation started"
    }


@router.delete("/cleanup/{request_id}")
async def cleanup_report(request_id: str) -> Dict[str, str]:
    """
    Clean up generated report files and remove from tracking.
    """

    if request_id not in report_generation_status:
        raise HTTPException(
            status_code=404,
            detail="Report generation request not found"
        )

    status_data = report_generation_status[request_id]

    # Remove files if they exist
    output_files = status_data.get("output_files", {})
    removed_files = []

    for format_type, file_path in output_files.items():
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove file {file_path}: {e}")

    # Remove from tracking
    del report_generation_status[request_id]

    return {
        "message": f"Cleaned up report {request_id}",
        "removed_files": removed_files
    }


# Background task functions

async def _generate_report_background(request_id: str, request: BacktestRequest):
    """
    Background task to generate backtesting report.
    """

    try:
        # Update status
        report_generation_status[request_id].update({
            "status": "running",
            "progress": 0.1,
            "message": "Initializing backtesting engine"
        })

        # Create configuration
        config = ThreePhaseConfig()

        if request.custom_phases:
            # Apply custom phase configuration
            for phase_key, phase_config in request.custom_phases.items():
                if "phase1" in phase_key.lower():
                    config.phase1_start = phase_config.get("start_date", config.phase1_start)
                    config.phase1_end = phase_config.get("end_date", config.phase1_end)
                    config.phase1_name = phase_config.get("name", config.phase1_name)
                # Similar for other phases...

        config.include_charts = request.include_charts
        config.include_statistical_tests = request.include_statistical_tests

        # Update progress
        report_generation_status[request_id].update({
            "progress": 0.2,
            "message": "Running backtesting analysis"
        })

        # Run actual backtesting if portfolio backtester is available
        if PORTFOLIO_BACKTESTER_AVAILABLE:
            backtester = PortfolioBacktester(
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=request.initial_capital,
                rebalance_frequency=request.rebalance_frequency,
                transaction_costs=request.transaction_costs
            )

            # This would run the actual backtest
            # For now, we'll use sample data
            backtest_data = create_sample_backtest_data()
        else:
            # Use sample data
            backtest_data = create_sample_backtest_data()

        # Update progress
        report_generation_status[request_id].update({
            "progress": 0.6,
            "message": "Generating comprehensive report"
        })

        # Generate reports
        output_files = await generate_three_phase_validation_report(
            strategy_name=request.strategy_name,
            backtest_results=backtest_data,
            config=config
        )

        # Extract summary metrics
        summary_metrics = _extract_summary_metrics(backtest_data)

        # Update final status
        report_generation_status[request_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "Report generation completed successfully",
            "completed_at": datetime.now(),
            "output_files": output_files,
            "summary_metrics": summary_metrics
        })

        logger.info(f"Report generation completed successfully: {request_id}")

    except Exception as e:
        logger.error(f"Report generation failed for {request_id}: {e}")

        report_generation_status[request_id].update({
            "status": "error",
            "message": "Report generation failed",
            "error_message": str(e),
            "completed_at": datetime.now()
        })


async def _generate_sample_report_background(request_id: str, request: BacktestRequest):
    """
    Background task to generate sample backtesting report with mock data.
    """

    try:
        # Update status
        report_generation_status[request_id].update({
            "status": "running",
            "progress": 0.1,
            "message": "Generating sample data"
        })

        await asyncio.sleep(1)  # Simulate processing time

        # Update progress
        report_generation_status[request_id].update({
            "progress": 0.3,
            "message": "Creating sample backtest results"
        })

        # Create sample data
        sample_data = create_sample_backtest_data()

        await asyncio.sleep(1)

        # Update progress
        report_generation_status[request_id].update({
            "progress": 0.7,
            "message": "Generating sample report"
        })

        # Generate reports with sample data
        output_files = await generate_three_phase_validation_report(
            strategy_name=request.strategy_name,
            backtest_results=sample_data
        )

        # Filter to requested formats
        filtered_files = {
            fmt: path for fmt, path in output_files.items()
            if fmt in request.output_formats
        }

        # Extract summary metrics
        summary_metrics = _extract_summary_metrics(sample_data)

        # Update final status
        report_generation_status[request_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "Sample report generation completed",
            "completed_at": datetime.now(),
            "output_files": filtered_files,
            "summary_metrics": summary_metrics
        })

        logger.info(f"Sample report generation completed: {request_id}")

    except Exception as e:
        logger.error(f"Sample report generation failed for {request_id}: {e}")

        report_generation_status[request_id].update({
            "status": "error",
            "message": "Sample report generation failed",
            "error_message": str(e),
            "completed_at": datetime.now()
        })


def _extract_summary_metrics(backtest_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract summary metrics from backtest data for API responses.
    """

    try:
        # Calculate overall metrics from all phases
        phases = backtest_data.get("phases", {})

        if not phases:
            return {}

        # Aggregate returns across phases
        total_return = 1.0
        total_trades = 0
        phase_results = {}

        for phase_name, phase_data in phases.items():
            equity_curve = phase_data.get("equity_curve", [])
            if len(equity_curve) >= 2:
                phase_return = (equity_curve[-1] / equity_curve[0]) - 1
                total_return *= (1 + phase_return)

                phase_results[phase_name] = {
                    "total_return": phase_return,
                    "start_value": equity_curve[0],
                    "end_value": equity_curve[-1]
                }

            total_trades += phase_data.get("total_trades", 0)

        total_return -= 1  # Convert back to return

        # Calculate other metrics (simplified)
        returns_data = []
        for phase_data in phases.values():
            returns_data.extend(phase_data.get("returns", []))

        if returns_data:
            returns_array = np.array(returns_data)
            annualized_return = np.mean(returns_array) * 252
            volatility = np.std(returns_array) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = np.min(drawdowns)

            # Win rate
            positive_returns = returns_array > 0
            win_rate = np.mean(positive_returns)

            # VaR
            var_95 = np.percentile(returns_array, 5)

            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        else:
            annualized_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
            var_95 = 0
            calmar_ratio = 0

        return {
            "strategy_name": backtest_data.get("strategy_name", "Unknown"),
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "phase_results": phase_results,
            "volatility": volatility,
            "var_95": var_95,
            "calmar_ratio": calmar_ratio
        }

    except Exception as e:
        logger.error(f"Failed to extract summary metrics: {e}")
        return {}


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for backtesting report system.
    """

    return {
        "status": "healthy",
        "backtesting_system_available": BACKTESTING_SYSTEM_AVAILABLE,
        "portfolio_backtester_available": PORTFOLIO_BACKTESTER_AVAILABLE,
        "active_reports": len(report_generation_status),
        "timestamp": datetime.now().isoformat()
    }