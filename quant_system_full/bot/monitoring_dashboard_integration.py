"""
Monitoring Dashboard Integration
监控面板集成系统

Integrates the real-time monitoring system with the existing FastAPI dashboard backend
to provide unified access to investment-grade monitoring, alerts, and reporting.

Extends the existing dashboard API with institutional-quality monitoring endpoints.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, FileResponse
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from real_time_monitor import RealTimeMonitor, SystemHealthMetrics, MonitoringAlert
from eod_reporting_system import EODReportingSystem

class MonitoringDashboardIntegration:
    """
    Integration layer between investment-grade monitoring system
    and existing dashboard infrastructure.

    Provides REST API endpoints and WebSocket connections for
    real-time monitoring data access.
    """

    def __init__(self):
        self.logger = self._setup_logging()
        self.monitor = RealTimeMonitor()
        self.reporting_system = EODReportingSystem()
        self.router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])
        self.active_websockets: List[WebSocket] = []

        # Setup API routes
        self._setup_routes()

        self.logger.info("Monitoring dashboard integration initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for dashboard integration"""
        logger = logging.getLogger('MonitoringDashboard')
        logger.setLevel(logging.INFO)

        # File handler
        log_path = Path("logs/dashboard_monitoring.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _setup_routes(self):
        """Setup FastAPI routes for monitoring endpoints"""

        @self.router.get("/status")
        async def get_monitoring_status():
            """Get current monitoring system status"""
            try:
                status = self.monitor.get_monitoring_status()
                return JSONResponse(content={
                    "status": "success",
                    "data": status
                })
            except Exception as e:
                self.logger.error(f"Status endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/start")
        async def start_monitoring(background_tasks: BackgroundTasks):
            """Start the monitoring system"""
            try:
                if self.monitor.is_monitoring:
                    return JSONResponse(content={
                        "status": "info",
                        "message": "Monitoring already active"
                    })

                # Start monitoring in background
                background_tasks.add_task(self.monitor.start_monitoring)

                return JSONResponse(content={
                    "status": "success",
                    "message": "Monitoring system started"
                })
            except Exception as e:
                self.logger.error(f"Start monitoring error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/stop")
        async def stop_monitoring():
            """Stop the monitoring system"""
            try:
                await self.monitor.stop_monitoring()
                return JSONResponse(content={
                    "status": "success",
                    "message": "Monitoring system stopped"
                })
            except Exception as e:
                self.logger.error(f"Stop monitoring error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/health/current")
        async def get_current_health():
            """Get current system health metrics"""
            try:
                if not self.monitor.health_history:
                    return JSONResponse(content={
                        "status": "warning",
                        "message": "No health data available"
                    })

                latest_health = self.monitor.health_history[-1]
                health_dict = {
                    "timestamp": latest_health.timestamp.isoformat(),
                    "portfolio_es_975": latest_health.portfolio_es_975,
                    "current_drawdown": latest_health.current_drawdown,
                    "risk_budget_utilization": latest_health.risk_budget_utilization,
                    "tail_dependence": latest_health.tail_dependence,
                    "daily_transaction_costs": latest_health.daily_transaction_costs,
                    "capacity_utilization": latest_health.capacity_utilization,
                    "implementation_shortfall": latest_health.implementation_shortfall,
                    "factor_hhi": latest_health.factor_hhi,
                    "max_correlation": latest_health.max_correlation,
                    "crowding_risk_score": latest_health.crowding_risk_score,
                    "daily_pnl": latest_health.daily_pnl,
                    "sharpe_ratio_ytd": latest_health.sharpe_ratio_ytd,
                    "max_drawdown_ytd": latest_health.max_drawdown_ytd,
                    "active_positions": latest_health.active_positions,
                    "data_freshness": latest_health.data_freshness,
                    "system_uptime": latest_health.system_uptime
                }

                return JSONResponse(content={
                    "status": "success",
                    "data": health_dict
                })
            except Exception as e:
                self.logger.error(f"Current health endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/health/history")
        async def get_health_history(hours: int = 24):
            """Get historical health metrics"""
            try:
                if not self.monitor.health_history:
                    return JSONResponse(content={
                        "status": "warning",
                        "message": "No health data available"
                    })

                # Get data for requested time period
                cutoff_time = datetime.now() - timedelta(hours=hours)
                recent_health = [
                    h for h in self.monitor.health_history
                    if h.timestamp > cutoff_time
                ]

                if not recent_health:
                    return JSONResponse(content={
                        "status": "warning",
                        "message": f"No data available for last {hours} hours"
                    })

                # Convert to time series format
                time_series_data = {
                    "timestamps": [h.timestamp.isoformat() for h in recent_health],
                    "portfolio_es_975": [h.portfolio_es_975 for h in recent_health],
                    "current_drawdown": [h.current_drawdown for h in recent_health],
                    "daily_transaction_costs": [h.daily_transaction_costs for h in recent_health],
                    "factor_hhi": [h.factor_hhi for h in recent_health],
                    "daily_pnl": [h.daily_pnl for h in recent_health],
                    "system_uptime": [h.system_uptime for h in recent_health]
                }

                return JSONResponse(content={
                    "status": "success",
                    "data": time_series_data,
                    "period_hours": hours,
                    "data_points": len(recent_health)
                })
            except Exception as e:
                self.logger.error(f"Health history endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/alerts")
        async def get_alerts(severity: Optional[str] = None, hours: int = 24):
            """Get recent alerts"""
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                recent_alerts = [
                    alert for alert in self.monitor.alerts
                    if alert.timestamp > cutoff_time
                ]

                # Filter by severity if specified
                if severity:
                    recent_alerts = [
                        alert for alert in recent_alerts
                        if alert.severity.upper() == severity.upper()
                    ]

                alerts_data = []
                for alert in recent_alerts:
                    alerts_data.append({
                        "timestamp": alert.timestamp.isoformat(),
                        "severity": alert.severity,
                        "category": alert.category,
                        "message": alert.message,
                        "source_module": alert.source_module,
                        "metric_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "recommendation": alert.recommendation
                    })

                return JSONResponse(content={
                    "status": "success",
                    "data": alerts_data,
                    "total_alerts": len(alerts_data),
                    "period_hours": hours
                })
            except Exception as e:
                self.logger.error(f"Alerts endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/alerts/summary")
        async def get_alerts_summary():
            """Get alerts summary statistics"""
            try:
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                today_alerts = [
                    alert for alert in self.monitor.alerts
                    if alert.timestamp >= today
                ]

                summary = {
                    "total_today": len(today_alerts),
                    "by_severity": {
                        "CRITICAL": len([a for a in today_alerts if a.severity == "CRITICAL"]),
                        "HIGH": len([a for a in today_alerts if a.severity == "HIGH"]),
                        "MEDIUM": len([a for a in today_alerts if a.severity == "MEDIUM"]),
                        "LOW": len([a for a in today_alerts if a.severity == "LOW"])
                    },
                    "by_category": {},
                    "latest_alert": None
                }

                # Count by category
                categories = set(alert.category for alert in today_alerts)
                for category in categories:
                    summary["by_category"][category] = len([
                        a for a in today_alerts if a.category == category
                    ])

                # Get latest alert
                if today_alerts:
                    latest = max(today_alerts, key=lambda x: x.timestamp)
                    summary["latest_alert"] = {
                        "timestamp": latest.timestamp.isoformat(),
                        "severity": latest.severity,
                        "message": latest.message
                    }

                return JSONResponse(content={
                    "status": "success",
                    "data": summary
                })
            except Exception as e:
                self.logger.error(f"Alerts summary endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/reports/daily")
        async def generate_daily_report(background_tasks: BackgroundTasks):
            """Generate daily EOD report"""
            try:
                # Generate report in background
                background_tasks.add_task(self._generate_daily_report_task)

                return JSONResponse(content={
                    "status": "success",
                    "message": "Daily report generation started"
                })
            except Exception as e:
                self.logger.error(f"Daily report endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/reports/latest")
        async def get_latest_reports():
            """Get list of latest generated reports"""
            try:
                reports_path = Path("reports/eod/")
                if not reports_path.exists():
                    return JSONResponse(content={
                        "status": "warning",
                        "message": "No reports directory found"
                    })

                # Get all report files
                report_files = []
                for file in reports_path.glob("*.html"):
                    stat = file.stat()
                    report_files.append({
                        "filename": file.name,
                        "path": str(file),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "html"
                    })

                for file in reports_path.glob("*.json"):
                    stat = file.stat()
                    report_files.append({
                        "filename": file.name,
                        "path": str(file),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "json"
                    })

                # Sort by modification time, newest first
                report_files.sort(key=lambda x: x["modified"], reverse=True)

                return JSONResponse(content={
                    "status": "success",
                    "data": report_files[:10]  # Return latest 10 reports
                })
            except Exception as e:
                self.logger.error(f"Latest reports endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/reports/download/{filename}")
        async def download_report(filename: str):
            """Download a specific report file"""
            try:
                reports_path = Path("reports/eod/")
                file_path = reports_path / filename

                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="Report file not found")

                return FileResponse(
                    path=str(file_path),
                    filename=filename,
                    media_type='application/octet-stream'
                )
            except Exception as e:
                self.logger.error(f"Download report endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/risk/stress-test")
        async def get_stress_test_results():
            """Get latest stress test results"""
            try:
                # This would integrate with stress testing module
                # For now, return simulated data
                stress_scenarios = {
                    "market_crash_2008": {
                        "scenario": "2008 Financial Crisis",
                        "portfolio_loss": -0.35,
                        "max_drawdown": -0.42,
                        "recovery_time_days": 180
                    },
                    "covid_crash_2020": {
                        "scenario": "COVID-19 Market Crash",
                        "portfolio_loss": -0.28,
                        "max_drawdown": -0.32,
                        "recovery_time_days": 90
                    },
                    "tech_bubble_2000": {
                        "scenario": "Tech Bubble Burst",
                        "portfolio_loss": -0.45,
                        "max_drawdown": -0.52,
                        "recovery_time_days": 365
                    }
                }

                return JSONResponse(content={
                    "status": "success",
                    "data": stress_scenarios,
                    "last_updated": datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Stress test endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time monitoring data"""
            await websocket.accept()
            self.active_websockets.append(websocket)

            try:
                while True:
                    # Send latest health data every 30 seconds
                    if self.monitor.health_history:
                        latest_health = self.monitor.health_history[-1]
                        health_data = {
                            "type": "health_update",
                            "timestamp": latest_health.timestamp.isoformat(),
                            "data": {
                                "portfolio_es_975": latest_health.portfolio_es_975,
                                "current_drawdown": latest_health.current_drawdown,
                                "daily_transaction_costs": latest_health.daily_transaction_costs,
                                "factor_hhi": latest_health.factor_hhi,
                                "daily_pnl": latest_health.daily_pnl,
                                "active_positions": latest_health.active_positions
                            }
                        }

                        await websocket.send_json(health_data)

                    # Send recent alerts
                    recent_alerts = [
                        alert for alert in self.monitor.alerts[-5:]
                    ]

                    if recent_alerts:
                        alerts_data = {
                            "type": "alerts_update",
                            "timestamp": datetime.now().isoformat(),
                            "data": [
                                {
                                    "severity": alert.severity,
                                    "category": alert.category,
                                    "message": alert.message,
                                    "timestamp": alert.timestamp.isoformat()
                                }
                                for alert in recent_alerts
                            ]
                        }

                        await websocket.send_json(alerts_data)

                    await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                if websocket in self.active_websockets:
                    self.active_websockets.remove(websocket)

    async def _generate_daily_report_task(self):
        """Background task to generate daily report"""
        try:
            result = await self.reporting_system.generate_daily_report()
            self.logger.info(f"Daily report generation completed: {result['status']}")

            # Notify connected WebSocket clients
            if result['status'] == 'success':
                notification = {
                    "type": "report_generated",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "report_type": "daily",
                        "files": result.get('output_files', [])
                    }
                }

                # Send to all connected WebSocket clients
                for websocket in self.active_websockets:
                    try:
                        await websocket.send_json(notification)
                    except Exception as e:
                        self.logger.warning(f"Failed to send WebSocket notification: {e}")

        except Exception as e:
            self.logger.error(f"Daily report generation task failed: {e}")

    async def broadcast_alert(self, alert: MonitoringAlert):
        """Broadcast new alert to all connected WebSocket clients"""
        try:
            alert_data = {
                "type": "new_alert",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "severity": alert.severity,
                    "category": alert.category,
                    "message": alert.message,
                    "source_module": alert.source_module,
                    "recommendation": alert.recommendation
                }
            }

            # Send to all connected WebSocket clients
            for websocket in self.active_websockets:
                try:
                    await websocket.send_json(alert_data)
                except Exception as e:
                    self.logger.warning(f"Failed to send WebSocket alert: {e}")

        except Exception as e:
            self.logger.error(f"Alert broadcast failed: {e}")

    def get_router(self) -> APIRouter:
        """Get the FastAPI router for integration with main app"""
        return self.router

    async def start_background_monitoring(self):
        """Start monitoring system in background"""
        try:
            if not self.monitor.is_monitoring:
                asyncio.create_task(self.monitor.start_monitoring())
                self.logger.info("Background monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start background monitoring: {e}")

    async def stop_background_monitoring(self):
        """Stop background monitoring"""
        try:
            await self.monitor.stop_monitoring()
            self.logger.info("Background monitoring stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop background monitoring: {e}")

# Integration with existing dashboard backend
def integrate_monitoring_dashboard(app):
    """
    Integrate monitoring dashboard with existing FastAPI app

    Usage:
    from monitoring_dashboard_integration import integrate_monitoring_dashboard
    integrate_monitoring_dashboard(app)
    """
    try:
        monitoring_integration = MonitoringDashboardIntegration()

        # Include the monitoring router
        app.include_router(monitoring_integration.get_router())

        # Store reference for lifecycle management
        app.state.monitoring_integration = monitoring_integration

        # Add startup event to begin monitoring
        @app.on_event("startup")
        async def start_monitoring():
            await monitoring_integration.start_background_monitoring()

        # Add shutdown event to stop monitoring
        @app.on_event("shutdown")
        async def stop_monitoring():
            await monitoring_integration.stop_background_monitoring()

        print("Monitoring dashboard integration completed successfully")

    except Exception as e:
        print(f"Failed to integrate monitoring dashboard: {e}")

if __name__ == "__main__":
    # Test the integration
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="Monitoring Dashboard Test")
    integrate_monitoring_dashboard(app)

    print("Starting test server on http://localhost:8001")
    print("Monitoring endpoints available at http://localhost:8001/api/monitoring/")
    uvicorn.run(app, host="0.0.0.0", port=8001)