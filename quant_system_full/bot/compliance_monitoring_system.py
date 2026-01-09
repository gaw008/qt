"""
Compliance Monitoring System
合规监控系统

Automated compliance monitoring and reporting system that ensures adherence
to investment-grade standards, regulatory requirements, and risk management policies.

Integrates with all system components to provide comprehensive compliance oversight.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ComplianceViolationType(Enum):
    """Types of compliance violations"""
    RISK_LIMIT_BREACH = "risk_limit_breach"
    POSITION_LIMIT_EXCESS = "position_limit_excess"
    CONCENTRATION_VIOLATION = "concentration_violation"
    EXECUTION_DEVIATION = "execution_deviation"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    OPERATIONAL_FAILURE = "operational_failure"
    REGULATORY_BREACH = "regulatory_breach"

class ViolationSeverity(Enum):
    """Severity levels for compliance violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceRule:
    """Definition of a compliance rule"""
    rule_id: str
    name: str
    description: str
    category: str

    # Rule parameters
    metric_name: str
    threshold_value: float
    threshold_type: str  # "maximum", "minimum", "range"

    # Violation handling
    violation_type: ComplianceViolationType
    severity: ViolationSeverity
    auto_remediation: bool = False
    notification_required: bool = True

    # Rule metadata
    regulatory_reference: Optional[str] = None
    business_justification: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceViolation:
    """Record of a compliance violation"""
    violation_id: str
    rule_id: str
    violation_type: ComplianceViolationType
    severity: ViolationSeverity

    # Violation details
    metric_name: str
    actual_value: float
    threshold_value: float
    deviation_amount: float

    # Context information
    affected_entity: str  # portfolio, position, system component
    timestamp: datetime
    detection_method: str

    # Resolution tracking
    status: str = "open"  # open, investigating, remediated, closed
    assigned_to: Optional[str] = None
    remediation_action: Optional[str] = None
    resolution_timestamp: Optional[datetime] = None

    # Additional context
    market_conditions: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None

@dataclass
class ComplianceReport:
    """Compliance monitoring report"""
    report_id: str
    report_date: datetime
    period_start: datetime
    period_end: datetime

    # Summary statistics
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_type: Dict[str, int]

    # Rule compliance rates
    rule_compliance_rates: Dict[str, float]
    overall_compliance_rate: float

    # Trend analysis
    violation_trend: str  # "improving", "stable", "deteriorating"
    key_risk_areas: List[str]

    # Recommendations
    recommendations: List[str]
    action_items: List[str]

class ComplianceMonitoringSystem:
    """
    Investment-Grade Compliance Monitoring System

    Provides comprehensive compliance monitoring including:
    - Real-time rule monitoring and violation detection
    - Automated remediation for specific violation types
    - Regulatory reporting and documentation
    - Trend analysis and risk identification
    """

    def __init__(self, config_path: str = "config/compliance_config.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Compliance rules and violations
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.active_violations: Dict[str, ComplianceViolation] = {}
        self.violation_history: List[ComplianceViolation] = []

        # Monitoring state
        self.is_monitoring = False
        self.last_check_time = datetime.now()

        # Database for persistence
        self.db_path = "data_cache/compliance.db"
        self._initialize_database()

        # Load standard compliance rules
        self._load_standard_rules()

        self.logger.info("Compliance Monitoring System initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load compliance monitoring configuration"""
        default_config = {
            "monitoring": {
                "check_interval": 30,  # seconds
                "batch_processing": True,
                "real_time_alerts": True
            },
            "thresholds": {
                "es_975_critical": 0.12,
                "es_975_high": 0.10,
                "es_975_medium": 0.08,
                "drawdown_critical": 0.15,
                "drawdown_high": 0.12,
                "drawdown_medium": 0.08,
                "position_limit": 0.05,
                "sector_limit": 0.25,
                "hhi_limit": 0.30,
                "correlation_limit": 0.75
            },
            "notifications": {
                "email_alerts": True,
                "dashboard_alerts": True,
                "regulatory_reporting": True
            },
            "remediation": {
                "auto_risk_reduction": True,
                "auto_position_scaling": True,
                "emergency_stop_enabled": True
            }
        }

        try:
            import json
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for compliance monitoring"""
        logger = logging.getLogger('ComplianceMonitoring')
        logger.setLevel(logging.INFO)

        # File handler
        log_path = Path("logs/compliance.log")
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

    def _initialize_database(self):
        """Initialize SQLite database for compliance data"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                # Compliance rules table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_rules (
                        rule_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        category TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        threshold_value REAL NOT NULL,
                        threshold_type TEXT NOT NULL,
                        violation_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        auto_remediation BOOLEAN DEFAULT FALSE,
                        notification_required BOOLEAN DEFAULT TRUE,
                        regulatory_reference TEXT,
                        business_justification TEXT,
                        last_updated TEXT NOT NULL
                    )
                """)

                # Compliance violations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_violations (
                        violation_id TEXT PRIMARY KEY,
                        rule_id TEXT NOT NULL,
                        violation_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        actual_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        deviation_amount REAL NOT NULL,
                        affected_entity TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        detection_method TEXT NOT NULL,
                        status TEXT DEFAULT 'open',
                        assigned_to TEXT,
                        remediation_action TEXT,
                        resolution_timestamp TEXT,
                        FOREIGN KEY (rule_id) REFERENCES compliance_rules (rule_id)
                    )
                """)

                # Compliance reports table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_reports (
                        report_id TEXT PRIMARY KEY,
                        report_date TEXT NOT NULL,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        total_violations INTEGER NOT NULL,
                        overall_compliance_rate REAL NOT NULL,
                        violation_trend TEXT NOT NULL,
                        report_data TEXT NOT NULL
                    )
                """)

                conn.commit()

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")

    def _load_standard_rules(self):
        """Load standard investment-grade compliance rules"""
        standard_rules = [
            ComplianceRule(
                rule_id="RISK_001",
                name="Expected Shortfall Limit",
                description="Portfolio ES@97.5% must not exceed 10% of NAV",
                category="Risk Management",
                metric_name="portfolio_es_975",
                threshold_value=0.10,
                threshold_type="maximum",
                violation_type=ComplianceViolationType.RISK_LIMIT_BREACH,
                severity=ViolationSeverity.HIGH,
                auto_remediation=True,
                regulatory_reference="Investment Adviser Act Section 206",
                business_justification="Tail risk protection for client capital"
            ),

            ComplianceRule(
                rule_id="RISK_002",
                name="Maximum Drawdown Control",
                description="Portfolio drawdown must not exceed 15%",
                category="Risk Management",
                metric_name="current_drawdown",
                threshold_value=-0.15,
                threshold_type="minimum",
                violation_type=ComplianceViolationType.RISK_LIMIT_BREACH,
                severity=ViolationSeverity.CRITICAL,
                auto_remediation=True,
                regulatory_reference="Fiduciary Duty Requirements",
                business_justification="Capital preservation mandate"
            ),

            ComplianceRule(
                rule_id="POS_001",
                name="Individual Position Limit",
                description="No single position may exceed 5% of portfolio",
                category="Position Management",
                metric_name="max_position_weight",
                threshold_value=0.05,
                threshold_type="maximum",
                violation_type=ComplianceViolationType.POSITION_LIMIT_EXCESS,
                severity=ViolationSeverity.MEDIUM,
                auto_remediation=False,
                regulatory_reference="Diversification Requirements",
                business_justification="Concentration risk management"
            ),

            ComplianceRule(
                rule_id="CON_001",
                name="Sector Concentration Limit",
                description="No sector exposure may exceed 25% of portfolio",
                category="Concentration Risk",
                metric_name="max_sector_weight",
                threshold_value=0.25,
                threshold_type="maximum",
                violation_type=ComplianceViolationType.CONCENTRATION_VIOLATION,
                severity=ViolationSeverity.MEDIUM,
                auto_remediation=False,
                regulatory_reference="Prudent Diversification Standards",
                business_justification="Sector diversification requirements"
            ),

            ComplianceRule(
                rule_id="CON_002",
                name="Factor Concentration Control",
                description="Factor HHI must not exceed 0.30",
                category="Concentration Risk",
                metric_name="factor_hhi",
                threshold_value=0.30,
                threshold_type="maximum",
                violation_type=ComplianceViolationType.CONCENTRATION_VIOLATION,
                severity=ViolationSeverity.HIGH,
                auto_remediation=True,
                business_justification="Factor crowding risk management"
            ),

            ComplianceRule(
                rule_id="EXE_001",
                name="Implementation Shortfall Limit",
                description="Transaction costs must not exceed 50 basis points",
                category="Execution Quality",
                metric_name="implementation_shortfall",
                threshold_value=0.005,
                threshold_type="maximum",
                violation_type=ComplianceViolationType.EXECUTION_DEVIATION,
                severity=ViolationSeverity.MEDIUM,
                auto_remediation=False,
                business_justification="Best execution requirements"
            ),

            ComplianceRule(
                rule_id="OPS_001",
                name="System Uptime Requirement",
                description="System uptime must exceed 99.5% during market hours",
                category="Operational Risk",
                metric_name="system_uptime",
                threshold_value=0.995,
                threshold_type="minimum",
                violation_type=ComplianceViolationType.OPERATIONAL_FAILURE,
                severity=ViolationSeverity.HIGH,
                auto_remediation=False,
                business_justification="Operational reliability standards"
            ),

            ComplianceRule(
                rule_id="DAT_001",
                name="Data Quality Standard",
                description="Data quality score must exceed 99%",
                category="Data Quality",
                metric_name="data_quality_score",
                threshold_value=0.99,
                threshold_type="minimum",
                violation_type=ComplianceViolationType.DATA_QUALITY_ISSUE,
                severity=ViolationSeverity.MEDIUM,
                auto_remediation=False,
                business_justification="Decision quality depends on data quality"
            )
        ]

        for rule in standard_rules:
            self.compliance_rules[rule.rule_id] = rule
            self._store_rule(rule)

        self.logger.info(f"Loaded {len(standard_rules)} standard compliance rules")

    async def start_monitoring(self):
        """Start continuous compliance monitoring"""
        if self.is_monitoring:
            self.logger.warning("Compliance monitoring already active")
            return

        self.is_monitoring = True
        self.logger.info("Starting compliance monitoring")

        try:
            while self.is_monitoring:
                await self._monitoring_cycle()
                await asyncio.sleep(self.config["monitoring"]["check_interval"])

        except Exception as e:
            self.logger.error(f"Compliance monitoring error: {e}")
            await self.stop_monitoring()

    async def stop_monitoring(self):
        """Stop compliance monitoring"""
        self.is_monitoring = False
        self.logger.info("Compliance monitoring stopped")

    async def _monitoring_cycle(self):
        """Single compliance monitoring cycle"""
        try:
            self.last_check_time = datetime.now()

            # Get current system metrics
            current_metrics = await self._collect_current_metrics()

            # Check all compliance rules
            violations_detected = []

            for rule_id, rule in self.compliance_rules.items():
                violation = await self._check_rule_compliance(rule, current_metrics)
                if violation:
                    violations_detected.append(violation)

            # Process new violations
            for violation in violations_detected:
                await self._process_violation(violation)

            # Update violation statuses
            await self._update_violation_statuses()

            self.logger.debug(f"Compliance check completed: {len(violations_detected)} new violations")

        except Exception as e:
            self.logger.error(f"Compliance monitoring cycle failed: {e}")

    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics for compliance checking"""
        try:
            # In a real system, this would collect from all monitoring systems
            # For now, simulate with representative values

            current_metrics = {
                # Risk metrics
                "portfolio_es_975": 0.085,  # 8.5%
                "current_drawdown": -0.06,  # -6%
                "risk_budget_utilization": 0.75,

                # Position metrics
                "max_position_weight": 0.045,  # 4.5%
                "position_count": 25,

                # Concentration metrics
                "max_sector_weight": 0.22,  # 22%
                "factor_hhi": 0.28,
                "max_correlation": 0.68,

                # Execution metrics
                "implementation_shortfall": 0.0035,  # 35 bps
                "daily_transaction_costs": 0.0025,

                # Operational metrics
                "system_uptime": 0.998,  # 99.8%
                "data_quality_score": 0.995,  # 99.5%
                "processing_latency": 2.5,  # seconds
            }

            # Add some realistic variability
            for metric in current_metrics:
                if metric.startswith(('portfolio_', 'max_', 'factor_')):
                    # Add small random variation
                    variation = np.random.normal(0, 0.05) * current_metrics[metric]
                    current_metrics[metric] += variation

            return current_metrics

        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return {}

    async def _check_rule_compliance(self, rule: ComplianceRule,
                                   current_metrics: Dict[str, float]) -> Optional[ComplianceViolation]:
        """Check if a specific rule is being violated"""
        try:
            metric_value = current_metrics.get(rule.metric_name)
            if metric_value is None:
                self.logger.warning(f"Metric {rule.metric_name} not available for rule {rule.rule_id}")
                return None

            # Check if violation exists
            violation_detected = False
            deviation_amount = 0.0

            if rule.threshold_type == "maximum":
                if metric_value > rule.threshold_value:
                    violation_detected = True
                    deviation_amount = metric_value - rule.threshold_value

            elif rule.threshold_type == "minimum":
                if metric_value < rule.threshold_value:
                    violation_detected = True
                    deviation_amount = rule.threshold_value - metric_value

            elif rule.threshold_type == "range":
                # For range violations, would need additional logic
                pass

            if not violation_detected:
                return None

            # Create violation record
            violation_id = f"VIOL_{rule.rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            violation = ComplianceViolation(
                violation_id=violation_id,
                rule_id=rule.rule_id,
                violation_type=rule.violation_type,
                severity=rule.severity,
                metric_name=rule.metric_name,
                actual_value=metric_value,
                threshold_value=rule.threshold_value,
                deviation_amount=deviation_amount,
                affected_entity="portfolio",  # Would be more specific in real system
                timestamp=datetime.now(),
                detection_method="automated_monitoring",
                market_conditions=current_metrics,
                system_state={"monitoring_active": True, "last_check": self.last_check_time.isoformat()}
            )

            return violation

        except Exception as e:
            self.logger.error(f"Rule compliance check failed for {rule.rule_id}: {e}")
            return None

    async def _process_violation(self, violation: ComplianceViolation):
        """Process a detected compliance violation"""
        try:
            # Check if this is a duplicate of an existing open violation
            existing_violation = self._find_existing_violation(violation)
            if existing_violation:
                self.logger.debug(f"Duplicate violation detected: {violation.violation_id}")
                return

            # Add to active violations
            self.active_violations[violation.violation_id] = violation

            # Store in database
            self._store_violation(violation)

            # Log violation
            self.logger.warning(f"Compliance violation detected: {violation.rule_id} - "
                              f"{violation.metric_name} = {violation.actual_value:.4f} "
                              f"(threshold: {violation.threshold_value:.4f})")

            # Check if auto-remediation is enabled
            rule = self.compliance_rules.get(violation.rule_id)
            if rule and rule.auto_remediation:
                await self._auto_remediate_violation(violation)

            # Send notifications
            if rule and rule.notification_required:
                await self._send_violation_notification(violation)

        except Exception as e:
            self.logger.error(f"Violation processing failed: {e}")

    def _find_existing_violation(self, violation: ComplianceViolation) -> Optional[ComplianceViolation]:
        """Check if similar violation already exists"""
        for existing_violation in self.active_violations.values():
            if (existing_violation.rule_id == violation.rule_id and
                existing_violation.affected_entity == violation.affected_entity and
                existing_violation.status == "open"):
                return existing_violation
        return None

    async def _auto_remediate_violation(self, violation: ComplianceViolation):
        """Attempt automatic remediation of violation"""
        try:
            remediation_action = None

            if violation.violation_type == ComplianceViolationType.RISK_LIMIT_BREACH:
                if violation.rule_id == "RISK_001":  # ES limit
                    remediation_action = "Automatic risk reduction: 20% position scaling"
                    # In real system, would trigger risk reduction
                elif violation.rule_id == "RISK_002":  # Drawdown limit
                    remediation_action = "Emergency stop: All new trading halted"
                    # In real system, would halt trading

            elif violation.violation_type == ComplianceViolationType.CONCENTRATION_VIOLATION:
                if violation.rule_id == "CON_002":  # Factor HHI
                    remediation_action = "Factor rebalancing: Reduce concentrated exposures"
                    # In real system, would trigger rebalancing

            if remediation_action:
                violation.remediation_action = remediation_action
                violation.status = "auto_remediated"
                violation.resolution_timestamp = datetime.now()

                self.logger.info(f"Auto-remediation applied to {violation.violation_id}: {remediation_action}")

        except Exception as e:
            self.logger.error(f"Auto-remediation failed for {violation.violation_id}: {e}")

    async def _send_violation_notification(self, violation: ComplianceViolation):
        """Send notification for compliance violation"""
        try:
            # In real system, would send email/SMS/dashboard alerts
            notification_message = (
                f"COMPLIANCE ALERT [{violation.severity.value.upper()}]\n"
                f"Rule: {violation.rule_id}\n"
                f"Metric: {violation.metric_name}\n"
                f"Actual: {violation.actual_value:.4f}\n"
                f"Threshold: {violation.threshold_value:.4f}\n"
                f"Deviation: {violation.deviation_amount:.4f}\n"
                f"Time: {violation.timestamp.isoformat()}"
            )

            self.logger.warning(f"Compliance notification: {notification_message}")

        except Exception as e:
            self.logger.error(f"Notification sending failed: {e}")

    async def _update_violation_statuses(self):
        """Update status of existing violations"""
        try:
            current_metrics = await self._collect_current_metrics()

            resolved_violations = []

            for violation_id, violation in self.active_violations.items():
                if violation.status != "open":
                    continue

                # Check if violation is still occurring
                rule = self.compliance_rules.get(violation.rule_id)
                if not rule:
                    continue

                current_value = current_metrics.get(violation.metric_name)
                if current_value is None:
                    continue

                # Check if back in compliance
                is_compliant = False

                if rule.threshold_type == "maximum":
                    is_compliant = current_value <= rule.threshold_value
                elif rule.threshold_type == "minimum":
                    is_compliant = current_value >= rule.threshold_value

                if is_compliant:
                    violation.status = "resolved"
                    violation.resolution_timestamp = datetime.now()
                    resolved_violations.append(violation_id)

                    self.logger.info(f"Violation resolved: {violation_id}")

            # Move resolved violations to history
            for violation_id in resolved_violations:
                violation = self.active_violations.pop(violation_id)
                self.violation_history.append(violation)
                self._update_violation_in_db(violation)

        except Exception as e:
            self.logger.error(f"Violation status update failed: {e}")

    async def generate_compliance_report(self, period_days: int = 30) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            # Get violations in period
            period_violations = [
                v for v in self.violation_history
                if start_date <= v.timestamp <= end_date
            ]

            # Add current active violations
            period_violations.extend(self.active_violations.values())

            # Calculate summary statistics
            total_violations = len(period_violations)

            violations_by_severity = {}
            for severity in ViolationSeverity:
                violations_by_severity[severity.value] = len([
                    v for v in period_violations if v.severity == severity
                ])

            violations_by_type = {}
            for vtype in ComplianceViolationType:
                violations_by_type[vtype.value] = len([
                    v for v in period_violations if v.violation_type == vtype
                ])

            # Calculate rule compliance rates
            rule_compliance_rates = {}
            total_checks = period_days * 24 * 2  # Assuming checks every 30 minutes

            for rule_id in self.compliance_rules.keys():
                rule_violations = len([v for v in period_violations if v.rule_id == rule_id])
                compliance_rate = max(0.0, (total_checks - rule_violations) / total_checks)
                rule_compliance_rates[rule_id] = compliance_rate

            overall_compliance_rate = np.mean(list(rule_compliance_rates.values()))

            # Analyze trends
            if period_days >= 14:  # Need sufficient data for trend analysis
                mid_point = start_date + timedelta(days=period_days // 2)
                first_half_violations = len([v for v in period_violations if v.timestamp < mid_point])
                second_half_violations = len([v for v in period_violations if v.timestamp >= mid_point])

                if second_half_violations < first_half_violations * 0.8:
                    violation_trend = "improving"
                elif second_half_violations > first_half_violations * 1.2:
                    violation_trend = "deteriorating"
                else:
                    violation_trend = "stable"
            else:
                violation_trend = "insufficient_data"

            # Identify key risk areas
            key_risk_areas = []
            for rule_id, compliance_rate in rule_compliance_rates.items():
                if compliance_rate < 0.95:  # Less than 95% compliance
                    rule = self.compliance_rules[rule_id]
                    key_risk_areas.append(f"{rule.category}: {rule.name}")

            # Generate recommendations
            recommendations = self._generate_recommendations(
                period_violations, rule_compliance_rates, violation_trend
            )

            # Generate action items
            action_items = self._generate_action_items(period_violations, key_risk_areas)

            report_id = f"COMP_RPT_{end_date.strftime('%Y%m%d_%H%M%S')}"

            report = ComplianceReport(
                report_id=report_id,
                report_date=end_date,
                period_start=start_date,
                period_end=end_date,
                total_violations=total_violations,
                violations_by_severity=violations_by_severity,
                violations_by_type=violations_by_type,
                rule_compliance_rates=rule_compliance_rates,
                overall_compliance_rate=overall_compliance_rate,
                violation_trend=violation_trend,
                key_risk_areas=key_risk_areas,
                recommendations=recommendations,
                action_items=action_items
            )

            # Store report
            self._store_compliance_report(report)

            self.logger.info(f"Compliance report generated: {report_id}")
            return report

        except Exception as e:
            self.logger.error(f"Compliance report generation failed: {e}")
            raise

    def _generate_recommendations(self, violations: List[ComplianceViolation],
                                compliance_rates: Dict[str, float],
                                trend: str) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []

        # Trend-based recommendations
        if trend == "deteriorating":
            recommendations.append("Compliance trend is deteriorating - implement enhanced monitoring")
        elif trend == "improving":
            recommendations.append("Compliance trend is improving - maintain current procedures")

        # Rule-specific recommendations
        for rule_id, rate in compliance_rates.items():
            if rate < 0.90:  # Less than 90% compliance
                rule = self.compliance_rules[rule_id]
                recommendations.append(f"Review and strengthen controls for {rule.name}")

        # Severity-based recommendations
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            recommendations.append("Critical violations detected - immediate review required")

        # Type-based recommendations
        risk_violations = [v for v in violations if v.violation_type == ComplianceViolationType.RISK_LIMIT_BREACH]
        if len(risk_violations) > 5:
            recommendations.append("Multiple risk limit breaches - review risk management framework")

        return recommendations

    def _generate_action_items(self, violations: List[ComplianceViolation],
                             risk_areas: List[str]) -> List[str]:
        """Generate specific action items"""
        action_items = []

        # Unresolved violations
        open_violations = [v for v in violations if v.status == "open"]
        if open_violations:
            action_items.append(f"Resolve {len(open_violations)} open violations")

        # Risk area improvements
        for risk_area in risk_areas:
            action_items.append(f"Implement enhanced controls for {risk_area}")

        # Operational improvements
        if len(violations) > 10:
            action_items.append("Review and update compliance monitoring procedures")

        # Training and documentation
        if any(v.severity == ViolationSeverity.HIGH for v in violations):
            action_items.append("Conduct compliance training for relevant staff")

        return action_items

    def _store_rule(self, rule: ComplianceRule):
        """Store compliance rule in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO compliance_rules
                    (rule_id, name, description, category, metric_name, threshold_value,
                     threshold_type, violation_type, severity, auto_remediation,
                     notification_required, regulatory_reference, business_justification, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule.rule_id, rule.name, rule.description, rule.category,
                    rule.metric_name, rule.threshold_value, rule.threshold_type,
                    rule.violation_type.value, rule.severity.value, rule.auto_remediation,
                    rule.notification_required, rule.regulatory_reference,
                    rule.business_justification, rule.last_updated.isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Rule storage failed: {e}")

    def _store_violation(self, violation: ComplianceViolation):
        """Store compliance violation in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO compliance_violations
                    (violation_id, rule_id, violation_type, severity, metric_name,
                     actual_value, threshold_value, deviation_amount, affected_entity,
                     timestamp, detection_method, status, assigned_to, remediation_action,
                     resolution_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.violation_id, violation.rule_id, violation.violation_type.value,
                    violation.severity.value, violation.metric_name, violation.actual_value,
                    violation.threshold_value, violation.deviation_amount, violation.affected_entity,
                    violation.timestamp.isoformat(), violation.detection_method, violation.status,
                    violation.assigned_to, violation.remediation_action,
                    violation.resolution_timestamp.isoformat() if violation.resolution_timestamp else None
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Violation storage failed: {e}")

    def _update_violation_in_db(self, violation: ComplianceViolation):
        """Update violation record in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE compliance_violations
                    SET status = ?, remediation_action = ?, resolution_timestamp = ?
                    WHERE violation_id = ?
                """, (
                    violation.status, violation.remediation_action,
                    violation.resolution_timestamp.isoformat() if violation.resolution_timestamp else None,
                    violation.violation_id
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Violation update failed: {e}")

    def _store_compliance_report(self, report: ComplianceReport):
        """Store compliance report in database"""
        try:
            import json
            report_data = {
                "violations_by_severity": report.violations_by_severity,
                "violations_by_type": report.violations_by_type,
                "rule_compliance_rates": report.rule_compliance_rates,
                "key_risk_areas": report.key_risk_areas,
                "recommendations": report.recommendations,
                "action_items": report.action_items
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO compliance_reports
                    (report_id, report_date, period_start, period_end, total_violations,
                     overall_compliance_rate, violation_trend, report_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report.report_id, report.report_date.isoformat(),
                    report.period_start.isoformat(), report.period_end.isoformat(),
                    report.total_violations, report.overall_compliance_rate,
                    report.violation_trend, json.dumps(report_data)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Report storage failed: {e}")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current compliance monitoring status"""
        return {
            "monitoring_active": self.is_monitoring,
            "total_rules": len(self.compliance_rules),
            "active_violations": len(self.active_violations),
            "last_check": self.last_check_time.isoformat(),
            "violation_summary": {
                severity.value: len([v for v in self.active_violations.values() if v.severity == severity])
                for severity in ViolationSeverity
            }
        }

async def main():
    """Test the compliance monitoring system"""
    compliance_system = ComplianceMonitoringSystem()

    print("Testing Compliance Monitoring System...")

    # Get initial status
    status = compliance_system.get_current_status()
    print(f"System initialized: {status['total_rules']} rules loaded")

    # Run a few monitoring cycles
    for i in range(3):
        await compliance_system._monitoring_cycle()
        print(f"Monitoring cycle {i+1} completed")

    # Generate compliance report
    report = await compliance_system.generate_compliance_report(7)  # 7-day report
    print(f"Compliance report generated: {report.overall_compliance_rate:.1%} compliance rate")

    # Get final status
    final_status = compliance_system.get_current_status()
    print(f"Final status: {final_status['active_violations']} active violations")

    print("Compliance monitoring system test completed")

if __name__ == "__main__":
    asyncio.run(main())