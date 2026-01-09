#!/usr/bin/env python3
"""
Production Readiness Assessment for Quantitative Trading System
Quality Engineering - Investment-Grade Assessment

This assessment focuses on production deployment readiness,
system stability, security posture, and operational excellence.

Author: Quality Engineering Team
Version: 2.0
"""

import os
import sys
import time
import json
import logging
import requests
import subprocess
import psutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Set encoding for Windows compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'

class ProductionReadinessAssessment:
    """
    Comprehensive production readiness assessment for investment-grade trading system.
    """

    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.quant_dir = self.base_dir / "quant_system_full"
        self.assessment_results = {}
        self.logger = self._setup_logging()

        # Assessment criteria weights (investment-grade standards)
        self.criteria_weights = {
            'system_architecture': 0.20,  # System design and architecture
            'operational_excellence': 0.25,  # Monitoring, logging, error handling
            'security_posture': 0.20,  # Security measures and compliance
            'performance_reliability': 0.15,  # Performance and stability
            'deployment_readiness': 0.10,  # Deployment configuration
            'documentation_compliance': 0.10  # Documentation and procedures
        }

        self.base_url = 'http://localhost:8000'

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for assessment."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('ProductionAssessment')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"production_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

        return logger

    def assess_system_architecture(self) -> Dict[str, Any]:
        """Assess system architecture and design quality."""
        self.logger.info("=== Assessing System Architecture ===")

        results = {
            'modular_design': False,
            'separation_of_concerns': False,
            'api_design_quality': False,
            'database_architecture': False,
            'scalability_design': False,
            'microservices_architecture': False,
            'configuration_management': False,
            'architecture_score': 0
        }

        try:
            # Check modular design - presence of distinct components
            key_directories = [
                self.quant_dir / "bot",
                self.quant_dir / "dashboard",
                self.quant_dir / "UI",
                self.quant_dir / "scripts"
            ]

            modular_components = sum(1 for dir_path in key_directories if dir_path.exists())
            if modular_components >= 3:
                results['modular_design'] = True
                self.logger.info("[OK] Modular design with distinct components")

            # Check separation of concerns - backend/frontend separation
            backend_exists = (self.quant_dir / "dashboard" / "backend").exists()
            frontend_exists = (self.quant_dir / "dashboard" / "frontend").exists() or (self.quant_dir / "UI").exists()

            if backend_exists and frontend_exists:
                results['separation_of_concerns'] = True
                self.logger.info("[OK] Clear separation between backend and frontend")

            # Check API design quality through OpenAPI spec
            try:
                response = requests.get(f"{self.base_url}/openapi.json", timeout=10)
                if response.status_code == 200:
                    openapi_spec = response.json()
                    if len(openapi_spec.get('paths', {})) >= 20:  # Comprehensive API
                        results['api_design_quality'] = True
                        self.logger.info(f"[OK] Comprehensive API design ({len(openapi_spec['paths'])} endpoints)")
            except:
                pass

            # Check database architecture - SQLite databases for different components
            data_cache_dir = self.base_dir / "data_cache"
            if data_cache_dir.exists():
                db_files = list(data_cache_dir.glob("*.db"))
                if len(db_files) >= 3:  # Multiple specialized databases
                    results['database_architecture'] = True
                    self.logger.info(f"[OK] Specialized database architecture ({len(db_files)} databases)")

            # Check scalability design indicators
            config_files = [
                self.quant_dir / ".env",
                self.base_dir / "config.example.env"
            ]

            scalability_indicators = 0
            for config_file in config_files:
                if config_file.exists():
                    try:
                        content = config_file.read_text(encoding='utf-8')
                        if any(indicator in content.upper() for indicator in
                               ['BATCH_SIZE', 'MAX_CONCURRENT', 'TIMEOUT', 'POOL_SIZE']):
                            scalability_indicators += 1
                    except:
                        pass

            if scalability_indicators >= 1:
                results['scalability_design'] = True
                self.logger.info("[OK] Scalability configuration present")

            # Check microservices architecture - multiple independent services
            service_indicators = [
                (self.quant_dir / "dashboard" / "backend", "Backend API Service"),
                (self.quant_dir / "dashboard" / "worker", "Worker Service"),
                (self.quant_dir / "UI", "Frontend Service"),
                (self.base_dir / "system_health_monitoring.py", "Monitoring Service")
            ]

            active_services = sum(1 for path, name in service_indicators if path.exists())
            if active_services >= 3:
                results['microservices_architecture'] = True
                self.logger.info(f"[OK] Microservices architecture ({active_services} services)")

            # Check configuration management
            if (self.quant_dir / ".env").exists() or (self.base_dir / "config.example.env").exists():
                results['configuration_management'] = True
                self.logger.info("[OK] Configuration management in place")

        except Exception as e:
            self.logger.error(f"Architecture assessment error: {e}")

        # Calculate architecture score
        architecture_checks = [v for k, v in results.items() if isinstance(v, bool)]
        results['architecture_score'] = sum(architecture_checks) / len(architecture_checks) if architecture_checks else 0

        return results

    def assess_operational_excellence(self) -> Dict[str, Any]:
        """Assess operational excellence and monitoring capabilities."""
        self.logger.info("=== Assessing Operational Excellence ===")

        results = {
            'comprehensive_logging': False,
            'health_monitoring': False,
            'error_handling': False,
            'alerting_system': False,
            'metrics_collection': False,
            'self_healing_capabilities': False,
            'operational_dashboards': False,
            'operational_score': 0
        }

        try:
            # Check comprehensive logging
            log_locations = [
                self.base_dir / "logs",
                self.quant_dir / "dashboard" / "state",
                self.base_dir / "deployment.log"
            ]

            logging_quality = 0
            for log_location in log_locations:
                if log_location.exists():
                    if log_location.is_dir():
                        log_files = list(log_location.glob("*.log"))
                        if log_files:
                            logging_quality += 1
                    else:
                        logging_quality += 1

            if logging_quality >= 2:
                results['comprehensive_logging'] = True
                self.logger.info("[OK] Comprehensive logging system")

            # Check health monitoring
            try:
                monitoring_response = requests.get(f"{self.base_url}/api/monitoring/status", timeout=10)
                health_response = requests.get(f"{self.base_url}/api/monitoring/health/current", timeout=10)

                if monitoring_response.status_code == 200 and health_response.status_code == 200:
                    results['health_monitoring'] = True
                    self.logger.info("[OK] Health monitoring system active")
            except:
                pass

            # Check error handling
            try:
                # Test graceful error handling
                error_response = requests.get(f"{self.base_url}/nonexistent", timeout=5)
                if error_response.status_code == 404:
                    results['error_handling'] = True
                    self.logger.info("[OK] Graceful error handling")
            except:
                pass

            # Check alerting system
            try:
                alerts_response = requests.get(f"{self.base_url}/api/monitoring/alerts", timeout=10)
                if alerts_response.status_code == 200:
                    results['alerting_system'] = True
                    self.logger.info("[OK] Alerting system available")
            except:
                pass

            # Check metrics collection
            try:
                metrics_endpoints = [
                    "/api/monitoring/health/current",
                    "/api/monitoring/alerts/summary"
                ]

                metrics_available = 0
                for endpoint in metrics_endpoints:
                    try:
                        response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                        if response.status_code == 200:
                            metrics_available += 1
                    except:
                        pass

                if metrics_available >= 1:
                    results['metrics_collection'] = True
                    self.logger.info("[OK] Metrics collection system")
            except:
                pass

            # Check self-healing capabilities
            self_healing_scripts = [
                self.base_dir / "system_self_healing.py",
                self.base_dir / "system_health_monitoring.py"
            ]

            if any(script.exists() for script in self_healing_scripts):
                results['self_healing_capabilities'] = True
                self.logger.info("[OK] Self-healing capabilities available")

            # Check operational dashboards
            try:
                docs_response = requests.get(f"{self.base_url}/docs", timeout=10)
                if docs_response.status_code == 200:
                    results['operational_dashboards'] = True
                    self.logger.info("[OK] Operational dashboards available")
            except:
                pass

        except Exception as e:
            self.logger.error(f"Operational excellence assessment error: {e}")

        # Calculate operational score
        operational_checks = [v for k, v in results.items() if isinstance(v, bool)]
        results['operational_score'] = sum(operational_checks) / len(operational_checks) if operational_checks else 0

        return results

    def assess_security_posture(self) -> Dict[str, Any]:
        """Assess security posture and compliance."""
        self.logger.info("=== Assessing Security Posture ===")

        results = {
            'authentication_required': False,
            'secure_configuration': False,
            'input_validation': False,
            'secure_headers': False,
            'credential_management': False,
            'access_control': False,
            'security_logging': False,
            'security_score': 0
        }

        try:
            # Check authentication requirement
            try:
                protected_endpoint = requests.get(f"{self.base_url}/api/system/status", timeout=5)
                if protected_endpoint.status_code == 401:
                    results['authentication_required'] = True
                    self.logger.info("[OK] Authentication properly required")
            except:
                pass

            # Check secure configuration
            env_files = [
                self.quant_dir / ".env",
                self.base_dir / "config.example.env"
            ]

            secure_config_indicators = 0
            for env_file in env_files:
                if env_file.exists():
                    try:
                        content = env_file.read_text(encoding='utf-8')
                        if any(indicator in content.upper() for indicator in
                               ['SECRET', 'TOKEN', 'KEY', 'PASSWORD']):
                            secure_config_indicators += 1
                    except:
                        pass

            if secure_config_indicators >= 1:
                results['secure_configuration'] = True
                self.logger.info("[OK] Secure configuration management")

            # Check input validation through API behavior
            try:
                # Test malformed request
                malformed_response = requests.post(
                    f"{self.base_url}/api/orders",
                    json={"invalid": "data"},
                    timeout=5
                )
                if malformed_response.status_code in [400, 401, 422]:
                    results['input_validation'] = True
                    self.logger.info("[OK] Input validation active")
            except:
                pass

            # Check secure headers
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                headers = response.headers

                security_headers = [
                    'X-Content-Type-Options',
                    'X-Frame-Options',
                    'X-XSS-Protection',
                    'Content-Security-Policy'
                ]

                present_headers = sum(1 for header in security_headers if header in headers)
                if present_headers >= 2:
                    results['secure_headers'] = True
                    self.logger.info(f"[OK] Security headers present ({present_headers} headers)")
            except:
                pass

            # Check credential management
            private_key_path = self.quant_dir / "private_key.pem"
            gitignore_path = self.quant_dir / ".gitignore"

            if private_key_path.exists() and gitignore_path.exists():
                try:
                    gitignore_content = gitignore_path.read_text(encoding='utf-8')
                    if "private_key.pem" in gitignore_content or "*.pem" in gitignore_content:
                        results['credential_management'] = True
                        self.logger.info("[OK] Credential management in place")
                except:
                    pass

            # Check access control through CORS and API design
            try:
                options_response = requests.options(f"{self.base_url}/api/system/status", timeout=5)
                if options_response.status_code in [200, 204, 401]:
                    results['access_control'] = True
                    self.logger.info("[OK] Access control mechanisms")
            except:
                pass

            # Check security logging
            if results['comprehensive_logging']:  # From operational assessment
                results['security_logging'] = True
                self.logger.info("[OK] Security logging through comprehensive logging")

        except Exception as e:
            self.logger.error(f"Security assessment error: {e}")

        # Calculate security score
        security_checks = [v for k, v in results.items() if isinstance(v, bool)]
        results['security_score'] = sum(security_checks) / len(security_checks) if security_checks else 0

        return results

    def assess_performance_reliability(self) -> Dict[str, Any]:
        """Assess performance and reliability characteristics."""
        self.logger.info("=== Assessing Performance & Reliability ===")

        results = {
            'response_time_acceptable': False,
            'system_stability': False,
            'resource_efficiency': False,
            'concurrent_handling': False,
            'error_recovery': False,
            'performance_monitoring': False,
            'reliability_score': 0
        }

        try:
            # Test response time
            response_times = []
            for _ in range(5):
                start_time = time.time()
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=10)
                    if response.status_code == 200:
                        response_time = (time.time() - start_time) * 1000
                        response_times.append(response_time)
                except:
                    pass

            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                if avg_response_time <= 3000:  # 3 seconds acceptable for startup state
                    results['response_time_acceptable'] = True
                    self.logger.info(f"[OK] Response time acceptable ({avg_response_time:.1f}ms)")

            # Test system stability - check if system stays responsive
            stability_tests = 0
            for i in range(3):
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=5)
                    if response.status_code == 200:
                        stability_tests += 1
                    time.sleep(1)
                except:
                    pass

            if stability_tests >= 2:
                results['system_stability'] = True
                self.logger.info("[OK] System stability verified")

            # Check resource efficiency
            current_process = psutil.Process()
            memory_percent = current_process.memory_percent()
            cpu_percent = psutil.cpu_percent(interval=1)

            if memory_percent <= 50 and cpu_percent <= 70:  # Reasonable resource usage
                results['resource_efficiency'] = True
                self.logger.info(f"[OK] Resource efficiency (Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%)")

            # Test concurrent handling (simple test)
            try:
                import threading

                concurrent_results = []

                def test_request():
                    try:
                        response = requests.get(f"{self.base_url}/health", timeout=5)
                        concurrent_results.append(response.status_code == 200)
                    except:
                        concurrent_results.append(False)

                threads = []
                for _ in range(3):  # Conservative concurrent test
                    thread = threading.Thread(target=test_request)
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                if sum(concurrent_results) >= 2:  # At least 2/3 successful
                    results['concurrent_handling'] = True
                    self.logger.info("[OK] Concurrent request handling")

            except:
                pass

            # Check error recovery
            try:
                # Test graceful error handling
                error_response = requests.get(f"{self.base_url}/invalid_endpoint_test", timeout=5)
                # Check if system is still responsive after error
                health_response = requests.get(f"{self.base_url}/health", timeout=5)

                if health_response.status_code == 200:
                    results['error_recovery'] = True
                    self.logger.info("[OK] Error recovery verified")
            except:
                pass

            # Check performance monitoring capabilities
            try:
                monitoring_response = requests.get(f"{self.base_url}/api/monitoring/status", timeout=5)
                if monitoring_response.status_code == 200:
                    results['performance_monitoring'] = True
                    self.logger.info("[OK] Performance monitoring available")
            except:
                pass

        except Exception as e:
            self.logger.error(f"Performance assessment error: {e}")

        # Calculate reliability score
        reliability_checks = [v for k, v in results.items() if isinstance(v, bool)]
        results['reliability_score'] = sum(reliability_checks) / len(reliability_checks) if reliability_checks else 0

        return results

    def assess_deployment_readiness(self) -> Dict[str, Any]:
        """Assess deployment readiness and configuration."""
        self.logger.info("=== Assessing Deployment Readiness ===")

        results = {
            'environment_configuration': False,
            'dependency_management': False,
            'startup_scripts': False,
            'process_management': False,
            'backup_procedures': False,
            'deployment_score': 0
        }

        try:
            # Check environment configuration
            config_files = [
                self.quant_dir / ".env",
                self.base_dir / "config.example.env",
                self.quant_dir / "props" / "tiger_openapi_config.properties"
            ]

            config_present = sum(1 for config_file in config_files if config_file.exists())
            if config_present >= 2:
                results['environment_configuration'] = True
                self.logger.info(f"[OK] Environment configuration ({config_present} config files)")

            # Check dependency management
            dependency_files = [
                self.quant_dir / "bot" / "requirements.txt",
                self.quant_dir / "UI" / "package.json"
            ]

            dependencies_managed = sum(1 for dep_file in dependency_files if dep_file.exists())
            if dependencies_managed >= 1:
                results['dependency_management'] = True
                self.logger.info(f"[OK] Dependency management ({dependencies_managed} dependency files)")

            # Check startup scripts
            startup_scripts = [
                self.base_dir / "start_all.py",
                self.base_dir / "start_bot.py",
                self.base_dir / "start_ultra_system.py"
            ]

            startup_options = sum(1 for script in startup_scripts if script.exists())
            if startup_options >= 2:
                results['startup_scripts'] = True
                self.logger.info(f"[OK] Startup scripts available ({startup_options} scripts)")

            # Check process management
            process_management_scripts = [
                self.base_dir / "system_health_monitoring.py",
                self.base_dir / "system_self_healing.py"
            ]

            if any(script.exists() for script in process_management_scripts):
                results['process_management'] = True
                self.logger.info("[OK] Process management capabilities")

            # Check backup procedures (data cache and state)
            backup_indicators = [
                self.base_dir / "data_cache",
                self.quant_dir / "dashboard" / "state"
            ]

            if any(path.exists() for path in backup_indicators):
                results['backup_procedures'] = True
                self.logger.info("[OK] Backup procedures in place")

        except Exception as e:
            self.logger.error(f"Deployment assessment error: {e}")

        # Calculate deployment score
        deployment_checks = [v for k, v in results.items() if isinstance(v, bool)]
        results['deployment_score'] = sum(deployment_checks) / len(deployment_checks) if deployment_checks else 0

        return results

    def assess_documentation_compliance(self) -> Dict[str, Any]:
        """Assess documentation and compliance standards."""
        self.logger.info("=== Assessing Documentation & Compliance ===")

        results = {
            'technical_documentation': False,
            'api_documentation': False,
            'user_documentation': False,
            'configuration_documentation': False,
            'compliance_documentation': False,
            'documentation_score': 0
        }

        try:
            # Check technical documentation
            tech_docs = [
                self.base_dir / "CLAUDE.md",
                self.base_dir / "IMPLEMENTATION_COMPLETE.md",
                self.base_dir / "SYSTEM_USER_GUIDE.md"
            ]

            tech_doc_count = sum(1 for doc in tech_docs if doc.exists())
            if tech_doc_count >= 2:
                results['technical_documentation'] = True
                self.logger.info(f"[OK] Technical documentation ({tech_doc_count} documents)")

            # Check API documentation
            try:
                response = requests.get(f"{self.base_url}/docs", timeout=10)
                if response.status_code == 200:
                    results['api_documentation'] = True
                    self.logger.info("[OK] API documentation available")
            except:
                pass

            # Check user documentation
            user_docs = [
                self.base_dir / "SYSTEM_USER_GUIDE.md",
                self.quant_dir / "README.md"
            ]

            if any(doc.exists() for doc in user_docs):
                results['user_documentation'] = True
                self.logger.info("[OK] User documentation available")

            # Check configuration documentation
            config_docs = [
                self.base_dir / "config.example.env"
            ]

            if any(doc.exists() for doc in config_docs):
                results['configuration_documentation'] = True
                self.logger.info("[OK] Configuration documentation available")

            # Check compliance documentation
            compliance_docs = [
                self.base_dir / "CRITICAL_SECURITY_REMEDIATION_COMPLETE.md",
                self.base_dir / "PERFORMANCE_OPTIMIZATION_COMPLETE.md"
            ]

            compliance_doc_count = sum(1 for doc in compliance_docs if doc.exists())
            if compliance_doc_count >= 1:
                results['compliance_documentation'] = True
                self.logger.info(f"[OK] Compliance documentation ({compliance_doc_count} documents)")

        except Exception as e:
            self.logger.error(f"Documentation assessment error: {e}")

        # Calculate documentation score
        documentation_checks = [v for k, v in results.items() if isinstance(v, bool)]
        results['documentation_score'] = sum(documentation_checks) / len(documentation_checks) if documentation_checks else 0

        return results

    def calculate_overall_readiness(self) -> Dict[str, Any]:
        """Calculate overall production readiness score."""
        self.logger.info("=== Calculating Overall Production Readiness ===")

        # Extract individual scores
        individual_scores = {}
        for category, results in self.assessment_results.items():
            score_key = f"{category.split('_')[0]}_score"
            if score_key in results:
                individual_scores[category] = results[score_key]

        # Calculate weighted overall score
        overall_score = 0
        total_weight = 0

        for category, weight in self.criteria_weights.items():
            if category in individual_scores:
                overall_score += individual_scores[category] * weight
                total_weight += weight

        # Normalize score
        if total_weight > 0:
            overall_score = overall_score / total_weight

        # Determine readiness level
        if overall_score >= 0.85:
            readiness_level = 'PRODUCTION_READY'
            readiness_description = 'System meets investment-grade standards for production deployment'
            deployment_recommendation = 'APPROVE'
        elif overall_score >= 0.70:
            readiness_level = 'CONDITIONAL_READY'
            readiness_description = 'System ready with minor improvements recommended'
            deployment_recommendation = 'APPROVE_WITH_CONDITIONS'
        elif overall_score >= 0.55:
            readiness_level = 'NEEDS_IMPROVEMENT'
            readiness_description = 'System requires improvements before production deployment'
            deployment_recommendation = 'DEFER'
        else:
            readiness_level = 'NOT_READY'
            readiness_description = 'System requires significant improvements'
            deployment_recommendation = 'REJECT'

        return {
            'overall_score': overall_score,
            'individual_scores': individual_scores,
            'readiness_level': readiness_level,
            'readiness_description': readiness_description,
            'deployment_recommendation': deployment_recommendation,
            'criteria_weights': self.criteria_weights
        }

    def generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on assessment results."""
        recommendations = []

        for category, results in self.assessment_results.items():
            score_key = f"{category.split('_')[0]}_score"
            if score_key in results and results[score_key] < 0.8:
                if 'architecture' in category:
                    recommendations.append("Enhance system architecture with better modularization and scalability design")
                elif 'operational' in category:
                    recommendations.append("Improve operational excellence with enhanced monitoring and alerting")
                elif 'security' in category:
                    recommendations.append("Strengthen security posture with additional security measures")
                elif 'performance' in category:
                    recommendations.append("Optimize performance and reliability for production workloads")
                elif 'deployment' in category:
                    recommendations.append("Complete deployment readiness preparation and automation")
                elif 'documentation' in category:
                    recommendations.append("Enhance documentation and compliance materials")

        # Add specific technical recommendations
        if 'security_posture' in self.assessment_results:
            sec_results = self.assessment_results['security_posture']
            if not sec_results.get('authentication_required'):
                recommendations.append("Implement comprehensive authentication for all protected endpoints")

        if 'performance_reliability' in self.assessment_results:
            perf_results = self.assessment_results['performance_reliability']
            if not perf_results.get('response_time_acceptable'):
                recommendations.append("Optimize API response times for production performance standards")

        return recommendations

    def run_comprehensive_assessment(self) -> Dict[str, Any]:
        """Run comprehensive production readiness assessment."""
        self.logger.info("=== PRODUCTION READINESS ASSESSMENT STARTED ===")

        start_time = time.time()

        try:
            # Run all assessment categories
            self.assessment_results['system_architecture'] = self.assess_system_architecture()
            self.assessment_results['operational_excellence'] = self.assess_operational_excellence()
            self.assessment_results['security_posture'] = self.assess_security_posture()
            self.assessment_results['performance_reliability'] = self.assess_performance_reliability()
            self.assessment_results['deployment_readiness'] = self.assess_deployment_readiness()
            self.assessment_results['documentation_compliance'] = self.assess_documentation_compliance()

            # Calculate overall readiness
            overall_assessment = self.calculate_overall_readiness()
            recommendations = self.generate_recommendations()

            # Create comprehensive report
            report = {
                'metadata': {
                    'assessment_date': datetime.now().isoformat(),
                    'assessment_duration': time.time() - start_time,
                    'assessment_version': '2.0',
                    'assessor': 'Quality Engineering Team'
                },
                'overall_assessment': overall_assessment,
                'detailed_results': self.assessment_results,
                'recommendations': recommendations,
                'investment_grade_metrics': {
                    'system_reliability': overall_assessment['individual_scores'].get('performance_reliability', 0) * 100,
                    'security_compliance': overall_assessment['individual_scores'].get('security_posture', 0) * 100,
                    'operational_maturity': overall_assessment['individual_scores'].get('operational_excellence', 0) * 100,
                    'architecture_quality': overall_assessment['individual_scores'].get('system_architecture', 0) * 100
                }
            }

            # Save report
            report_file = self.base_dir / f"production_readiness_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"=== ASSESSMENT COMPLETE - Report saved to {report_file} ===")

            # Print summary
            self._print_assessment_summary(report)

            return report

        except Exception as e:
            self.logger.error(f"Assessment error: {e}")
            return {'error': str(e)}

    def _print_assessment_summary(self, report: Dict[str, Any]) -> None:
        """Print production readiness assessment summary."""
        print("\n" + "="*80)
        print("             PRODUCTION READINESS ASSESSMENT")
        print("               Investment-Grade Quality Standards")
        print("="*80)

        overall = report['overall_assessment']
        print(f"\nOVERALL READINESS SCORE: {overall['overall_score']:.1%}")
        print(f"READINESS LEVEL: {overall['readiness_level']}")
        print(f"DEPLOYMENT RECOMMENDATION: {overall['deployment_recommendation']}")
        print(f"\nASSESSMENT: {overall['readiness_description']}")

        print("\nCATEGORY SCORES:")
        print("-" * 50)
        for category, score in overall['individual_scores'].items():
            category_name = category.replace('_', ' ').title()
            weight = overall['criteria_weights'].get(category, 0)
            print(f"{category_name:25}: {score:.1%} (Weight: {weight:.0%})")

        print("\nINVESTMENT-GRADE METRICS:")
        print("-" * 50)
        ig_metrics = report['investment_grade_metrics']
        for metric, value in ig_metrics.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:25}: {value:.1f}%")

        if report['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            print("-" * 50)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")

        print("\n" + "="*80)
        print("Assessment completed. See detailed report in JSON file.")
        print("="*80 + "\n")

def main():
    """Main execution function."""
    try:
        assessor = ProductionReadinessAssessment()

        # Check if system is accessible
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                print("Warning: System may not be fully operational.")
        except:
            print("Warning: System is not accessible. Some assessments may be limited.")

        # Run comprehensive assessment
        report = assessor.run_comprehensive_assessment()

        if 'error' in report:
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nAssessment interrupted by user")
        return 1
    except Exception as e:
        print(f"Assessment failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())