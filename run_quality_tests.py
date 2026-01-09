#!/usr/bin/env python3
"""
Quality Test Runner for Refactored Quantitative Trading System

This script runs all quality tests including:
- Unit tests for refactored components
- Integration tests
- Performance benchmarks
- Code quality analysis

Generates comprehensive reports for code quality assessment.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Any

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'quant_system_full' / 'bot'))
sys.path.insert(0, str(current_dir / 'tests'))


class QualityTestRunner:
    """
    Orchestrates all quality tests and generates comprehensive reports.
    """

    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        self.reports_dir = current_dir / 'test_reports'
        self.reports_dir.mkdir(exist_ok=True)

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests and collect results."""
        print("Running Unit Tests...")
        print("-" * 40)

        unit_test_results = {
            'risk_services': self._run_test_module('test_risk_calculation_services'),
            'scoring_services': self._run_test_module('test_scoring_services')
        }

        # Calculate overall unit test metrics
        total_tests = sum(result.get('tests_run', 0) for result in unit_test_results.values())
        total_failures = sum(result.get('failures', 0) for result in unit_test_results.values())
        total_errors = sum(result.get('errors', 0) for result in unit_test_results.values())

        overall_results = {
            'modules': unit_test_results,
            'summary': {
                'total_tests': total_tests,
                'total_failures': total_failures,
                'total_errors': total_errors,
                'success_rate': ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
            }
        }

        self.test_results['unit_tests'] = overall_results
        return overall_results

    def _run_test_module(self, module_name: str) -> Dict[str, Any]:
        """Run a specific test module and parse results."""
        try:
            test_file = current_dir / 'tests' / f'{module_name}.py'
            if not test_file.exists():
                return {'error': f'Test file {test_file} not found'}

            print(f"  Running {module_name}...")

            # Run the test module
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Parse output for test metrics
            output = result.stdout + result.stderr
            test_metrics = self._parse_test_output(output)

            return {
                'success': result.returncode == 0,
                'output': output,
                'return_code': result.returncode,
                **test_metrics
            }

        except subprocess.TimeoutExpired:
            return {'error': 'Test timed out', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _parse_test_output(self, output: str) -> Dict[str, int]:
        """Parse test output to extract metrics."""
        metrics = {
            'tests_run': 0,
            'failures': 0,
            'errors': 0
        }

        try:
            lines = output.split('\n')
            for line in lines:
                if 'Tests run:' in line and 'Failures:' in line and 'Errors:' in line:
                    # Parse line like "Tests run: 15, Failures: 0, Errors: 0"
                    parts = line.split(',')
                    for part in parts:
                        part = part.strip()
                        if 'Tests run:' in part:
                            metrics['tests_run'] = int(part.split(':')[1].strip())
                        elif 'Failures:' in part:
                            metrics['failures'] = int(part.split(':')[1].strip())
                        elif 'Errors:' in part:
                            metrics['errors'] = int(part.split(':')[1].strip())
                    break
        except Exception as e:
            print(f"Warning: Could not parse test metrics: {e}")

        return metrics

    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("\nRunning Performance Benchmarks...")
        print("-" * 40)

        try:
            benchmark_file = current_dir / 'tests' / 'performance_benchmarks.py'
            if not benchmark_file.exists():
                return {'error': 'Performance benchmark file not found'}

            print("  Executing comprehensive performance tests...")

            # Run performance benchmarks
            result = subprocess.run(
                [sys.executable, str(benchmark_file)],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for benchmarks
            )

            benchmark_results = {
                'success': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode
            }

            # Try to find generated report files
            report_files = list(self.reports_dir.glob('performance_benchmark_report_*.txt'))
            if not report_files:
                # Look in current directory
                report_files = list(Path('.').glob('performance_benchmark_report_*.txt'))

            if report_files:
                latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_report, 'r', encoding='utf-8') as f:
                        benchmark_results['report_content'] = f.read()
                    benchmark_results['report_file'] = str(latest_report)
                except Exception as e:
                    print(f"Warning: Could not read benchmark report: {e}")

            self.test_results['performance_benchmarks'] = benchmark_results
            return benchmark_results

        except subprocess.TimeoutExpired:
            return {'error': 'Performance benchmarks timed out', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def analyze_code_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity of refactored modules."""
        print("\nAnalyzing Code Complexity...")
        print("-" * 40)

        complexity_results = {}

        # Key files to analyze
        key_files = [
            'quant_system_full/bot/risk_calculation_services.py',
            'quant_system_full/bot/risk_assessment_orchestrator.py',
            'quant_system_full/bot/scoring_services.py',
            'quant_system_full/bot/scoring_orchestrator.py'
        ]

        for file_path in key_files:
            full_path = current_dir / file_path
            if full_path.exists():
                complexity_results[file_path] = self._analyze_file_complexity(full_path)

        self.test_results['code_complexity'] = complexity_results
        return complexity_results

    def _analyze_file_complexity(self, file_path: Path) -> Dict[str, Any]:
        """Analyze complexity of a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')

            # Basic metrics
            total_lines = len(lines)
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])

            # Count functions and classes
            functions = len([line for line in lines if line.strip().startswith('def ')])
            classes = len([line for line in lines if line.strip().startswith('class ')])

            # Estimate complexity based on control structures
            complexity_keywords = ['if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'with ']
            complexity_score = sum(content.count(keyword) for keyword in complexity_keywords)

            # Calculate average function length (rough estimate)
            avg_function_length = code_lines / functions if functions > 0 else 0

            # Find longest function (simplified analysis)
            longest_function = self._find_longest_function(lines)

            return {
                'total_lines': total_lines,
                'code_lines': code_lines,
                'comment_lines': comment_lines,
                'comment_ratio': comment_lines / total_lines if total_lines > 0 else 0,
                'functions': functions,
                'classes': classes,
                'complexity_score': complexity_score,
                'avg_function_length': avg_function_length,
                'longest_function_lines': longest_function,
                'complexity_per_line': complexity_score / code_lines if code_lines > 0 else 0,
                'meets_targets': {
                    'avg_function_length_under_50': avg_function_length < 50,
                    'longest_function_under_100': longest_function < 100,
                    'good_comment_ratio': comment_lines / total_lines > 0.1
                }
            }

        except Exception as e:
            return {'error': str(e)}

    def _find_longest_function(self, lines: List[str]) -> int:
        """Find the longest function in the file."""
        longest = 0
        current_function_length = 0
        in_function = False
        base_indent = 0

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('def '):
                if in_function:
                    longest = max(longest, current_function_length)
                in_function = True
                current_function_length = 1
                base_indent = len(line) - len(line.lstrip())
            elif in_function:
                if line.strip() == '' or line.strip().startswith('#'):
                    # Skip empty lines and comments
                    continue
                elif stripped.startswith(('def ', 'class ')) and len(line) - len(line.lstrip()) <= base_indent:
                    # New function or class at same or higher level
                    longest = max(longest, current_function_length)
                    if stripped.startswith('def '):
                        current_function_length = 1
                        base_indent = len(line) - len(line.lstrip())
                    else:
                        in_function = False
                else:
                    current_function_length += 1

        if in_function:
            longest = max(longest, current_function_length)

        return longest

    def check_import_structure(self) -> Dict[str, Any]:
        """Check import structure and dependencies."""
        print("\nChecking Import Structure...")
        print("-" * 40)

        import_results = {}

        # Test importing refactored modules
        modules_to_test = [
            'risk_calculation_services',
            'risk_assessment_orchestrator',
            'scoring_services',
            'scoring_orchestrator'
        ]

        for module_name in modules_to_test:
            try:
                exec(f'import {module_name}')
                import_results[module_name] = {'importable': True, 'error': None}
                print(f"  [OK] {module_name}")
            except Exception as e:
                import_results[module_name] = {'importable': False, 'error': str(e)}
                print(f"  [FAIL] {module_name}: {e}")

        self.test_results['import_structure'] = import_results
        return import_results

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive quality assessment report."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        report = []
        report.append("INVESTMENT-GRADE QUANTITATIVE TRADING SYSTEM")
        report.append("CODE QUALITY ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {end_time.isoformat()}")
        report.append(f"Duration: {duration.total_seconds():.1f} seconds")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score()
        report.append(f"Overall Quality Score: {quality_score:.1f}/10")
        report.append("")

        # Unit Test Results
        if 'unit_tests' in self.test_results:
            unit_results = self.test_results['unit_tests']
            report.append("UNIT TEST RESULTS")
            report.append("-" * 20)

            if 'summary' in unit_results:
                summary = unit_results['summary']
                report.append(f"Total Tests: {summary['total_tests']}")
                report.append(f"Success Rate: {summary['success_rate']:.1f}%")
                report.append(f"Failures: {summary['total_failures']}")
                report.append(f"Errors: {summary['total_errors']}")

            # Module breakdown
            if 'modules' in unit_results:
                report.append("\nModule Breakdown:")
                for module, results in unit_results['modules'].items():
                    status = "[OK]" if results.get('success', False) else "[FAIL]"
                    tests = results.get('tests_run', 0)
                    failures = results.get('failures', 0)
                    errors = results.get('errors', 0)
                    report.append(f"  {status} {module}: {tests} tests, {failures} failures, {errors} errors")
            report.append("")

        # Performance Benchmark Results
        if 'performance_benchmarks' in self.test_results:
            bench_results = self.test_results['performance_benchmarks']
            report.append("PERFORMANCE BENCHMARK RESULTS")
            report.append("-" * 20)

            if bench_results.get('success'):
                report.append("[OK] Performance benchmarks completed successfully")
                if 'report_file' in bench_results:
                    report.append(f"  Detailed report: {bench_results['report_file']}")
            else:
                report.append("[FAIL] Performance benchmarks failed")
                if 'error' in bench_results:
                    report.append(f"  Error: {bench_results['error']}")
            report.append("")

        # Code Complexity Analysis
        if 'code_complexity' in self.test_results:
            complexity_results = self.test_results['code_complexity']
            report.append("CODE COMPLEXITY ANALYSIS")
            report.append("-" * 20)

            total_functions = 0
            total_avg_length = 0
            files_analyzed = 0
            target_compliance = {'avg_length': 0, 'longest_func': 0, 'comment_ratio': 0}

            for file_path, metrics in complexity_results.items():
                if 'error' not in metrics:
                    files_analyzed += 1
                    total_functions += metrics['functions']
                    total_avg_length += metrics['avg_function_length']

                    meets_targets = metrics.get('meets_targets', {})
                    if meets_targets.get('avg_function_length_under_50'): target_compliance['avg_length'] += 1
                    if meets_targets.get('longest_function_under_100'): target_compliance['longest_func'] += 1
                    if meets_targets.get('good_comment_ratio'): target_compliance['comment_ratio'] += 1

                    report.append(f"  {Path(file_path).name}:")
                    report.append(f"    Functions: {metrics['functions']}")
                    report.append(f"    Avg Function Length: {metrics['avg_function_length']:.1f} lines")
                    report.append(f"    Longest Function: {metrics['longest_function_lines']} lines")
                    report.append(f"    Comment Ratio: {metrics['comment_ratio']:.1%}")

            if files_analyzed > 0:
                report.append(f"\nOverall Metrics:")
                report.append(f"  Files Analyzed: {files_analyzed}")
                report.append(f"  Total Functions: {total_functions}")
                report.append(f"  Average Function Length: {total_avg_length/files_analyzed:.1f} lines")
                report.append(f"\nTarget Compliance:")
                report.append(f"  Avg Length <50 lines: {target_compliance['avg_length']}/{files_analyzed} files")
                report.append(f"  Longest Function <100 lines: {target_compliance['longest_func']}/{files_analyzed} files")
                report.append(f"  Good Comment Ratio: {target_compliance['comment_ratio']}/{files_analyzed} files")
            report.append("")

        # Import Structure Results
        if 'import_structure' in self.test_results:
            import_results = self.test_results['import_structure']
            report.append("IMPORT STRUCTURE ANALYSIS")
            report.append("-" * 20)

            importable_count = sum(1 for result in import_results.values() if result['importable'])
            total_modules = len(import_results)

            report.append(f"Importable Modules: {importable_count}/{total_modules}")

            for module, result in import_results.items():
                status = "[OK]" if result['importable'] else "[FAIL]"
                report.append(f"  {status} {module}")
                if not result['importable'] and result['error']:
                    report.append(f"    Error: {result['error']}")
            report.append("")

        # Quality Recommendations
        report.append("QUALITY RECOMMENDATIONS")
        report.append("-" * 20)
        recommendations = self._generate_recommendations()
        for recommendation in recommendations:
            report.append(f"• {recommendation}")
        report.append("")

        # Next Steps
        report.append("NEXT STEPS")
        report.append("-" * 20)
        next_steps = self._generate_next_steps(quality_score)
        for step in next_steps:
            report.append(f"1. {step}")
        report.append("")

        return "\n".join(report)

    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score (0-10)."""
        score = 0.0
        max_score = 10.0

        # Unit test score (0-3 points)
        if 'unit_tests' in self.test_results:
            unit_results = self.test_results['unit_tests']
            if 'summary' in unit_results:
                success_rate = unit_results['summary']['success_rate']
                score += (success_rate / 100) * 3.0

        # Performance score (0-2 points)
        if 'performance_benchmarks' in self.test_results:
            bench_results = self.test_results['performance_benchmarks']
            if bench_results.get('success'):
                score += 2.0

        # Complexity score (0-3 points)
        if 'code_complexity' in self.test_results:
            complexity_results = self.test_results['code_complexity']
            compliant_files = 0
            total_files = 0

            for metrics in complexity_results.values():
                if 'meets_targets' in metrics:
                    total_files += 1
                    meets_targets = metrics['meets_targets']
                    if (meets_targets.get('avg_function_length_under_50', False) and
                        meets_targets.get('longest_function_under_100', False)):
                        compliant_files += 1

            if total_files > 0:
                score += (compliant_files / total_files) * 3.0

        # Import structure score (0-2 points)
        if 'import_structure' in self.test_results:
            import_results = self.test_results['import_structure']
            importable_count = sum(1 for result in import_results.values() if result['importable'])
            total_modules = len(import_results)
            if total_modules > 0:
                score += (importable_count / total_modules) * 2.0

        return min(score, max_score)

    def _generate_recommendations(self) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        # Based on unit tests
        if 'unit_tests' in self.test_results:
            unit_results = self.test_results['unit_tests']
            if 'summary' in unit_results:
                success_rate = unit_results['summary']['success_rate']
                if success_rate < 90:
                    recommendations.append("Improve unit test coverage and fix failing tests")
                elif success_rate < 95:
                    recommendations.append("Address remaining test failures for better reliability")

        # Based on complexity
        if 'code_complexity' in self.test_results:
            complexity_results = self.test_results['code_complexity']
            for file_path, metrics in complexity_results.items():
                if 'meets_targets' in metrics:
                    meets_targets = metrics['meets_targets']
                    if not meets_targets.get('avg_function_length_under_50', True):
                        recommendations.append(f"Refactor {Path(file_path).name} to reduce average function length")
                    if not meets_targets.get('longest_function_under_100', True):
                        recommendations.append(f"Break down longest function in {Path(file_path).name}")

        # Based on imports
        if 'import_structure' in self.test_results:
            import_results = self.test_results['import_structure']
            failed_imports = [module for module, result in import_results.items() if not result['importable']]
            if failed_imports:
                recommendations.append(f"Fix import issues in modules: {', '.join(failed_imports)}")

        # General recommendations
        if not recommendations:
            recommendations.append("Code quality is good - continue with integration testing")
            recommendations.append("Consider implementing additional edge case tests")
            recommendations.append("Add performance regression tests to CI/CD pipeline")

        return recommendations

    def _generate_next_steps(self, quality_score: float) -> List[str]:
        """Generate next steps based on quality score."""
        if quality_score >= 8.5:
            return [
                "Proceed with production deployment",
                "Implement monitoring and alerting",
                "Set up automated quality gates",
                "Document deployment procedures"
            ]
        elif quality_score >= 7.0:
            return [
                "Address critical quality issues identified above",
                "Increase test coverage to 80%+",
                "Complete performance optimization",
                "Re-run quality assessment"
            ]
        else:
            return [
                "Focus on fixing failing tests",
                "Refactor complex functions identified",
                "Improve code documentation",
                "Implement missing test cases",
                "Re-assess quality after improvements"
            ]

    def save_results(self):
        """Save test results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON results
        json_file = self.reports_dir / f'quality_test_results_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        # Save comprehensive report
        report_file = self.reports_dir / f'quality_assessment_report_{timestamp}.txt'
        report_content = self.generate_comprehensive_report()
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\nResults saved:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}")

        return str(report_file)

    def run_all_tests(self):
        """Run all quality tests."""
        print("Investment-Grade Quantitative Trading System")
        print("Code Quality Assessment")
        print("=" * 50)

        # Run all test phases
        self.check_import_structure()
        self.run_unit_tests()
        self.analyze_code_complexity()
        self.run_performance_benchmarks()

        # Generate and save reports
        report_file = self.save_results()

        # Print summary
        print("\n" + "=" * 50)
        print("QUALITY ASSESSMENT SUMMARY")
        print("=" * 50)

        quality_score = self._calculate_quality_score()
        print(f"Overall Quality Score: {quality_score:.1f}/10")

        if quality_score >= 8.5:
            print("[OK] EXCELLENT - Ready for production deployment")
        elif quality_score >= 7.0:
            print("⚠ GOOD - Some improvements needed")
        else:
            print("[FAIL] NEEDS WORK - Address critical issues")

        print(f"\nDetailed report: {report_file}")

        return quality_score


def main():
    """Main execution function."""
    runner = QualityTestRunner()
    quality_score = runner.run_all_tests()

    # Exit with appropriate code for CI/CD
    if quality_score >= 8.5:
        exit_code = 0  # Success
    elif quality_score >= 7.0:
        exit_code = 1  # Warning
    else:
        exit_code = 2  # Failure

    sys.exit(exit_code)


if __name__ == '__main__':
    main()