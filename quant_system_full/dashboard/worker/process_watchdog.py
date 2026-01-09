#!/usr/bin/env python3
"""
Process Monitoring Watchdog
Continuously monitors for multiple runner.py processes and terminates duplicates
"""

import os
import sys
import time
import subprocess
import logging
from typing import List, Dict
from pathlib import Path

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WATCHDOG] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('watchdog.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProcessWatchdog:
    def __init__(self, process_name: str = "runner.py", max_instances: int = 1, check_interval: int = 10):
        self.process_name = process_name
        self.max_instances = max_instances
        self.check_interval = check_interval
        self.allowed_pids = set()
        
    def get_runner_processes(self) -> List[Dict]:
        """Get detailed information about existing runner.py processes."""
        runner_processes = []
        
        try:
            if sys.platform == "win32":
                # Use WMI for comprehensive process information
                try:
                    cmd = ['cmd', '/c', 'wmic', 'process', 'where', 'name="python.exe"', 'get', 
                           'ProcessId,CommandLine,CreationDate', '/format:csv']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines[1:]:  # Skip header
                            if line.strip() and 'runner.py' in line.lower():
                                parts = line.split(',')
                                if len(parts) >= 3:
                                    try:
                                        pid = int(parts[2])  # ProcessId is usually the 3rd column
                                        cmdline = parts[1] if len(parts) > 1 else "Unknown"
                                        runner_processes.append({
                                            'pid': pid,
                                            'cmdline': cmdline,
                                            'method': 'WMI'
                                        })
                                    except ValueError:
                                        pass
                except Exception as e:
                    logger.warning(f"WMI query failed: {e}")
                
                # Fallback to tasklist with detailed format
                try:
                    result = subprocess.run(['cmd', '/c', 'tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/V'], 
                                          capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines[1:]:  # Skip header
                            if 'runner.py' in line.lower():
                                # Parse CSV format: "Image Name","PID","Session Name",...
                                parts = [p.strip('"') for p in line.split('","')]
                                if len(parts) >= 2:
                                    try:
                                        pid = int(parts[1])
                                        # Check if not already found
                                        if not any(p['pid'] == pid for p in runner_processes):
                                            runner_processes.append({
                                                'pid': pid,
                                                'cmdline': line,
                                                'method': 'tasklist'
                                            })
                                    except ValueError:
                                        pass
                except Exception as e:
                    logger.warning(f"tasklist query failed: {e}")
            else:
                # Linux/Mac: Use pgrep and ps
                try:
                    result = subprocess.run(['pgrep', '-f', 'runner.py'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        pids = result.stdout.strip().split('\n')
                        for pid_str in pids:
                            if pid_str.strip():
                                try:
                                    pid = int(pid_str.strip())
                                    # Get command line
                                    ps_result = subprocess.run(['ps', '-p', str(pid), '-o', 'cmd='], 
                                                             capture_output=True, text=True)
                                    cmdline = ps_result.stdout.strip() if ps_result.returncode == 0 else "Unknown"
                                    
                                    runner_processes.append({
                                        'pid': pid,
                                        'cmdline': cmdline,
                                        'method': 'pgrep'
                                    })
                                except ValueError:
                                    pass
                except Exception as e:
                    logger.warning(f"pgrep query failed: {e}")
        
        except Exception as e:
            logger.error(f"Failed to get process information: {e}")
        
        return runner_processes
    
    def terminate_process(self, pid: int, method: str = "unknown") -> bool:
        """Terminate a process by PID."""
        logger.info(f"Terminating PID {pid} (detected via {method})")
        
        try:
            if sys.platform == "win32":
                # Try graceful termination first
                result = subprocess.run(['cmd', '/c', 'taskkill', '/PID', str(pid)], 
                                      capture_output=True, text=True, timeout=5, encoding='utf-8', errors='replace')
                
                if result.returncode != 0:
                    # Force termination
                    result = subprocess.run(['cmd', '/c', 'taskkill', '/F', '/PID', str(pid)], 
                                          capture_output=True, text=True, timeout=5, encoding='utf-8', errors='replace')
                
                if result.returncode == 0:
                    logger.info(f"Successfully terminated PID {pid}")
                    return True
                else:
                    logger.warning(f"Failed to terminate PID {pid}: {result.stderr}")
                    return False
            else:
                # Unix/Linux: Try TERM first, then KILL
                try:
                    subprocess.run(['kill', '-TERM', str(pid)], timeout=5)
                    time.sleep(1)
                    
                    # Check if still running
                    check = subprocess.run(['kill', '-0', str(pid)], capture_output=True)
                    if check.returncode == 0:
                        # Still running, use KILL
                        subprocess.run(['kill', '-KILL', str(pid)], timeout=5)
                    
                    logger.info(f"Successfully terminated PID {pid}")
                    return True
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout terminating PID {pid}")
                    return False
                except Exception as e:
                    logger.warning(f"Failed to terminate PID {pid}: {e}")
                    return False
        
        except Exception as e:
            logger.error(f"Exception terminating PID {pid}: {e}")
            return False
    
    def cleanup_lock_files(self):
        """Clean up any stale lock files."""
        worker_dir = Path(__file__).parent
        
        lock_files = [
            worker_dir / "runner.pid",
            worker_dir / "runner.lock"
        ]
        
        cleaned = 0
        for lock_file in lock_files:
            try:
                if lock_file.exists():
                    lock_file.unlink()
                    logger.info(f"Removed stale lock file: {lock_file}")
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Could not remove {lock_file}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} lock file(s)")
    
    def monitor(self):
        """Main monitoring loop."""
        logger.info(f"Starting process watchdog for {self.process_name}")
        logger.info(f"Max instances: {self.max_instances}, Check interval: {self.check_interval}s")
        
        while True:
            try:
                processes = self.get_runner_processes()
                
                if len(processes) > self.max_instances:
                    logger.warning(f"Found {len(processes)} runner.py processes (max allowed: {self.max_instances})")
                    
                    # Sort by PID (older processes first)
                    processes.sort(key=lambda x: x['pid'])
                    
                    # Keep the first process, terminate the rest
                    processes_to_keep = processes[:self.max_instances]
                    processes_to_terminate = processes[self.max_instances:]
                    
                    logger.info(f"Keeping PID {processes_to_keep[0]['pid']} (oldest)")
                    
                    terminated_count = 0
                    for proc in processes_to_terminate:
                        if self.terminate_process(proc['pid'], proc['method']):
                            terminated_count += 1
                    
                    # Clean up lock files after termination
                    if terminated_count > 0:
                        time.sleep(2)  # Wait for processes to fully terminate
                        self.cleanup_lock_files()
                        logger.info(f"Terminated {terminated_count} duplicate processes")
                
                elif len(processes) == 1:
                    logger.debug(f"Single runner.py process found: PID {processes[0]['pid']}")
                elif len(processes) == 0:
                    logger.debug("No runner.py processes found")
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Watchdog stopped by user")
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(self.check_interval)

def main():
    """Main function."""
    watchdog = ProcessWatchdog(
        process_name="runner.py",
        max_instances=1,
        check_interval=15  # Check every 15 seconds
    )
    
    try:
        watchdog.monitor()
    except KeyboardInterrupt:
        logger.info("Watchdog shutdown")
        sys.exit(0)

if __name__ == "__main__":
    main()