#!/usr/bin/env python3
"""
启动脚本：专业交易平台中心
================================

快速启动Agent D1设计的5,700+资产交易界面系统

使用方法:
    python run_trading_hub.py                    # 启动主交易中心
    python run_trading_hub.py --module advanced  # 直接启动高级交易界面
    python run_trading_hub.py --module futures   # 直接启动期货交易界面
    python run_trading_hub.py --module realtime  # 直接启动实时监控
    python run_trading_hub.py --module ai        # 直接启动AI学习中心
    python run_trading_hub.py --port 8502        # 自定义端口

Author: Agent D1 - Interface Optimization Specialist
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='启动专业交易平台界面系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模块说明:
  hub         主交易中心 (默认)
  advanced    高级多资产交易平台
  futures     专业期货交易界面
  realtime    增强实时监控系统
  ai          AI学习进度中心
  original    原始仪表盘

示例:
  python run_trading_hub.py
  python run_trading_hub.py --module futures --port 8502
  python run_trading_hub.py --module realtime --debug
        """
    )
    
    parser.add_argument(
        '--module', '-m',
        choices=['hub', 'advanced', 'futures', 'realtime', 'ai', 'original'],
        default='hub',
        help='要启动的模块 (默认: hub)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8501,
        help='端口号 (默认: 8501)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='不自动打开浏览器'
    )
    
    args = parser.parse_args()
    
    # 模块文件映射
    module_files = {
        'hub': 'main_trading_hub.py',
        'advanced': 'advanced_trading_interface.py', 
        'futures': 'futures_trading_interface.py',
        'realtime': 'enhanced_realtime_monitor.py',
        'ai': 'ai_learning_monitor.py',
        'original': 'streamlit_app.py'
    }
    
    # 获取当前目录
    current_dir = Path(__file__).parent
    target_file = current_dir / module_files[args.module]
    
    if not target_file.exists():
        print(f"错误: 文件 {target_file} 不存在")
        print(f"可用模块: {', '.join(module_files.keys())}")
        sys.exit(1)
    
    # 构建Streamlit命令
    cmd = [
        'streamlit', 'run', str(target_file),
        '--server.port', str(args.port),
        '--server.headless', 'true' if args.no_browser else 'false',
        '--theme.primaryColor', '#667eea',
        '--theme.backgroundColor', '#ffffff',
        '--theme.secondaryBackgroundColor', '#f0f2f6',
        '--theme.textColor', '#262730'
    ]
    
    if args.debug:
        cmd.extend(['--logger.level', 'debug'])
    
    # 显示启动信息
    print("=" * 60)
    print("🚀 Agent D1 专业交易平台启动中...")
    print("=" * 60)
    print(f"模块: {args.module.title()}")
    print(f"文件: {target_file.name}")
    print(f"端口: {args.port}")
    print(f"地址: http://localhost:{args.port}")
    print("=" * 60)
    
    if args.module == 'hub':
        print("🎛️ 主交易中心功能:")
        print("  • 5,700+ 多资产交易平台")
        print("  • 专业期货交易界面")
        print("  • 增强实时监控系统")
        print("  • AI学习进度跟踪")
        print("  • 统一导航和用户体验")
    elif args.module == 'advanced':
        print("📊 高级交易平台功能:")
        print("  • 5,700+ 资产实时监控")
        print("  • 多维度资产筛选")
        print("  • 动态性能热力图")
        print("  • 智能分页和搜索")
        print("  • 响应式界面设计")
    elif args.module == 'futures':
        print("📈 期货交易平台功能:")
        print("  • 专业合约规格管理")
        print("  • 实时保证金监控")
        print("  • 价差套利分析")
        print("  • 自动展期提醒")
        print("  • 期现套利机会")
    elif args.module == 'realtime':
        print("🔴 实时监控功能:")
        print("  • 5,700+ 资产实时流")
        print("  • 系统性能监控")
        print("  • AI模型进度跟踪")
        print("  • 高级风险预警")
        print("  • 相关性分析")
    elif args.module == 'ai':
        print("🤖 AI学习中心功能:")
        print("  • 模型训练进度可视化")
        print("  • 超参数优化结果")
        print("  • 策略性能演化")
        print("  • 神经网络架构图")
        print("  • 特征重要性分析")
    
    print("=" * 60)
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    try:
        # 启动Streamlit应用
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 服务器已停止")
        print("感谢使用Agent D1专业交易平台!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 启动失败: {e}")
        print("请检查Streamlit是否已安装: pip install streamlit")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ 未找到streamlit命令")
        print("请先安装Streamlit: pip install streamlit")
        sys.exit(1)

if __name__ == '__main__':
    main()