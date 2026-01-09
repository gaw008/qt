  🚀 最新系统启动方法

  方法1: 一键启动完整系统（推荐）⭐

  python start_all.py

  启动内容：
  - ✅ React 前端 (http://localhost:3000+)
  - ✅ FastAPI 后端 (http://localhost:8000)
  - ✅ 交易机器人 (Worker)
  - ✅ WebSocket 实时更新
  - ✅ 系统健康监控
  - ✅ 所有任务调度器

  这是最简单的方式，所有组件一起启动，协调运行。

  ---
  方法2: 只运行交易机器人（独立模式）

  cd dashboard/worker
  python runner.py

  启动内容：
  - ✅ 交易执行引擎
  - ✅ 选股任务（每3小时）
  - ✅ 交易任务（每15秒）
  - ✅ 监控任务（每60秒）
  - ✅ AI训练任务（每日）
  - ⚠️ 但没有Web界面

  适用场景：
  - 只需要后台自动交易
  - 不需要查看仪表盘
  - 服务器上无头运行

  ---
  方法3: 独立启动各组件（手动控制）

  如果您想分别启动各组件：

  1. 启动后端API

  cd dashboard/backend
  python -m uvicorn app:app --host 0.0.0.0 --port 8000

  2. 启动Worker（交易机器人）

  cd dashboard/worker
  python runner.py

  3. 启动React前端

  cd UI
  npm install  # 首次运行需要
  npm run dev

  4. 启动Streamlit管理界面（可选）

  cd dashboard/frontend
  streamlit run streamlit_app.py --server.port 8501

  ---
  🎯 推荐启动流程

  首次启动（从头开始）

  # 1. 确认虚拟环境已激活
  .venv\Scripts\activate  # Windows
  # source .venv/bin/activate  # Linux/Mac

  # 2. 确认依赖已安装
  pip install -r bot/requirements.txt

  # 3. 检查配置文件
  # 确保 .env 文件存在且配置正确
  # 特别是 USE_IMPROVED_STRATEGIES=true

  # 4. 一键启动完整系统
  python start_all.py

  日常启动（已配置好）

  # 直接运行
  python start_all.py

  # 或者只启动交易机器人
  cd dashboard/worker
  python runner.py

  ---
  📊 启动后验证

  检查系统是否正常运行

  # 方法1: 访问Web界面
  # React前端: http://localhost:3000 (或3001, 5173)
  # Streamlit: http://localhost:8501

  # 方法2: 查看状态文件
  cat dashboard/state/status.json | python -m json.tool | head -50

  # 方法3: 查看日志
  tail -f dashboard/state/bot.log

  # 方法4: 启动监控仪表盘（刚创建的）
  streamlit run strategy_monitor_dashboard.py --server.port 8503
  # 访问: http://localhost:8503

  ---
  ⚙️ 当前系统配置

  根据您的 .env 文件：

  | 配置项                     | 当前值    | 说明                 |
  |-------------------------|--------|--------------------|
  | USE_IMPROVED_STRATEGIES | true ✅ | 改进策略已启用            |
  | DRY_RUN                 | false  | 真实交易模式             |
  | DATA_SOURCE             | auto   | 自动数据源（Tiger→Yahoo） |
  | SELECTION_TASK_INTERVAL | 10800秒 | 每3小时选股             |
  | TRADING_TASK_INTERVAL   | 15秒    | 每15秒检查交易           |
  | PRIMARY_MARKET          | US     | 美国市场               |

  ---
  🔍 启动选项对比

  | 启动方式         | Web界面 | 交易执行 | 监控  | 适用场景     |
  |--------------|-------|------|-----|----------|
  | start_all.py | ✅     | ✅    | ✅   | 日常使用（推荐） |
  | runner.py    | ❌     | ✅    | ✅   | 服务器后台运行  |
  | 手动启动各组件      | ✅     | ✅    | ✅   | 开发调试     |

  ---
  💡 常见问题

  Q: start_all.py 和 runner.py 有什么区别？

  - start_all.py:
    - 启动完整生态系统（前端+后端+worker）
    - 有Web界面可查看
    - 协调管理所有组件
  - runner.py:
    - 只启动交易核心（worker进程）
    - 无Web界面
    - 适合后台运行

  Q: 如果只想看仪表盘不交易？

  # 设置为模拟模式
  # 修改 .env: DRY_RUN=true

  # 然后启动
  python start_all.py

  Q: 如何停止系统？

  # start_all.py 启动的：
  # 按 Ctrl+C 优雅关闭

  # runner.py 启动的：
  # 按 Ctrl+C 或
  taskkill /F /IM python.exe  # Windows（慎用，会杀所有Python进程）
  pkill -f runner.py          # Linux/Mac

  ---
  🎯 总结：推荐您的启动方式

  根据您的使用场景：

  场景1: 日常交易（有Web界面）

  python start_all.py
  优点：完整功能，可视化界面，实时监控

  场景2: 服务器后台运行（无界面）

  cd dashboard/worker
  python runner.py &  # 后台运行
  优点：资源占用少，稳定持久运行

  场景3: 监控策略表现（新功能）

  # 主系统运行中，另开终端：
  streamlit run strategy_monitor_dashboard.py --server.port 8503
  # 访问: http://localhost:8503
  优点：专门监控改进策略V2的表现

  ---
  简单来说：
  - ✅ 有界面需求 → python start_all.py
  - ✅ 只要交易 → cd dashboard/worker && python runner.py
  - ✅ 监控策略 → streamlit run strategy_monitor_dashboard.py --server.port 8503

  现在的 USE_IMPROVED_STRATEGIES=true 已经启用，无论用哪种方式启动都会使用改进策略！🎉