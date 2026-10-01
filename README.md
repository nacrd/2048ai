# 2048 AI · 棋盘实验室

基于 [gabrielecirulli/2048](https://github.com/gabrielecirulli/2048) 的 MIT 二开版本：自动游玩、自定义棋盘、TAS 离线路线规划、有限步求解和可选 CUDA 批量模拟。

## 运行

浏览器 CPU 版只需要 Node.js（本机验证为 26.3.1，Vite 要求 20.19+ / 22.12+）：

```powershell
npm ci
npm run dev
```

打开 http://127.0.0.1:5173/ 。CPU 版不需要 Python、CUDA 或显卡。

Windows 启动脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

本机 CUDA 版需要 NVIDIA 驱动、Python 3.12+（推荐 3.13，本机验证为 3.13.15）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
npm ci
npm run build
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8765
```

打开 http://127.0.0.1:8765/ 。也可执行 `powershell -ExecutionPolicy Bypass -File scripts/start.ps1 -Cuda -Install`；后续启动省略 `-Install`。安装 CUDA wheel 会下载较大的 NVIDIA 运行库，不修改全局 Python 环境。若没有 CUDA，浏览器 CPU 版继续可用。

开发时运行 Vite，并另开终端启动服务；Vite 将 `/api` 代理至 8765。生产增强版由服务同源提供 `dist/` 页面。服务绑定本机地址。

## 功能

- 2×2–6×6 棋盘；逐格编辑、粘贴矩阵、JSON 文件导入导出。
- 自动游玩、暂停、AI 单步、只分析、键盘/触屏接管。
- 种子复现、自动保存当前状态、撤销、当前会话回放；导出存档包含回放。
- Expectimax 限时搜索、CPU/CUDA rollout、有限步精确分析。
- 可配置目标块、计算预算、步数和模拟数；四方向结果和概率置信区间。
- TAS 固定种子与理想出块两种模式；整条路线规划、逐步预览、执行、暂停、恢复起点及 JSON 导入导出。

「棋盘尺寸」变更会打开编辑器，点击应用或新游戏才改变实际棋盘。自定义棋盘不额外生成初始块；每次有效移动仍按经典规则随机出块。输入 0 或 2 的幂，手工输入最高 2³⁰。

GPU 使用方法：检测 CUDA → 选择 Rollout → 选择 CUDA → 推荐 32 步、512 条以上轨迹。自动后端根据 `benchmarks/calibration.json` 的本机设备/工作负载阈值选择；没有匹配校准时使用 CPU。显式 CUDA 出错会在交互界面回退 CPU 并说明原因，benchmark 不静默回退。

经典随机游戏的建议需要在每次出块后重新计算。Expectimax 是启发式评分；rollout 是固定后续策略的统计估计。精确模式只保证给定 H 步目标，超出预算时明确未完成，不提供最优动作。详见 [规则与协议](docs/PROTOCOL.md)。

## TAS 使用

选择算法 **TAS · 离线路线规划**，从当前局面规划，在工作台选择出块条件：

- **固定种子**：读取当前存档的真实 RNG，提前计算出块，回溯不同动作。相同起点和操作可复现。
- **理想出块**：联合搜索移动、出块空位和 2/4，模拟最佳出块条件。执行时主动控制出块，不按经典概率抽样，不消耗游戏 RNG。普通 AI / 手动游玩恢复随机出块。

「达到目标块」优化最少有效移动步数；「提高得分」优化 H 步以内最高累计新增合并分。设置 H（1–64）和整条路线预算（可选 30 秒），点击「规划路线」。搜索还受 300000 节点上限限制；起点已达标时最短路线为零步。

规划器先生成候选，再进行带缓存的回溯。目标模式迭代加深，完整排除更短路线后才标「已证明最短」；得分模式完整搜索 H 步后才标「已证明最高分」。预算不足显示当前候选并标明未证明最优；完整穷举未达标会说明 H 步内无法达标。结论仅适用于所选起点、出块条件和有限步目标。

工作台滑块预览路线，不修改当前局面。「执行路线」按移动间隔播放，「路线单步」执行一个动作；暂停保留路线，「回到路线起点」恢复棋盘、分数、步数及 RNG，并清空当前会话回放。每步检查实际状态，防止路线错用。修改棋盘、目标、算法或搜索条件会使路线失效。

「导出路线」生成可复制 / 下载的 JSON，粘贴后「加载路线文本」恢复起点及路线。导入逐步复算出块，不信任文件中的最优性声明。自动存档只保存当前游戏状态，TAS 路线需单独导出。

TAS 当前使用浏览器 CPU Worker，CUDA 继续用于 Rollout 批量模拟。新增 TAS 已完成编译与静态核验，游玩验收由使用者完成。

## 验证与评测

```powershell
npm run build
npm test
npx tsx tests/cross-fixtures.ts
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/test_server.py -q
npm run bench -- 100
# 服务已运行时：
.\.venv\Scripts\python.exe -m benchmarks.gpu
```

或者运行 `scripts/verify.ps1`。没有可用 GPU 时 CUDA 测试跳过；正式 CUDA 验收要求实际 GPU 测试通过。质量基线固定 10000 节点/3 层搜索，使用独立评测种子，与界面默认 200 ms / 250000 节点不同。性能评测对比同一策略和轨迹数的 Numba 并行 CPU 与 CUDA，并单独报告内核及 HTTP 总耗时。

结果文件位于 `benchmarks/`；完整说明见 [验收记录](docs/VALIDATION.md)。硬件上的实测收益只适用于记录的条件。

## 目录与来源

`src/core/` 纯规则与存档，`src/ai/` CPU 算法与 Worker，`src/main.ts` 界面控制，`server/` 本机 CUDA 服务。原版 `js/` 保留作为规则测试基准。

[初始项目计划](PROJECT_PLAN.md) · [上游来源](docs/UPSTREAM.md) · [MIT 许可证](LICENSE.txt)
