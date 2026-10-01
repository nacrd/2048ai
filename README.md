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

「达到目标块」优化最少有效移动步数；「提高得分」优化 H 步以内最高累计新增合并分。设置 H（正整数，没有 64 步上限）和整条路线预算（可选 30 秒），点击「规划路线」。节点预算可配置，界面默认 1000000；起点已达标时最短路线为零步。CPU 搜索和 TAS 使用显式栈，CUDA 步数使用 64 位整数；输入须在 JavaScript 安全整数范围内。

路线搜索分为三种：**混合**用约 70% 预算进行束搜索，剩余预算尝试证明；**束搜索**每层保留指定束宽的不同局面，适合长路线候选；**完整回溯**保留穷举及最优性证明。旧路线未指定策略时按完整回溯解释。

目标模式使用方块质量和最大指数增长的保守下界剪枝，固定 RNG 还利用动作无关的出块数值前缀。只有完整排除更短路线后才标「已证明最短」；得分完整搜索 H 步后才标「已证明最高分」。束搜索找到路线不等于证明最短，预算不足显示候选。若下界或穷举证明 H 步内无法达标会明确说明，同时允许展示用于继续推进的候选路线。结论限所选起点、出块条件及有限步目标。

工作台滑块预览路线，不修改当前局面。「执行路线」按移动间隔播放，「路线单步」执行一个动作；暂停保留路线，「回到路线起点」恢复棋盘、分数、步数及 RNG，并清空当前会话回放。每步检查实际状态，防止路线错用。修改棋盘、目标、算法或搜索条件会使路线失效。

「导出路线」生成可复制 / 下载的 JSON，粘贴后「加载路线文本」恢复起点及路线。导入逐步复算出块，不信任文件中的最优性声明。自动存档只保存当前游戏状态，TAS 路线需单独导出。

TAS 当前使用浏览器 CPU Worker，CUDA 继续用于 Rollout 批量模拟。新增 TAS 已完成编译与静态核验，游玩验收由使用者完成。

## 优化与长局续玩

4×4 非动画移动使用 5bit 行查表，正确支持 32768 → 65536 → 131072；指数≥31或其他尺寸回退通用规则。界面动画仍用通用引擎。新版评分比较八种蛇形、大块次序、空位、可合并关系和阶段权重，不硬性禁止移动大块。TypeScript、Numba、CUDA 采用相同评分和贪心后续策略。

目标 rollout 的成功率相同时使用下一步出块后的期望局面评分 `tieBreak`；无成功样本仍显示 0% 和真实置信区间。Worker 完成后保留复用，取消只销毁正在工作的实例；CUDA 批次复用输入、输出缓冲区及计时事件。

底层推理优化：浏览器 Rollout 在预算≥200ms、每方向≥128条轨迹时，按硬件并发数使用最多4个 Worker；小任务保留单 Worker。轨迹按全局编号拆分，结果仍按编号顺序累加；暂停终止所有活动 Worker。每个4×4 Worker查表约16MiB，首次启动和初始化也计入软预算。

本机 CPU Rollout 用 Numba 多核并行，单轨迹复用移动/评分缓冲区并释放 GIL；可在启动服务前设置 `NUMBA_NUM_THREADS` 限制线程数。CUDA 为2×2–6×6分别编译，减少动态索引和局部数组大小；目标达标与最后一步跳过多余贪心计算。CPU/CUDA 使用独立调度队列，批次根据设备并行度、上一批耗时及剩余预算自适应。常规安全整数得分及目标概率在 GPU 汇总，仅回传每个方向的总值；超出精度安全界限时回退原来的主机汇总。[Numba 线程配置](https://numba.readthedocs.io/en/stable/user/threading-layer.html) · [CuPy RawModule](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawModule.html)

GPU 初始化只编译内核，不自动运行游玩验证。健康接口显示CPU线程数和GPU SM数；求解响应新增排队耗时、批次数和批次大小，便于自行评测。TAS 缓存节点的最大块/达标状态，减少排序中的重复扫描；仍使用单 Worker。新版吞吐和设备利用率需要重新实测。

可在终端离线从存档分段规划并续玩，无需动画等待：

```powershell
npm run continue -- --save .\save.json --mode fixed --target 131072 --seconds 600 --width 128 --window 32 --commit 8 --out artifacts\fixed-run
# 理想出块对照使用 --mode ideal，并使用不同 --out 目录
```

fixed 保留原 RNG，ideal 只按已记录的合法 2/4 出块。每步核对完整快照，约每 10 秒保存 `continued-save.json`、`route.json`、`progress.json`；原文件不覆盖，输出冲突会拒绝。`--window` 是每轮规划长度，`--commit` 是本轮实际执行的前缀，结束时保留检查点。直接运行脚本收到 SIGINT 时会在当前窗口后保存；通过 npm 或终端强制结束进程时，以最后一次检查点为准。需要从检查点再次续玩时使用新的输出目录。

固定种子增加质量障碍证明：合并保持总质量，每次有效移动的出块数值仅取决于 RNG。若未来总质量的二进制 1 位数等于棋盘格数，则所有格必须为互不相同的二次幂，满盘后无法继续。TAS 显示这个全局不可达证明；离线续玩将有效目标调整为障碍前的最大块上界，在 `progress.json` 同时记录 requestedTarget/target，并保存 `fixed-limit.json`。检测超时仅表示未得到证明，上界可达性仍需实际路线确认。

标准 4×4 在有利出块条件下的最大块为 131072，[理论论文](https://arxiv.org/abs/1804.07393)。这不保证任意自定义存档或固定 RNG 可达。续玩报告只记录实际已达最大块，不把局部束搜索失败当作全局不可达证明，也不把达到最大块当作最短路线证明。

## 验证与评测

新增「强力 · 阶段与风险搜索」：自适应加深、阶段/空位风险、大块合并通道、对称有界缓存及 CPU 根方向并行。刷新页面后在算法中选择；旧模式仍可用。强力值是启发式估值，分析面板单列下一次出块即死风险。权重与实际胜率需要使用者评测，P2学习模型及GPU学习策略尚未接入。

成对旧版/新版质量对照：`npm run bench:ab -- --games 10 --split tuning --budget-mode nodes --nodes 10000 --horizon 4`。工具冻结并执行 `3c0a2c4` 的旧代码，保留随机流、失败轨迹与独立输出，区分终局/截断/异常；Node仅测串行CPU。[完整使用与口径](docs/WIN_RATE_GUIDE.md) · [全面胜率方案](docs/WIN_RATE_PLAN.md)。

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

或者运行 `scripts/verify.ps1`。没有可用 GPU 时 CUDA 测试跳过；正式 CUDA 验收要求实际 GPU 测试通过。已保存质量/性能报告来自优化前版本，不能直接代表新版胜率或加速比；自动 CUDA 阈值仍沿用旧校准，需使用者重测更新。新版策略的跨实现样例需先重新生成 `tests/cross-fixtures.ts` 的输出。性能对比使用相同策略和轨迹数，分别报告内核及 HTTP 总耗时。

结果文件位于 `benchmarks/`；完整说明见 [验收记录](docs/VALIDATION.md)。硬件上的实测收益只适用于记录的条件。

## 目录与来源

`src/core/` 纯规则与存档，`src/ai/` CPU 算法与 Worker，`src/main.ts` 界面控制，`server/` 本机 CUDA 服务。原版 `js/` 保留作为规则测试基准。

[初始项目计划](PROJECT_PLAN.md) · [上游来源](docs/UPSTREAM.md) · [MIT 许可证](LICENSE.txt)
