# 规则与求解协议 v1

棋盘对外采用行优先数字矩阵；0 为空格，其他数字为 2 的幂。内部存储指数数组。方向 0/1/2/3 分别为上/右/下/左，维度 2–6。手工导入支持最大 2³⁰；保存和后端支持合法合并产生的大方块，最高 2⁵²。

经典规则：每块每步最多合并一次，分数加上合并后的块值；只在有效移动后出块，空位均匀随机，2/4 概率为 90%/10%。新局生成两块；导入棋盘不生成初始块。RNG 使用 xorshift32，种子字符串使用 FNV-1a 派生；随机抽样顺序与上游一致（先值，再列优先空位）。真实游戏和搜索模拟的种子独立。

浏览器 CPU：Worker 接受 `{id, board:{size,cells}, options}`，返回 `{id,result}` 或 `{id,error}`。完成后保留闲置 Worker，暂停或状态变更销毁活动 Worker，旧请求结果通过版本号和实例身份拒绝。CUDA 服务只负责模拟，复用批次缓冲区，通过 HTTP 提交，按批次查询断连并取消；单个已启动 kernel 运行至该批结束。4×4 非动画规则使用五位指数行查表；指数≥31或其他尺寸走通用实现，避免 65536/131072 截断。

浏览器 Rollout 大任务使用最多4路轨迹 Worker，任务增加 `batch:{offset,count,includeTies}`；返回全局连续编号轨迹的方向优先 Float64Array，并以 transferable 交回缓冲区。主线程按轨迹编号顺序累加，每方向相同样本数；共享整个请求软预算，完成一轮全部批次后检查。结果可含 workers 并行路数。取消终止所有活动实例并拒绝待处理批次，闲置实例继续复用。Expectimax/exact/TAS 仍使用单 Worker。

`GET /api/health` 返回 available、设备名、显存、不可用原因和本机校准配置。`POST /api/solve` 请求示例：

```json
{"board":[[2,2],[0,4]],"algorithm":"rollout","backend":"cuda","objective":"target","target":16,"budgetMs":200,"horizon":16,"trajectories":256,"seed":71231,"requestId":"12"}
```

仅支持 rollout；backend 为 cuda/cpu，objective 为 score/target。horizon 为正的安全整数（≤2⁵³−1），没有 64 步业务上限；CUDA 参数和循环计数为无符号 64 位整数。每方向最多 65536 条轨迹、10000 ms 预算。响应包含 direction、choices、algorithm、backend、complete、depth、nodes、elapsedMs、kernelMs、requestId、note。每个 choice 包含 direction、value、samples，目标模式另有 95% Wilson 区间 confidence。

服务 CPU/CUDA 各自串行调度，同一CPU请求由 Numba prange 多核运行。两种后端可以并发；取消中的原生计算完成前保留后端锁，防止缓冲区被新请求覆盖。初始批次依据 CPU 线程数/GPU SM 与128线程块配置，后续按实测批次耗时调整，目标每批≤20ms但不保证硬上限。响应增加 queueMs（后端排队）、batches、batchSize（最后一批每方向轨迹数）、parallelism（CPU线程数或GPU SM数）。健康接口增加 cpuThreads/cudaMultiprocessors/cudaBlockSize。

CUDA 按 N=2…6 分别编译，保留同一策略、方向平局顺序与随机映射。HTTP 用设备端 float64 汇总：目标结果为0/1；得分仅在 `(offset+count)*H*(初始质量+4H)≤2^53` 时启用，保证整数累加不因归约顺序变化。其他情况沿用完整矩阵回传及 NumPy 汇总。直接 CudaEngine.batch 仍返回完整轨迹矩阵供验收使用；kernelMs 只计 rollout 内核，设备归约和拷贝计入 elapsedMs。

score 模拟值为 H 步累计新增合并分数；target 模拟值为 H 步内达到目标的样本比例。后续策略使用同一固定启发式贪心算法；它们是该策略的估计，不是最优策略的胜率。每个方向使用独立派生种子，轨迹编号在批次间连续。

增强评分比较八个蛇形方向、指数平方单调性、加权可合并性、随最大指数变化的空位权重；TS/Numba/CUDA 同步。模拟 choice 增加 tieBreak，为按经典出块概率计算的一步后期望启发式评分，仅在 value 相同的情况下参与动作选择，不改变 value 或 confidence。全零成功率明确提示使用辅助评分。

有限步精确搜索在浏览器 Worker 执行：动作节点取最大值，所有空位及 2/4 出块分支完整展开，优化 H 步内期望新增分数或达标概率。maxNodes 与时间预算可配置，界面默认 1000000 节点；未完成时 direction=null、choices=[]、complete=false。完整展开后的浮点比较容差为 1e-9。已达目标时 target 的值为 1。精确范围不覆盖无限步完整游戏。

Expectimax 为迭代加深的启发式搜索，保留最后一个完整搜索层；深度 0 表示只提供合法动作的启发式回退。其 value 是综合评分，不能解释为期望真实得分或概率。

Strong 新增 `algorithm:"strong"`，只用 CPU。options 可含 `continueAfterTarget` 与 `strong` 调参对象，具体字段见 [P0/P1 指南](WIN_RATE_GUIDE.md)。阶段/风险/通道评分只影响新模式，原有算法与 TAS 的定义保持不变。target 的未达标叶子值低于1、达标值为1，没有概率校准；score 的叶子为归一化启发式，加上缩放合并奖赏。两者均返回 `valueKind:"heuristic"` 和 `policyVersion:"strong-p1-v1"`，不解释为概率或真实分数。

Strong 按请求总预算迭代加深，仅使用全根共同完成的层；`complete` 表示已达到输入深度上限或无合法根，不是全局最优证明。自适应停止也可能 complete=false，以 `strongStats.stopReason` 区分。stats 含 profile、cacheHits、cacheEntries、prunedEdges、rootRisks；deadProbability 为下一次随机出块即无合法动作的精确枚举值，mobility 为期望剩余空位数。缓存仅保存完成子树，按值的 D4 对称性归并；不缓存原坐标动作。reuseCache 控制跨根/请求缓存保留，容量上限默认每 Worker 100000 条。

并行 Worker 消息可含 `strong:{prepare:true}` 或 `strong:{direction,depth,deadline}`；deadline 是统一的 Unix 毫秒时间，在 Worker 内转换为剩余 performance 预算。主线程公平分配每根节点配额，回收未用额度一次，保留相同深度的完整结果；总访问节点不超配置。取消与既有版本隔离相同。并行 stats.cacheEntries 报最近根任务观测到的最大单 Worker 缓存条数，非共享缓存总量。阈值>0会以叶子值替代低概率出块边并保留权重，prunedEdges 为替代边的访问次数。

预算是软上限：CPU 搜索周期检查、CPU rollout 完成一轮、CUDA 完成一个批次后检查。首次编译/Worker 启动/排队可能超过预算，不能保证硬实时。界面耗时包含浏览器到结果的开销；服务 elapsedMs 包含排队、计算、拷贝和统计，kernelMs 单独记录。nodes 在 rollout 模式是轨迹数×H 的步数预算，并非实际完成步数。

存档格式：version=1、rules=classic-90-10、board（矩阵）、score、rngState、target、moves，可选 history 记录历史快照。自动存档只保存当前状态；导出文件包含本次会话回放。撤销恢复分数和 RNG。输入 seed 用于新游戏；普通 AI 不读取未来 RNG，TAS 固定种子模式允许读取。

## TAS 协议

独立 Worker 接收 `{id,snapshot,options}`。snapshot 包含完整起点和真实 rngState。options 为 `{mode:"fixed"|"ideal",objective:"score"|"target",target,horizon,budgetMs,maxNodes,strategy?,beamWidth?}`。strategy 为 exact/beam/hybrid，省略按 exact；beamWidth=1–4096，默认128。H 和 maxNodes 是正安全整数；界面最多30000 ms、默认1000000节点。完成后保留 Worker，取消销毁活动 Worker 并使版本失效。过程中可返回 `{id,progress:{depth,nodes,maxTile}}`；最终仍返回result/error。TAS 和 CPU 概率树使用显式栈。

固定模式复用 session.advance，动作记录实际出块。理想模式展开每个合法动作后全部空位及值 2/4，动作和出块节点都取最大收益，不进行概率平均；执行时 RNG 保持不变。普通玩法从理想路线结束后的棋盘及保留的 RNG 继续随机游玩。

固定 RNG 目标搜索还检查质量障碍。出块数值序列与动作无关；质量的二进制 1 位数是表示它所需的最少方块数。等于格数时只能满盘摆放互不相同的方块，因此没有下一步。若障碍质量小于目标，构成该起点/RNG 的全局不可达证明，complete=true；仍可返回用于提高最大块的候选。检测预算不足返回未知，不声称不可达。离线 CLI 单独记录请求目标和调整后的最大块上界，达到上界不等于证明最短路线。

返回 TasPlan：options、frames（起点加逐步快照）、actions、complete、provenDepth、nodes、elapsedMs、note。action 为 `{direction,spawn:{index,value}|null}`；index 行优先零基，value=2/4。beam 按每层棋盘+RNG去重并保留束宽；hybrid 给beam约70%时间/节点预算，再完整回溯。束剪枝不构成最优性证明。目标使用质量和每步最大指数增长下界，fixed 再利用出块值前缀；下界可证明短深度不可能。provenDepth 为已排除短深度，达标完成时为证明的最少步数。得分完整展开≤H，按棋盘/RNG/剩余深度缓存最高累计分；未完成不声称最高分。节点数包含生成与访问，预算为软限制。

路线 JSON：`{version:1,type:"2048-tas",start:<游戏存档>,options,actions}`。导入校验起点、参数、动作数≤H 并逐步复算；固定出块必须匹配 RNG，理想出块必须是合法空位上的 2/4。导入后 complete=false，不能携带伪造的最优性声明。执行逐帧比较棋盘、分数、步数、目标和 RNG，不匹配即停止。预览只读，恢复起点重置当前历史。
