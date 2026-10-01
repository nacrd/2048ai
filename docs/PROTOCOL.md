# 规则与求解协议 v1

棋盘对外采用行优先数字矩阵；0 为空格，其他数字为 2 的幂。内部存储指数数组。方向 0/1/2/3 分别为上/右/下/左，维度 2–6。手工导入支持最大 2³⁰；保存和后端支持合法合并产生的大方块，最高 2⁵²。

经典规则：每块每步最多合并一次，分数加上合并后的块值；只在有效移动后出块，空位均匀随机，2/4 概率为 90%/10%。新局生成两块；导入棋盘不生成初始块。RNG 使用 xorshift32，种子字符串使用 FNV-1a 派生；随机抽样顺序与上游一致（先值，再列优先空位）。真实游戏和搜索模拟的种子独立。

浏览器 CPU：Worker 接受 `{id, board:{size,cells}, options}`，返回 `{id,result}` 或 `{id,error}`。暂停和状态变更销毁 Worker，旧请求结果通过版本号拒绝。CUDA 服务只负责模拟，通过 HTTP 提交，按批次查询断连并取消；单个已启动 kernel 运行至该批结束。

`GET /api/health` 返回 available、设备名、显存、不可用原因和本机校准配置。`POST /api/solve` 请求示例：

```json
{"board":[[2,2],[0,4]],"algorithm":"rollout","backend":"cuda","objective":"target","target":16,"budgetMs":200,"horizon":16,"trajectories":256,"seed":71231,"requestId":"12"}
```

仅支持 rollout；backend 为 cuda/cpu，objective 为 score/target。最大 64 步、每方向 65536 条轨迹、10000 ms 预算。响应包含 direction、choices、algorithm、backend、complete、depth、nodes、elapsedMs、kernelMs、requestId、note。每个 choice 包含 direction、value、samples，目标模式另有 95% Wilson 区间 confidence。

score 模拟值为 H 步累计新增合并分数；target 模拟值为 H 步内达到目标的样本比例。后续策略使用同一固定启发式贪心算法；它们是该策略的估计，不是最优策略的胜率。每个方向使用独立派生种子，轨迹编号在批次间连续。

有限步精确搜索在浏览器 Worker 执行：动作节点取最大值，所有空位及 2/4 出块分支完整展开，优化 H 步内期望新增分数或达标概率。节点上限 250000 和时间预算；未完成时 direction=null、choices=[]、complete=false。完整展开后的浮点比较容差为 1e-9。已达目标时 target 的值为 1。精确范围不覆盖无限步完整游戏。

Expectimax 为迭代加深的启发式搜索，保留最后一个完整搜索层；深度 0 表示只提供合法动作的启发式回退。其 value 是综合评分，不能解释为期望真实得分或概率。

预算是软上限：CPU 搜索周期检查、CPU rollout 完成一轮、CUDA 完成一个批次后检查。首次编译/Worker 启动/排队可能超过预算，不能保证硬实时。界面耗时包含浏览器到结果的开销；服务 elapsedMs 包含排队、计算、拷贝和统计，kernelMs 单独记录。nodes 在 rollout 模式是轨迹数×H 的步数预算，并非实际完成步数。

存档格式：version=1、rules=classic-90-10、board（矩阵）、score、rngState、target、moves，可选 history 记录历史快照。自动存档只保存当前状态；导出文件包含本次会话回放。撤销恢复分数和 RNG。输入 seed 用于新游戏；普通 AI 不读取未来 RNG，TAS 固定种子模式允许读取。

## TAS 协议

独立 Worker 接收 `{id,snapshot,options}`。snapshot 包含完整起点和真实 rngState。options 为 `{mode:"fixed"|"ideal",objective:"score"|"target",target,horizon,budgetMs,maxNodes}`。界面 H=1–64、最大预算 30000 ms、maxNodes=300000。取消销毁 Worker 并使版本失效。

固定模式复用 session.advance，动作记录实际出块。理想模式展开每个合法动作后全部空位及值 2/4，动作和出块节点都取最大收益，不进行概率平均；执行时 RNG 保持不变。普通玩法从理想路线结束后的棋盘及保留的 RNG 继续随机游玩。

返回 TasPlan：options、frames（起点加逐步快照）、actions、complete、provenDepth、nodes、elapsedMs、note。action 为 `{direction,spawn:{index,value}|null}`；index 行优先零基，value=2/4。目标搜索迭代加深，中断时 provenDepth 表示完整排除的短路线深度，成功时表示证明的最少步数。得分完整展开 ≤H，按棋盘/RNG/剩余深度缓存最高累计分；未完成不声称最高分。启发式候选不构成证明。节点数包含生成与访问，预算为软限制。

路线 JSON：`{version:1,type:"2048-tas",start:<游戏存档>,options,actions}`。导入校验起点、参数、动作数≤H 并逐步复算；固定出块必须匹配 RNG，理想出块必须是合法空位上的 2/4。导入后 complete=false，不能携带伪造的最优性声明。执行逐帧比较棋盘、分数、步数、目标和 RNG，不匹配即停止。预览只读，恢复起点重置当前历史。
