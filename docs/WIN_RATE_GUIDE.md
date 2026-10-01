# P0/P1 使用与对照评测

本阶段实现代码与评测入口，未运行新版本胜率、游戏或性能测试。构建通过不代表强力策略已经优于旧策略。P2 学习模型、P3 GPU 学习策略和 P4 残局表尚未接入。

## 界面使用

刷新本机页面，在算法中选择「强力 · 阶段与风险搜索」。页面新增默认[自动参数](AUTO_PARAMETERS.md)，根据当前复杂度与近期耗时设置预算/深度；选择手动参数时，可先设置200ms预算和8步上限。实际完成层在分析面板显示。上限只约束加深，不保证在预算内达到该深度。默认算法仍是原 Expectimax，便于切回。

新评分在原局面评分上加入阶段空位权重、低空位风险、活动区域、大块合并通道及蛇形中断。它们都是可调的启发式特征，活动区域与通道评分不构成未来可恢复性的证明。界面的百分比仅指「执行此动作后下一次随机出块即无合法动作」的风险；方向估值不是胜率或真实分数。

自适应搜索在开局/中盘、完成建议深度、各方向下一次出块即死风险为零且最好候选差距达到阈值时提前停止。残局继续加深直到输入上限或预算；预算不足保留四方向共同完成的上一层。深度 0 表示只使用一步出块后的回退评分，其初始化和计算属于软时间预算，未计入树节点数。

勾选「达到目标后继续游玩」时，强力 target 模式在根局面已达目标后改为下一倍最大块作为内部目标；实际棋盘和存档目标保持原值。根最大块已达 2³⁰ 时转用分数评分。score 模式始终使用分数相关启发式。

预算≥200ms且设备允许时，启用最多4个 CPU Worker 比较方向；小预算/关闭并行时使用单 Worker。各 Worker 持有独立有界缓存，每层先公平分配节点配额，再将空余配额向未完成的根回收一次。启动、查表和消息开销计入软时间预算。并行与串行的缓存布局、配额和时间消耗不同，有限预算下可能选择不同动作，不能假设多核模式一定更强。

旋转/镜像只合并缓存中的标量价值；根方向仍使用原棋盘坐标。缓存按策略版本、尺寸、目标、估值参数隔离，只保存完成的子树。暂停、撤销、编辑、改参数时终止活动实例并拒绝旧结果；闲置实例复用缓存。

近似阈值默认为 0。大于 0 时，对单次出块条件概率低于阈值的边，用该出块后的叶子估值替代继续搜索，保留原概率权重。它不是删掉概率再归一化；`prunedEdges` 记录实际替代边数，不代表全树被截断概率。exact、旧 Rollout 和 TAS 不使用这个选项。

## 随机新局成对 A/B

工具真正从提交 `3c0a2c42619e156cd0a59efaee468b4170fa608f` 提取旧求解器、评分、规则、行查表与会话代码，写入本次报告的 `frozen-baseline/`，再导入执行。当前源代码与评测脚本也归档在 `candidate-source/`，同时记录代码哈希、Git状态、参数、分析种子及硬件。

双方使用相同起始棋盘和 RNG 状态，每个有效移动消耗相同的两次随机抽样。不同动作产生不同空位，出块位置按各自棋盘计算；求解器只能看到棋盘，不能读取真实 RNG 的未来。

先做小批核对，再增加局数；以下命令均由使用者执行。输出默认创建带时间的独立目录，不覆盖旧结果：

```powershell
# 首轮节点预算对照，观察差异和失败轨迹
npm run bench:ab -- --games 10 --split tuning --budget-mode nodes --nodes 10000 --horizon 4

# 调参集100局；score 模式继续到终局，统计多个目标
npm run bench:ab -- --games 100 --split tuning --budget-mode nodes --nodes 100000 --horizon 8

# 相同时间预算下的串行 CPU 对照
npm run bench:ab -- --games 100 --split tuning --budget-mode time --budget 200 --horizon 8

# 参数固定后，独立留出种子；预先确定局数
npm run bench:ab -- --games 1000 --split holdout --budget-mode nodes --nodes 100000 --horizon 8

# 只考察2048目标，到目标即停；高于停止目标的到达率可能未知
npm run bench:ab -- --games 100 --objective target --target 2048 --stop-at-target true --budget-mode nodes --nodes 100000 --horizon 8

# 原版/当前版 Rollout 固定样本对照；不使用树节点预算
npm run bench:ab -- --games 100 --baseline-algorithm rollout --candidate-algorithm rollout --budget-mode samples --trajectories 128 --horizon 32
```

固定节点/样本模式将时间上限设为 10¹²ms，以免主机速度改变搜索结果；这仍是有限软上限。time 模式放宽节点上限。节点数是各实现的工作量定义，不是等量机器指令；Strong 的持久缓存会影响访问数。Node 评测强制串行并记录 `rootParallel=false`，不能用于证明浏览器多 Worker 或 CUDA 加速比；设备吞吐仍使用现有独立评测入口。

训练、调参、留出种子分别带 training/tuning/holdout 前缀；工具检查本批游戏种子哈希不重复。`--seed-start` 改变区间；同一分区与区间不要反复当成新独立样本。不要根据留出结果继续选择参数后仍称该批为留出验证。Strong 状态缓存每个案例前清空、局内跨步复用；JIT 与行查表仍保持预热，A/B 顺序逐局交替。

## 存档与消融

```powershell
# 一份存档只运行一对，保留 RNG；不覆盖输入
npm run bench:ab -- --save "C:\Users\59339_\Downloads\2048-ai-save.json" --target 65536 --budget-mode nodes --nodes 100000 --horizon 8 --max-moves 50000

# 多案例：manifest 是存档路径的 JSON 数组，相对路径以 manifest 目录为准
npm run bench:ab -- --manifest artifacts\cases.json --budget-mode nodes --nodes 100000 --horizon 8

# 复制样例后修改一个开关；相同 tuning 种子比较
npm run bench:ab -- --games 100 --strong-config benchmarks\strong-config.example.json --budget-mode nodes --nodes 100000 --horizon 8
```

存档案例按起点最大块、空位和阶段记录，最终局面保存完整棋盘便于进一步分层；每份只运行一次，不把重复确定性案例算作随机胜率样本，不输出 Wilson 区间。`--target` 设置本次比较目标，原文件只读。保存原 RNG 的普通随机续玩与 TAS 的主动出块/读取未来随机数应分别评测。

`strong-config` 支持 adaptive、stage、risk、channels、symmetry、reuseCache、rootParallel、chanceCutoff、cacheEntries、openingDepth、middleDepth、endgameDepth、clearGap。Node 强制关闭 rootParallel。首轮建议逐项关闭做消融；评分权重与0.025候选差距仍是初始设定，尚无留出收益证明。

## 报告口径

- `metadata.json`：冻结提交/源代码哈希、实际种子、所有配置与硬件。
- `summary.json`：每目标已达标、终局未达标、未知数量；完整率、成对成功/失败差、实际层深/节点、缓存命中、预算触顶及自适应停止。
- `<case>-baseline/` 和 `<case>-candidate/`：起始/最终存档、逐步动作与 RNG/棋盘 `trace.jsonl`、最后若干帧 `tail.json`、状态与局面阶段。每64次决策更新 `latest-save.json`；默认保留末64帧，可改 `--trace-tail`。

终局、达到停止目标、步数截断/取消、求解异常分别列出。已达某目标即为该目标的确定成功；未达且真正终局为失败；其他未达为未知。存在未知时到达率为 null，并报告整批可能范围；只有该目标全部已解决且来自随机新局才输出95% Wilson区间。成对差值只取两侧该目标都已解决的配对，同时报告未解决/缺侧数量，不能据此替代全体差值。

分数按终局、达标即停、截断/异常分别汇总。延迟在终局检查后测量（规则查表已由检查初始化），不含文件/轨迹输出；P50/P95 是固定直方图区间的上界，不是假称精确分位数。整局 elapsedMs 包含初始化、日志与存档开销，另行保留。

`--max-moves` 限新增有效移动数，默认30000，终局检查优先于步数上限。SIGINT 在当前决策结束后保存状态；npm/终端强制杀进程可能只留下最近检查点。报告会显示尚未开始的案例数，未执行案例没有胜率结论。
