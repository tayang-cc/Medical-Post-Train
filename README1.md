# Medical Reasoning RL 技术文档

> 基于 Qwen2.5-7B-Instruct 的医疗复杂推理：SFT（思维链微调）+ PRM（过程奖励模型）+ GRPO（强化学习对齐）的完整工程记录。
> 本文档面向「复现 + 理解设计决策」的读者，与 README.md 的快速上手互补。

---

## 1. 背景与核心问题

项目思路源自 **HuatuoGPT-o1**：用「可验证的医学问题 + 验证器」打通「思维链构造 → 过程监督 → 强化学习」闭环。

正式实验暴露出两个根因，构成 PRM 升级的动机：

1. **SFT 有害**：用 32B 老师生成的 1605 条 CoT 直接 SFT，准确率从基座 73.5% **反降到 68.0%**。根因是 CoT 数据「脏」——答案虽对，但推理过程混入了错误步骤，监督学习把这些错误一起学了进去。
2. **Judge 校准差导致 GRPO 不涨**：LLM-as-Judge 打连续分，均值仅 0.33、区分度差，组内相对优势被噪声淹没，GRPO（72.5%）未能超越基座。

**PRM 升级的核心主张**：用「真 PRM（步级过程奖励模型）」替换「整条奖励链路上的 LLM-as-Judge」，让过程监督有判别力，从而：
- 用 PRM 清洗 SFT 数据（去掉含错误步骤的轨迹）；
- 用 PRM 的 min 聚合分数作为 RL 过程奖励。

---

## 2. 整体架构

```
CMB（中文医学基准，FreedomIntelligence）
   │
   ├─ [数据层] 01b 准备 → 00 问题库严格隔离切分 pool_{sft,rl,val,test}
   │
   ├─ [CoT 层] 02 教师(32B)生成 CoT → 12 语义改写对齐风格 → 10 PRM 软过滤
   │
   ├─ [PRM 层] 07 步级标注(7B 生成 + 32B 投票) → 08 训练 7B-PRM → 09 阈值扫描
   │
   ├─ [训练层] SFT(清洗后 CoT) → GRPO(PRM 奖励)
   │
   └─ [评测层] CMB hold-out 200 题准确率
```

**关键隔离原则**：SFT / RL / 评测 / PRM 源 四者严格按题目互斥，禁止任何 step 级随机切分（会泄露题目）。

---

## 3. PRM（过程奖励模型）

### 3.1 设计

- **任务**：给定 `问题 + 选项 + 前置上下文 + 当前步骤`，输出该步的医学正确性分数（0~1）。
- **模型**：Qwen2.5-7B + 回归头（`AutoModelForSequenceClassification, num_labels=1`）。
- **损失**：`sigmoid 头 + BCEWithLogitsLoss`（不是 MSE——见 §6 教训），配 label smoothing(0.05) 抗硬标签饱和 + pos_weight(0.55) 加权错误步（漏检代价 > 误判）。
- **LoRA**：attention-only（q/k/v/o）、r=8，训练 0.07% 参数，防过拟合标注噪声。
- **输入不含标准答案**，避免答案泄露进步骤打分。

### 3.2 步级标注（构建 PRM 训练集）

1. **轨迹生成**：用 **7B 策略模型**（而非 32B）每题材 4 条轨迹，保留答案错误轨迹以提供负样本。这样 PRM 学的就是「它将来要评判的策略」的风格。
2. **步骤切分**：`split_steps` 按句号/换行切分 + 最小长度过滤（剔除切分噪声）。
3. **Judge 投票**：每步独立调 32B Judge K=3 次（temp>0），≥2 票一致保留标签（1.0/0.0），无法形成多数则丢弃该步样本；prompt 强制结构化输出（整串仅 0/1）+ 解析失败重试。
4. **源题目**：取 pool_rl 前 2000 题，内部切 PRM-train 1600 / PRM-val 400（早停用，禁碰全局 TEST）。

产出约 **9.3 万步样本**（正负比 64/36）。

### 3.3 训练与指标

- PRM-val 上早停，`metric_for_best_model = eval_auc`。
- 指标（不是只看 loss）：**AUC + 错误步检出（precision/recall/F1）+ pos_recall（正确步召回，监控 pos_weight+label_smoothing 是否过度纠偏）**。
- 最终：**step AUC 0.935，错误步 recall 0.86，pos recall 0.85**；min 聚合轨迹级 AUC 0.938；推理侧越界占比 0.000%（sigmoid 输出天然落 [0,1]，仍做 clip 双保险）。

### 3.4 阈值扫描

AUC 选出的是「排序最优」模型，但下游 SFT 过滤 / RL 奖励需要实际分数阈值。`09_scan_prm_threshold.py` 在 PRM-val 上双层面扫描：
- **step 级**：0.05~0.95 扫 accuracy / 错误步 precision-recall-F1；
- **轨迹级（min 聚合）**：`min(step score)` 预测「所有步正确」，这是 RL min 聚合奖励的真实信号。

---

## 4. 域偏移问题（本项目最重要的架构结论）

### 4.1 现象

PRM（7B 策略轨迹训练）对 **32B 老师 CoT** 打极低分：min-score 均值 **0.099**，仅 3.9% 过 0.5。但被保留的高分样本**也是 markdown 风格**，说明问题不在格式装饰，而在**措辞/详略/句式的分布偏移**。

### 4.2 三层解法对照实验

| 方案 | PRM min-score 均值 | 结论 |
|---|---|---|
| 原始 32B CoT | 0.101 | 基线 |
| 层次A：格式脱敏（去 markdown 加粗/编号/标题） | 0.120 | +0.019，几乎无效 |
| **层次B：语义改写（32B 改写为简洁逐步风格）** | **0.215** | **翻倍，有效** |

- **格式脱敏（层次A）无效**：域偏移主因是措辞分布，不是 markdown。
- **语义改写（层次B）有效**：`12_rewrite_cot.py` 用 32B 把自己 CoT 改写为简洁逐步风格（只改表达、不改结论，改写后 99% 答案一致），PRM 分数翻倍，且生成长度 390→146 token。
- **层次C（重训 PRM，用 32B 轨迹标注）**：理论上最彻底，但被否——引入「策略↔奖励」循环风险，且耗时。语义改写更干净（不引入循环）、更便宜。

**结论**：「SFT 风格脱敏」是对的，但要脱到**措辞层**，而非只去 markdown。语义改写既提 SFT 本身（75→77.5），又让生成风格对齐 PRM 训练分布，GRPO 才真正能起效。

---

## 5. 强化学习（GRPO）

### 5.1 奖励设计

`PRMCompositeReward`：

```
process      = min_weight × min(步分) + (1 - min_weight) × mean(步分)   # 软 min
length_bonus = -length_penalty × max(0, min_len - len(text)) / min_len   # 过短惩罚
raw          = result_weight × result + process_weight × process + length_bonus
reward       = max(0.0, raw)                                            # 非负 floor
```

- **格式门控**：最终答案可解析才通过，否则 reward=0（防格式劫持，与结果奖励解耦）。
- **二进制结果**：抽取答案 == 标准答案（0/1）。
- **过程奖励（软 min）**：`0.3×min + 0.7×mean`，min 保留致命错误信号、mean 提供平滑梯度（v4 引入；纯 min 太稀疏导致均值仅 ~0.18）。
- **长度激励**：低于 `min_len=75` 字符（≈50 token）视为「跳过推理」，线性惩罚，防「只答答案不推理」坍缩。
- **RL 循环内不再调 32B Judge**；PRM 在进程内打分（`PRMScorer` 批量前向）。

### 5.2 训练细节

- 从 SFT merge 起训（`--merge-from-sft`，内存中 merge 不落盘），套新 LoRA。
- KL 退火（cosine 0.04→0.005）+ 熵正则（`entropy_coef=0.001`）防策略坍缩。
- policy 必须 bf16 加载、单卡直接 python 跑（见 §6）。

### 5.3 lr 调参教训

| 版本 | lr | kl 终点 | acc | 说明 |
|---|---|---|---|---|
| v1 | 1e-6 | 0.0007 | 76.0% | 策略几乎不动 |
| v2 | 1e-5 | 0.0035 | 71.0% | **奖励劫持**（去拟合带噪声的过程奖励） |
| v3 | 5e-6 | 0.0017 | 78.0% | 风格对齐后，过程奖励有区分度 |

**结论：lr 不是瓶颈，域偏移导致过程奖励是噪声才是。** 不解决域偏移，GRPO 只能在「lr 极低（策略不动）」和「lr 稍高（奖励劫持）」之间选；语义改写对齐风格后，中等 lr 才有效。

### 5.4 奖励调参 vs 知识天花板（v4 实验）

v4 把奖励改成软 min + 长度激励 + 2 epoch：

| 版本 | 改动 | kl 终点 | acc |
|---|---|---|---|
| v3 | min 聚合、1 epoch | 0.0017 | 78.0% |
| v4 | 软min + 长度激励、2 epoch | **0.007** | **78.0%** |

kl 涨了 **4 倍**（策略真正动起来了），但 acc 纹丝不动。这从实验上坐实：

> **GRPO 在 CMB 上的天花板是 ~78%，瓶颈是 7B 的医学知识量，不是奖励函数或优化目标。** 策略移动更多（软 min 让过程信号更强更平滑），却无法凭空学会它不知道的知识。

要破 78% 只能换更硬的杠杆：更大基座（知识量）、多采样投票（评测手段）、扩数据（逼近同一知识天花板）——而非继续调 GRPO 超参。

---

## 6. 实验结果

CMB-200 hold-out（与正式实验同评测集）：

| 模型 | acc | 说明 |
|---|---|---|
| 基座 7B | 73.5% | — |
| 旧 SFT（1605 脏 CoT） | 68.0% | 反降 |
| SFT_clean（PRM 软过滤 1185 条） | 75.0% | 删 min<0.02 的 26% |
| GRPO v1（lr 1e-6） | 76.0% | kl≈0，策略未动 |
| SFT_rewrite（语义改写 + 软过滤 1432 条） | 77.5% | 生成长度 390→146 |
| GRPO v3（min 聚合，lr 5e-6） | 78.0% | kl≈0.0017 |
| GRPO v4（软min + 长度激励，2 epoch） | 78.0% | kl≈0.007 但 acc 不变（见 §5.4） |

**完整链路 +4.5%（73.5% → 78.0%）**，三阶段各司其职：
1. PRM 软过滤 SFT（68→75）：去掉极端错误推理，SFT 翻正；
2. 语义改写（75→77.5）：措辞对齐 7B 风格，既提 SFT 又缓解域偏移；
3. GRPO + PRM 奖励（77.5→78）：风格对齐后过程奖励可信，小增益；v4 证明再调 reward 超参已到知识天花板。

---

## 7. 工程问题与修复（沉淀）

### 7.1 数据/隔离

- pool_rl 存在 **940 条重复题**（CMB 勘误同题干）→ 按题干去重后再切分，训练脚本断言 train/val 题目集零重叠。
- 下游带 options 的环节（PRM 打分、RL 奖励）需保留 `options` 字段。

### 7.2 PRM 训练

- **MSE 回归头数值爆炸**（eval_loss 1.8）→ 改 sigmoid+BCE。
- **冻结基座 + 梯度检查点梯度断裂**（`None of the inputs have requires_grad` 告警）→ `enable_input_require_grads()`，或直接关梯度检查点（96G 单卡够用且更快）。
- 模型 `config.pad_token_id` 未继承 tokenizer → 分类头池化 batch>1 报错 → 显式设。
- torch 版本 `BCEWithLogitsLoss` 不支持 `label_smoothing` → 手动实现 `smooth = y*(1-2eps)+eps`。
- 阈值扫描 step 级 err 指标极性写反 → 修正。
- 轨迹级 GT 与 min 聚合不对齐（答案对≠步骤全对）→ 用 `min(label)==1`。

### 7.3 GRPO / RL

- **OOM 根因①：漏传 `--lora`** → 全量 7B 训练（fp32 AdamW 56GB）。
- **OOM 根因②：policy 默认 fp32 加载**（28GB）→ `model_init_kwargs={"torch_dtype":"bfloat16"}`。
- `--merge-from-sft` 与 `model_init_kwargs` 冲突（模型已实例化）→ 条件化。
- `Dataset.map(batched=True)` 回调收字段字典 → 按位置索引取值。

### 7.4 环境（AutoDL + RTX PRO 6000 Blackwell）

- **vllm + Blackwell**：flashinfer-python==0.6.18（清华源）+ flashinfer-cubin==0.6.18（GitHub）+ 给 `flashinfer/jit/core.py::check_cuda_arch` 打补丁（sm120 的 raise→return）。
- **torch 2.13 nightly import 期 bug**：`vllm` 启动导入 `torch._inductor` 时报 `duplicate template name` / `duplicate extern kernel: _grouped_mm` → 通用补丁：把 `torch/_inductor` 下所有 `assert ... duplicate ...` 断言中性化 + 清 pycache。
- 上述补丁均在系统盘 site-packages，**实例切换/系统盘重置后需重打**。
- 双环境：base（`/root/miniconda3`：vllm 0.27.1 + transformers 5.15.1 + flashinfer）与 train（`/root/autodl-tmp/train_env`：transformers 4.49 + trl 0.15.0），不能互换。

---

## 8. 关键决策回顾

| 决策 | 选择 | 理由 |
|---|---|---|
| 过程监督用「真 PRM」而非「整条 Judge」 | PRM | Judge 校准差是 GRPO 不涨的根因 |
| PRM 轨迹用 7B 策略生成 | 7B | PRM 应评判「它要评判的策略」的风格 |
| 域偏移解法 | 语义改写（层次B） | 比格式脱敏彻底，比重训 PRM 干净（不引入策略↔奖励循环） |
| SFT 数据清洗 | PRM 软过滤（只删极端低分） | 硬截断会被域偏移误伤，软过滤只抓真断裂 |
| RL 内不调 32B Judge | 进程内 7B-PRM | 离线标注才用 32B，在线用 7B 省时省资源 |
| 过程聚合 min → 软min | `0.3×min+0.7×mean` | 纯 min 太稀疏（均值 0.18），软 min 平滑梯度 |
| 是否继续调 GRPO 超参 | 停止 | v4 证明 kl 涨 4 倍 acc 不变，已到知识天花板 |
