# Medical Reasoning RL

基于 Qwen2.5-7B-Instruct 的医疗复杂推理项目：SFT 阶段 CoT 构造 + 过程监督（PRM）+ 强化学习（GRPO）对齐。

思路参考 HuatuoGPT-o1（Towards Medical Complex Reasoning with LLMs）：用可验证的医学问题 + 验证器，打通"思维链构造 → 过程监督 → 强化学习"闭环。

## 流水线

```
CMB（中文医学基准）
      │
      ▼
[0] 问题库 ─── 严格隔离切分 pool_{sft,rl,val,test}（互斥）
      │
      ▼
[1] CoT 构造 ─ 教师(32B)生成推理链 → 语义改写对齐风格 → PRM 软过滤
      │
      ▼
[2] SFT ────── Qwen2.5-7B-Instruct 在清洗后的 CoT 上微调
      │
      ▼
[3] RL ─────── GRPO，Reward = 格式门控 + 二进制结果 + PRM(min聚合)过程分
      │
      ▼
[4] 评测 ───── CMB hold-out 200 题准确率

PRM（过程奖励模型）自成一条支线：
  7B 策略生成轨迹 → 32B Judge 步级投票标注 → 训练 7B-PRM → 供 SFT 过滤 + RL 奖励
```

## 目录结构

```
configs/            训练与数据配置 (yaml / json)
scripts/            端到端脚本 (bash / python)
src/answer_extraction.py  统一的最终答案抽取（reward / CoT / 评测共用）
src/data/           问题库切分 + CoT 构造 + 步级标注
src/process/        PRM 推理打分器
src/reward/         复合 Reward 函数（Judge / PRM 两种）
src/train/          SFT + GRPO + DAPO 训练入口
src/eval/           评测（transformers 直接 / vllm 双轨）
data/raw|cot|splits|prm|processed|eval/  数据
checkpoints/        模型权重（base 未含，adapter 落盘）
```

## 快速开始

```bash
# 1. 安装依赖（trl 已固定 0.15.0，见 requirements.txt）
pip install -r requirements.txt

# 2. 数据准备 + 问题库严格隔离切分
python scripts/01b_prepare_data_cmb.py
python scripts/00_build_problem_library.py

# 3. 教师(32B)生成 CoT（需 vllm，端口 8000）
bash scripts/serve_vllm.sh /root/autodl-tmp/models/Qwen2.5-32B-Instruct-AWQ 8000 1
python scripts/02_build_cot_data.py --config configs/data.yaml

# ---- PRM 支线（过程奖励模型）----
# 4. 步级标注：7B 生成轨迹 + 32B Judge 投票（生成器端口 8002，Judge 端口 8000）
python scripts/07_step_annotation.py --limit 2000 --train-ratio 0.8
# 5. 训练 7B-PRM
python scripts/08_train_prm.py --config configs/prm.yaml
# 6. 阈值扫描（可选，产出 RL 用阈值报告）
python scripts/09_scan_prm_threshold.py

# ---- SFT 主线 ----
# 7. 语义改写 CoT（对齐 PRM 风格）+ PRM 软过滤
python scripts/12_rewrite_cot.py            # 32B 改写为简洁风格
python scripts/10_clean_sft_data.py         # PRM 软过滤（删极端低分）
# 8. SFT（单卡 LoRA 直接 python 跑）
python src/train/sft.py --config configs/sft.yaml --lora

# 9. 构造 RL 数据（排除 PRM 源，含 options）
python scripts/04_build_rl_prompts.py --limit 500

# 10. GRPO（from SFT merge + PRM 奖励，RL 内不再调 32B Judge）
python src/train/grpo.py --config configs/grpo.yaml --lora --merge-from-sft checkpoints/sft_rewrite

# 11. 评测
python src/eval/eval_transformers.py --model <base> --adapter <adapter> \
    --data data/eval/cmb_eval_200.jsonl --batch-size 8
```

## 关键组件

- `src/data/problem_library.py`：可验证问题库严格隔离切分（sft/rl/val/test 互斥）
- `src/data/step_annotation.py`：步级标注（split_steps + 32B Judge K=3 投票）
- `src/process/prm_scorer.py`：7B-PRM 推理打分器（sigmoid 输出 [0,1]，批量打分）
- `src/reward/composite.py`：复合 Reward = `CompositeReward`（Judge 连续分）+ `PRMCompositeReward`（格式门控 + 二进制结果 + PRM min 聚合）
- `src/train/grpo.py`：GRPO + KL 退火（`KLAnnealingCallback`）+ 熵监控（`MedicalGRPOTrainer` / `MonitoringCallback`）
- `src/train/dapo.py`：DAPO（`DAPOTrainer`）＝ GRPO + token-level loss + clip-higher + dynamic sampling + overlong shaping
- `src/train/sft.py`：SFT（LoRA，消息列表 prompt/completion）
- `src/eval/eval_transformers.py`：transformers 直接评测（单 adapter）；`src/eval/eval_stacked.py`：链式评测（merge adapter + 叠加 adapter，用于 GRPO 产物）
- `scripts/serve_vllm.sh`：vLLM 服务启动（32B teacher / Judge / 改写通用）

## 模型服务（vLLM）

```bash
# 教师/Judge/改写 模型（32B AWQ，端口 8000）
bash scripts/serve_vllm.sh /root/autodl-tmp/models/Qwen2.5-32B-Instruct-AWQ 8000 1

# 步级标注的轨迹生成器（7B，端口 8002）
bash scripts/serve_vllm.sh /root/autodl-tmp/models/Qwen2.5-7B-Instruct 8002 1
```

## 训练注意事项

- **版本**：`trl` 已固定为 `0.15.0`（GRPO/SFT 接口按其适配）。升级 trl 前先看 reward 签名与 `beta`/callback 是否兼容。
- **DeepSpeed**：`configs/accelerate.yaml` 是 accelerate 启动配置，真正的 DeepSpeed 参数在 `configs/ds_zero3.json`。不要在 `sft.yaml` / `grpo.yaml` 里再写 `deepspeed:`（会与 accelerate 冲突）。
- **单卡**：`accelerate launch --config_file configs/accelerate.yaml --num_processes 1`，且保证 GRPO 的 `num_generations` 整除 `per_device_train_batch_size x num_processes`（默认 G=8 时单卡需 `per_device_train_batch_size=8`，脚本会在启动时校验并提示）。
- **数据链路**：SFT 输入为语义改写 + PRM 软过滤后的 CoT（`10`/`12` 步产出）；GRPO 输入为 `04_build_rl_prompts.py` 产出的 `data/processed/rl_prompts.jsonl`（排除 PRM 源、含 `options`）。
- **RL 奖励**：正式链路用进程内 7B-PRM（`PRMCompositeReward`），**RL 循环内不再调 32B Judge**；32B 只在离线步级标注、CoT 语义改写时用。
- **DAPO vs GRPO**：`configs/dapo.yaml` 默认去掉 KL 惩罚（论文 §2.3，`beta=0`）、`eps_low=0.2 / eps_high=0.28`。由于 trl 0.15 没有 old-policy 缓冲，clip-higher 的 ratio 采用「当前策略 vs 冻结参考模型」，语义等价（见 `src/train/dapo.py` 文件头注释）。想保留 KL 约束可把 `beta` 调回 0.001~0.04 并开启 `kl_annealing`。
- **LoRA 单卡**：`--lora` 开关（`src/train/lora.py`）。注意 trl 0.15 在 deepspeed zero3 + peft 下仍加载独立 ref model（省显存失效），所以 LoRA 必须**直接 `python` 单进程跑**（`scripts/05*_lora.sh`），且 `per_device_train_batch_size >= num_generations`。LoRA 训练产物是 adapter，评测前需用 `vllm serve <基座> --enable-lora --lora-modules name=<checkpoints/...>` 挂载，或先 merge 成完整权重。

## 训练硬件与耗时

所有训练均在**单张 NVIDIA RTX PRO 6000 Blackwell（96 GB）**上完成，policy/PRM 均 bf16 + LoRA。

| 阶段 | 数据规模 | 耗时 | 峰值显存 |
|---|---|---|---|
| PRM 步级标注 | 2000 题 × 4 轨迹（32B Judge 投票） | ~3-4 小时 | ~62 GB（7B+32B 双 vllm） |
| PRM 训练 | 7.4 万步样本 × 2 epoch | ~2-4 小时（含调参） | ~25 GB |
| 语义改写 | 1605 条（32B 改写） | ~40 分钟 | ~62 GB |
| PRM 软过滤 | 1605 条（7B-PRM 打分） | ~7 分钟 | ~15 GB |
| SFT（LoRA） | ~1400 条 × 3 epoch | ~5 分钟 | ~25 GB |
| GRPO | 500 题 × G=4 × 125 步 | 45 分钟（短生成）~2 小时（长生成） | ~62 GB |
| 评测 | 200 题 × batch 8 | ~5 分钟 | ~15 GB |

## 文档导航

- [README.md](README.md)：快速上手 + 实验记录 + 结果
- [README1.md](README1.md)：技术文档（架构 / PRM / 域偏移 / 教训沉淀）
- [README3.md](README3.md)：全流程实例变化展示
- [dashboard.html](dashboard.html)：实验对比仪表盘（静态，浏览器直接打开）

## 实验记录（Smoke Test）

> 2026-08 冒烟验证：管线端到端跑通 + 拿到对比表。完整复现方式见下文「说明」。

- **数据**：CMB（中文医学基准，FreedomIntelligence 出品）。训练侧用 CMB-val 官方解析（`explanation`）当 CoT（240 条，因 vllm/Blackwell 未解决故跳过 teacher 生成）；RL 侧用 CMB-train 前 40 条。
- **评测集**：CMB-train 尾部 hold-out **200 题**（与训练侧无交集）。
- **基座**：`Qwen/Qwen2.5-7B-Instruct`，LoRA（r=16, alpha=32, all-linear），单卡 RTX PRO 6000 Blackwell 97.8G。

| 模型 | acc | miss | avg_len | 训练规模 |
|---|---|---|---|---|
| 基座（零样本） | **0.725** | 0.005 | 281 | — |
| SFT (LoRA) | 0.680 | 0.065 | 284 | 240 条 CoT × 3 epochs |
| GRPO (LoRA) | **0.755** | 0.005 | 279 | 基座起训，40 题 × G=4 × 5 步，纯规则 reward（格式 0/1） |
| GRPO (LoRA, 熵正则 0.001) | 0.730 | 0.005 | 273 | 同上 + `entropy_coef=0.001`，loss 由 0.0→-0.0008（熵项生效） |

**结论**：
1. 全链路（数据 → SFT → GRPO → 评测）跑通，产物格式正确。
2. GRPO 用纯规则 reward 仅 5 步就把准确率从 72.5% 提到 75.5%、miss 收敛到 0.5% —— 说明「格式硬约束 + 组内相对优势」能快速教会模型稳定输出答案格式。
3. SFT 反降（68.0%）：240 条样本过少 + 长解析 CoT 过拟合 + miss 升高（6.5%），符合「小数据 SFT 有害」的已知现象，属可解释结果。
4. **熵正则（`entropy_coef`）已实现并验证「生效」**：loss 从 0.0 变为 -0.0008，`policy_entropy` 进入日志监控。但 5 步冒烟下 acc 73.0% vs 75.5% 的差异属采样噪声且混入了 transformers 版本更换（4.50→4.49），**未形成「提升」证据**；需要同版本+固定 seed+更多步数的干净 A/B 才能下结论。

### 正式实验（2026-08，SFT 1605 条 + GRPO 200 题 × 25 步 + LLM-judge）

- SFT：teacher(32B) 生成 2000 题 CoT → 保留 1605 条正确 → 过程验证(LLM-judge 逐句打分，但 judge 校准偏严，均值 0.33，SFT 未按 process_score 过滤) → LoRA SFT。
- GRPO：基座 + LoRA，200 题 × G=4 × 25 步，**LLM-judge 连续 reward + 格式硬约束**（复合 reward）。

| 模型 | acc | miss | avg_len | 说明 |
|---|---|---|---|---|
| 基座（零样本） | **0.735** | 0.005 | 277 | — |
| SFT (LoRA, 1605 CoT) | 0.680 | 0.010 | 398 | 过拟合，反降（两次实验一致） |
| GRPO (LoRA, 200题, LLM-judge) | 0.725 | 0.005 | 275 | ≈基座，未复现纯规则 reward 的提升 |

**正式实验结论（诚实）**：
1. 完整管线（数据→CoT→过程验证→SFT→GRPO→评测）端到端跑通，LLM-judge 真实参与 RL。
2. **SFT 小数据过拟合是稳定结论**（两次实验 SFT 均低于基座），1605 条对 7B 全参/LoRA 都偏少，需放大数据或混通用数据防过拟合。
3. **LLM-judge 校准是 GRPO 未能超越基座的最可能根因**：judge 打分均值 0.33、区分度差，组内相对优势被噪声淹没（对比：纯规则 reward 的 GRPO 在冒烟中 75.5% > 基座 72.5%）。
4. 待办：校准 judge → 放大 SFT 数据 → merge SFT→GRPO 标准链路 → 试 DAPO。

**已知限制（正式实验前必须解决）**：
- ~~**vllm 0.27 + Blackwell 不可用**~~ **已解决（2026-08）**：base 环境装 `flashinfer-python==0.6.18`（清华源）+ `flashinfer-cubin==0.6.18`（GitHub releases），并给 `flashinfer/jit/core.py` 的 `check_cuda_arch` 打补丁（把对 sm120 的误拒 `raise` 改为 `return`，cubin 0.6.18 含 sm120 预编译内核）。补丁在 site-packages 里，系统盘重置后需重打；也可升级 flashinfer 到原生支持 sm120 的版本替代。vllm 0.27.1 现可在 RTX PRO 6000 Blackwell 上正常 serve 32B AWQ。
- 训练环境的坑已逐个修掉：trl 0.15 需配 transformers 4.49（`_get_train_sampler` / `self.control` 兼容）；LoRA `target_modules` 必须传字符串 `"all-linear"`（不能是列表）；GRPO 收尾保存因 trl/transformers 的 control bug 崩溃，已在 `MedicalGRPOTrainer.train()` 加兜底捕获。
- 训练数据格式：SFT 的 dataset 直接用 `prompt`/`completion` 消息列（勿用 `formatting_func` 返回消息列表，trl 0.15 会误判为 batch）。

## PRM 升级（过程奖励模型，2026-08）

> 目标：用「真 PRM（步级过程奖励）」替换 LLM-as-Judge 的整条奖励链路，解决正式实验两大根因——「SFT 有害」「judge 校准差导致 GRPO 不涨」。

### 方案（4 步）

1. **可验证问题库严格隔离**：`scripts/00_build_problem_library.py` 把 CMB 切成 `pool_{sft,rl,val,test}.jsonl`（互斥，17万/3.6万/1.2万/2.4万）。
2. **步级标注**：`scripts/07_step_annotation.py` 用 **7B 策略模型**每题材 4 条轨迹（保留错误轨迹供负样本），**32B Judge 每步 K=3 投票**（≥2 票一致保留，无法多数则丢步），产出 9.3 万步样本（正负比 64/36）。源题目取 pool_rl 前 2000 题，内部切 PRM-train 1600 / PRM-val 400（早停，禁碰全局 TEST）。
3. **训练 7B-PRM**：`scripts/08_train_prm.py` 用 sigmoid+BCE 回归头（`num_labels=1, problem_type=single_label_classification`）+ label smoothing(0.05) + pos_weight(0.55，加权错误步) + attention-only LoRA(r=8)。PRM-val 上早停，指标含 AUC / 错误步检出 / pos_recall（监控过度纠偏）。
4. **PRM 清洗 SFT + GRPO 接入**：`src/reward/composite.py::PRMCompositeReward` = 格式门控 + 二进制结果 + min 聚合 PRM 分数，**RL 循环内不再调 32B Judge**；`scripts/10_clean_sft_data.py` 用 PRM 软过滤 SFT 数据。

### 结果（CMB-200 评测，与正式实验同评测集）

| 模型 | acc | 说明 |
|---|---|---|
| 基座 7B | 73.5% | — |
| 旧 SFT（1605 脏 CoT） | 68.0% | 正式实验复现，反降 |
| SFT_clean（PRM 软过滤后 1185 条） | 75.0% | 只删 min-score<0.02 的 26% 极端低分样本 |
| GRPO v1（SFT_clean + PRM 奖励，lr 1e-6） | 76.0% | kl≈0.0007 策略几乎未动 |
| SFT_rewrite（32B 语义改写 + 软过滤 1432 条） | 77.5% | 生成长度 390→146，风格对齐 PRM |
| **GRPO v3（SFT_rewrite + PRM 奖励，lr 5e-6）** | **78.0%** | 最终结果 |

- **PRM 指标**：step AUC **0.935** / 错误步 recall 0.86 / pos recall 0.85（无过度纠偏）；min 聚合轨迹级 AUC **0.938**；推理侧越界占比 0.000%（sigmoid 输出天然落 [0,1]）。
- **完整链路**：base 73.5% → **78.0%（+4.5%）**。三阶段各司其职：① PRM 软过滤 SFT（68→75，SFT 翻正）② 语义改写（75→77.5，缓解域偏移）③ GRPO+PRM 奖励（77.5→78，小增益）。

### 域偏移的解决：格式脱敏 vs 语义改写（对照实验）

PRM（7B 策略轨迹训练）对 32B 老师 CoT 打极低分是「域偏移」。验证了两条脱敏路线：

| 方案 | PRM min-score 均值 | 变化 |
|---|---|---|
| 原始 32B CoT | 0.101 | — |
| 层次A：格式脱敏（去 markdown 加粗/编号/标题） | 0.120 | +0.019，几乎无效 |
| **层次B：语义改写（32B 改写为简洁逐步风格）** | **0.215** | **翻倍，有效** |

- **格式脱敏（层次A）无效**：域偏移主因不是 markdown 装饰，而是 32B 的措辞/详略/句式分布。
- **语义改写（层次B）有效**：`scripts/12_rewrite_cot.py` 用 32B 把自己 CoT 改写成简洁逐步风格（只改表达不改结论，改写后 99% 答案一致），PRM 分数翻倍，且生成长度 390→146 token。
- 结论：**「SFT 风格脱敏」是对的，但要脱到措辞层，而非只去 markdown**。语义改写既提 SFT 本身（75→77.5），又让生成风格对齐 PRM 训练分布，GRPO 才能有效（kl 从 v1 的 0.0007 增到 v3 的 0.0017，且 lr 5e-6 下不再过拟合）。

### 关键问题与修复

**数据/隔离**
- pool_rl 存在 **940 条重复题**（CMB 勘误，同题干），按索引切分导致 train/val 重叠 1 题 → 按题干去重后再切分，并在训练脚本断言 train/val 题目集零重叠。
- 训练/评测中所有带 options 的下游（PRM 打分、RL 奖励）需在数据里保留 `options` 字段（原 `03b` 产出缺 options）。

**PRM 训练**
- **MSE 回归头数值爆炸**（eval_loss 1.8，输出超 [0,1]）→ 改 **sigmoid+BCE**（`single_label_classification`），loss 降到 0.31、AUC 0.935。
- **冻结基座 + 梯度检查点梯度断裂**（`None of the inputs have requires_grad=True` 告警）→ `model.enable_input_require_grads()`；更简单的做法是直接关梯度检查点（96G 单卡够用，且更快）。
- 模型 `config.pad_token_id` 未继承 tokenizer 设置 → 分类头池化在 batch>1 时 `Cannot handle batch sizes > 1 if no padding token is defined` → 显式设 `model.config.pad_token_id = tok.pad_token_id`。
- torch 版本 `BCEWithLogitsLoss` 不支持 `label_smoothing` 参数 → 手动实现 `smooth = y*(1-2eps)+eps`。
- 阈值扫描 step 级 err 指标**极性写反**（`pred=score>=th` 被当成「预测错误」）→ 修正为 `pred_err = score < th`。
- 轨迹级 GT 与 min 聚合不对齐（「答案对」≠「步骤全对」，存在步骤错但答案蒙对）→ 用 `min(label)==1`（所有步正确）作为主标签，答案正确性仅作次参考（其 AUC 仅 0.586 印证噪声）。
- pos_weight + label_smoothing 叠加可能过度纠偏 → 增加 `eval_pos_recall`（正确步召回）对称监控。

**GRPO / RL**
- **OOM 根因①：漏传 `--lora`** → 全量 7B 训练（fp32 AdamW 优化器状态 56GB）吃满显存。
- **OOM 根因②：policy 默认 fp32 加载**（28GB）→ `GRPOConfig(model_init_kwargs={"torch_dtype": "bfloat16"})`。
- `--merge-from-sft`（模型已实例化）与 `model_init_kwargs` 冲突 → 仅当模型是字符串路径时才加 `model_init_kwargs`。
- `Dataset.map(batched=True)` 回调收到的是**字段字典**（`{"question":[...]}`）而非 dict 列表 → 按位置索引取值。
- **lr 调参教训**：v1(lr 1e-6) kl≈0.0007→acc 76（策略几乎不动）；v2(lr 1e-5) kl≈0.0035→acc 71（**奖励劫持**，去拟合带噪声的过程奖励）；v3(语义改写后 lr 5e-6) kl≈0.0017→acc 78。**lr 不是瓶颈，域偏移导致过程奖励是噪声才是**——不解决域偏移，GRPO 只能在「lr 极低（不动）」和「lr 稍高（劫持）」之间选。

**域偏移（重要架构结论）**
- PRM 是用 **7B 策略轨迹**训练的，对 **32B 老师 CoT** 打极低分（min-score 均值 0.099，仅 3.9% 过 0.5）——**域偏移**：PRM 学到 7B 简洁风格，无法跨风格评判 32B markdown 详细风格。决策：PRM 只做 RL 奖励（评判 7B 策略），SFT 清洗改为**软过滤**（只删 min-score<0.02 的极端低分，大概率真含逻辑断裂），不做二分类硬截断。
- **最终解法是语义改写（层次B）**：用 32B 把 SFT CoT 改写为简洁逐步风格（见上文对照实验），既提 SFT 本身（75→77.5）又缓解域偏移；比「重训 PRM」更干净（不引入策略↔奖励循环），比「格式脱敏」彻底。
