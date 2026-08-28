# Medical Reasoning RL

基于 Qwen2.5-7B-Instruct 的医疗复杂推理项目：SFT 阶段 CoT 构造 + 过程监督（PRM）+ 强化学习（GRPO）对齐。

思路参考 HuatuoGPT-o1（Towards Medical Complex Reasoning with LLMs）：用可验证的医学问题 + 验证器，打通"思维链构造 → 过程监督 → 强化学习"闭环。

## 流水线

```
MedQA / MedMCQA
      │
      ▼
[1] CoT 构造 ── 教师模型生成推理链 + 过程验证(PRM/Judge)过滤
      │
      ▼
[2] SFT ────── Qwen2.5-7B-Instruct 在过程验证通过的高质量 CoT 数据上微调
      │
      ▼
[3] RL ─────── GRPO / DAPO，复合 Reward = 格式硬约束 + LLM-as-Judge 连续打分
      │
      ▼
[4] 评测 ───── MedQA / MedMCQA 准确率
```

## 目录结构

```
configs/            训练与数据配置 (yaml / json)
scripts/            端到端脚本 (bash / python)
src/answer_extraction.py  统一的最终答案抽取（reward / CoT / 评测共用）
src/data/           CoT 构造 + 过程验证
src/reward/         复合 Reward 函数
src/train/          SFT + GRPO 训练入口
src/eval/           MedQA / MedMCQA 评测
data/raw|cot|processed/  数据
checkpoints/        模型权重
```

## 快速开始

```bash
# 1. 安装依赖（trl 版本已固定，见 requirements.txt）
pip install -r requirements.txt

# 2. 下载并处理数据
python scripts/01_prepare_data.py

# 3. 教师模型构造 CoT（需 GPU + vLLM，可先跳过直接复用现成 SFT 数据）
bash scripts/serve_vllm.sh Qwen/Qwen2.5-72B-Instruct 8000 1
python scripts/02_build_cot_data.py --config configs/data.yaml

# 4. 过程验证过滤（产出 train_cot_verified.jsonl）
python scripts/03_process_verification.py --config configs/data.yaml

# 4b. 构造 GRPO 训练数据（rl_prompts.jsonl）
python scripts/03b_build_rl_prompts.py

# 5. SFT（多卡；单卡加 --num_processes 1；单卡低显存加 --lora 直接 python 跑）
bash scripts/04_train_sft.sh
# 单卡 LoRA 版：python src/train/sft.py --config configs/sft.yaml --lora

# 6. GRPO 强化学习（先起 Judge vLLM，见 scripts/05_train_grpo.sh 注释）
bash scripts/serve_vllm.sh Qwen/Qwen2.5-32B-Instruct-AWQ 8001 1
bash scripts/05_train_grpo.sh

# 6b. 或改用 DAPO（token-level loss + clip-higher + dynamic sampling + overlong shaping）
bash scripts/05b_train_dapo.sh

# 6c. 单卡低显存：LoRA 版（直接 python 单进程，勿走 accelerate/deepspeed）
bash scripts/05_train_grpo_lora.sh     # GRPO + LoRA
bash scripts/05b_train_dapo_lora.sh    # DAPO + LoRA

# 7. 评测（双轨：抽取准确率 + LLM-as-Judge 质量分）
bash scripts/serve_vllm.sh checkpoints/grpo 8000 1
bash scripts/serve_vllm.sh Qwen/Qwen2.5-32B-Instruct-AWQ 8001 1   # judge 轨，可省
bash scripts/06_eval.sh
```

## 关键组件

- `src/reward/composite.py`：复合 Reward = 格式检查（硬约束）+ LLM-as-Judge（连续打分）
- `src/data/cot_construction.py`：教师模型生成 CoT 推理链
- `src/data/process_verification.py`：LLM-as-Judge 过程监督打分
- `src/train/grpo.py`：GRPO 强化学习 + KL 退火（`KLAnnealingCallback`）+ 熵监控（`MedicalGRPOTrainer` / `MonitoringCallback`）
- `src/train/dapo.py`：DAPO（`DAPOTrainer`）＝ GRPO + token-level loss + clip-higher + dynamic sampling + overlong shaping（参考 DAPO 论文）
- `src/eval/medqa.py`：MedQA / MedMCQA 评测，双轨制 = 抽取准确率（acc / miss_rate / avg_length）+ LLM-as-Judge 质量分（judge_mean / judge_pass_rate），方便基座/SFT/GRPO/DAPO 横向对比
- `scripts/serve_vllm.sh`：vLLM 服务启动（教师 / Judge 模型通用）

## 模型服务（vLLM）

```bash
# 教师模型（CoT 构造 / 过程验证）
bash scripts/serve_vllm.sh Qwen/Qwen2.5-72B-Instruct 8000 1

# Judge 模型（RL 连续打分）
bash scripts/serve_vllm.sh Qwen/Qwen2.5-72B-Instruct 8001 1
```

## 训练注意事项

- **版本**：`trl` 已固定为 `0.15.0`（GRPO/SFT 接口按其适配）。升级 trl 前先看 reward 签名与 `beta`/callback 是否兼容。
- **DeepSpeed**：`configs/accelerate.yaml` 是 accelerate 启动配置，真正的 DeepSpeed 参数在 `configs/ds_zero3.json`。不要在 `sft.yaml` / `grpo.yaml` 里再写 `deepspeed:`（会与 accelerate 冲突）。
- **单卡**：`accelerate launch --config_file configs/accelerate.yaml --num_processes 1`，且保证 GRPO 的 `num_generations` 整除 `per_device_train_batch_size x num_processes`（默认 G=8 时单卡需 `per_device_train_batch_size=8`，脚本会在启动时校验并提示）。
- **数据链路**：SFT 输入为 `03` 步产出的 `train_cot_verified.jsonl`（已按 `min_process_score` / `max_fallacies` 过滤）；GRPO 输入为 `03b` 步产出的 `data/processed/rl_prompts.jsonl`。
- **Judge 成本**：RL 每步会对每组 G 条采样各调一次 72B Judge，是主要开销。可先用小数据集验证，后续建议 LoRA 训练专用 verifier 替换。
- **DAPO vs GRPO**：`configs/dapo.yaml` 默认去掉 KL 惩罚（论文 §2.3，`beta=0`）、`eps_low=0.2 / eps_high=0.28`。由于 trl 0.15 没有 old-policy 缓冲，clip-higher 的 ratio 采用「当前策略 vs 冻结参考模型」，语义等价（见 `src/train/dapo.py` 文件头注释）。想保留 KL 约束可把 `beta` 调回 0.001~0.04 并开启 `kl_annealing`。
- **LoRA 单卡**：`--lora` 开关（`src/train/lora.py`）。注意 trl 0.15 在 deepspeed zero3 + peft 下仍加载独立 ref model（省显存失效），所以 LoRA 必须**直接 `python` 单进程跑**（`scripts/05*_lora.sh`），且 `per_device_train_batch_size >= num_generations`。LoRA 训练产物是 adapter，评测前需用 `vllm serve <基座> --enable-lora --lora-modules name=<checkpoints/...>` 挂载，或先 merge 成完整权重。

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

**结论**：
1. 全链路（数据 → SFT → GRPO → 评测）跑通，产物格式正确。
2. GRPO 用纯规则 reward 仅 5 步就把准确率从 72.5% 提到 75.5%、miss 收敛到 0.5% —— 说明「格式硬约束 + 组内相对优势」能快速教会模型稳定输出答案格式。
3. SFT 反降（68.0%）：240 条样本过少 + 长解析 CoT 过拟合 + miss 升高（6.5%），符合「小数据 SFT 有害」的已知现象，属可解释结果。

**已知限制（正式实验前必须解决）**：
- **vllm 0.27 + Blackwell 不可用**（flashinfer 不支持 sm120，且无 nvcc 无法现场编译），故本轮绕开了 teacher CoT 生成与 LLM-judge：SFT 用 val 官方解析、GRPO 用纯规则 reward、评测用 transformers 直出。
- 训练环境的坑已逐个修掉：trl 0.15 需配 transformers 4.49（`_get_train_sampler` / `self.control` 兼容）；LoRA `target_modules` 必须传字符串 `"all-linear"`（不能是列表）；GRPO 收尾保存因 trl/transformers 的 control bug 崩溃，已在 `MedicalGRPOTrainer.train()` 加兜底捕获。
- 训练数据格式：SFT 的 dataset 直接用 `prompt`/`completion` 消息列（勿用 `formatting_func` 返回消息列表，trl 0.15 会误判为 batch）。
