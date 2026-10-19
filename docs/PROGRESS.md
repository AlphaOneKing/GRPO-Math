# 项目进度与交接记录

> 每次新开会话，先让 Claude 读 README.md 和本文件。

## 环境
- 本地：Apple Silicon Mac，conda 环境 `grpo`（Python 3.11）
  torch 2.14.1 (MPS)、transformers 5.19.0、trl 1.15.0、peft 0.21.2、datasets 5.1.0
- Mac 没有 Triton：`src/mac_compat.py` 会自动换成纯 PyTorch 的 log 概率实现（只在 Mac 上生效）
- 正式实验：计划在租的 Linux GPU 服务器上跑（还没租）

## Day 1（已完成，10-13）
- [x] 依赖安装，15 个单元测试全部通过
- [x] 冒烟训练（0.5B，2 步）和冒烟评测（10 题）流程跑通
- 修复：TRL 在 Mac 上缺 Triton → mac_compat；`warmup_ratio` 改为 `warmup_steps`；eval.py 改用 `dtype`

## Day 2 待处理（重要）
冒烟结果显示 0.5B 模型**完全不遵守 `<think>/<answer>` 格式**：
- 评测：严格准确率 0%、格式正确率 0%，但宽松准确率 30%（模型会算，只是不按格式输出，常用 `\boxed{}`）
- 训练：两项奖励全是 0，`frac_reward_zero_std=1` → **没有任何学习信号**
- 另外补全长度 128 时 87.5% 被截断
需要在正式训练前决定：
1. 先用 1.5B、补全长度 512 测一下格式遵守率
2. 如果仍然很低：baseline 加入 soft_format 奖励，或者让正确性奖励也识别 `\boxed{}` / 最后一个数字
3. 这件事本身是“奖励稀疏导致 GRPO 无信号”的好面试素材，记录下来

## Day 2（已完成，10-13）
已完成（Claude 在 Linux 虚拟机里做的，不需要模型）：
- [x] 审查 `src/rewards.py`，修复解析误判：Unicode 减号 `−5`、`.5`、分数 `3/4`、`m^2`/`cm²` 里的指数被当成答案
- [x] 严格格式更严：恰好一个 think + 一个 answer，不允许重复或嵌套标签（之前两个 `<answer>` 也能拿 0.5）
- [x] `tests/test_rewards.py` 加了 4 组边界用例（10 个测试全过；旧版本备份在 `src/rewards.py.day1.bak`）
- [x] 新增 `answer_num_count()`；`eval.py` 逐条输出里加了 `nums_in_answer`、`correct_lenient`，并打印歧义答案数
- [x] 新增 `probe_format.py` + `day2.sh`（学习信号诊断 + base 评测 + 抽 20 条样本）

**诊断结果（probe_format.py，1.5B，16 题 × 8 采样，T=1.0，512 token）**：
| 指标 | system 风格 | user 风格 |
|---|---|---|
| 严格格式率 | 52% | 62% |
| 严格准确率 | 28% | 39% |
| 宽松准确率 | 54% | 55% |
| 截断率 / 平均长度 | 0.8% / 183 | 2.3% / 158 |
| 歧义答案率 | 17% | 27% |
| zero_std（baseline 奖励） | 0 | 0.0625（1 组 8/8 全对） |
- 结论 1：1.5B 有学习信号，0.5B 的“奖励全 0”问题不再存在 → 不需要 soft_format 或 \boxed 兜底
- 结论 2：两种 prompt 写法模型能力一样（宽松准确率相同），差别只在格式遵守
- 结论 3：解析器保持“取最后一个数字”。歧义答案基本是 `a + b = c` 这种算式，取第一个数字一条都救不回来；有标签但被误判的情况为 0
- 面试素材：0.5B 奖励稀疏 → frac_reward_zero_std=1 → 无梯度；换 1.5B 后格式遵守率约 50%，信号恢复
- **已定（10-13）：所有评测和训练都改用 user 风格**（格式说明追加在题目后面，不传 system）。理由：RL 的收益应该和最好的 prompt 比，不然会混进约 10 个点“靠改 prompt 就能拿到”的格式收益
  - 改动：`src/data.py` 加了 `FORMAT_HINT`、`PROMPT_STYLE="user"`、`question_of()`，旧写法保留为 `style="system"`（旧文件备份在 `src/data.py.day1.bak`）；probe 改为复用 data.build_prompt；eval/train 会记录 prompt_style；新增 `tests/test_data.py`
  - 注意：results.csv 里的 smoke 行用的是旧的 system 写法，不和后面的结果比
- ⚠️ **10-13 更正：下面这次 Mac base 评测作废**（已改名为 `base_mac_reppen1.1_INVALID`）。HF `generate()` 继承了 Qwen
  generation_config 里的 `repetition_penalty=1.1`，惩罚范围包括 prompt，把格式说明里出现过的 `<think>/<answer>` 压了下去。
  服务器 vLLM（惩罚=1.0，与训练一致）前 200 题：严格 44% / 宽松 57.5% / 格式 71%，而 Mac 同样 200 题是 18% / 62% / 20.5%。
  所以“贪心格式率只有 27%”是配置造成的假象。eval.py 已显式设 `repetition_penalty=1.0`，base 在服务器上重测。
  面试素材：评测要显式写全解码参数，别继承模型自带的 generation_config；两套后端交叉验证发现了这个问题
- [x] ~~base 评测完成（Mac MPS，1319 题，贪心，512 token，约 20 分钟）：~~（作废，保留记录）
  **严格准确率 19.3% / 宽松准确率 61.9% / 格式正确率 26.6% / 平均 195 token / 截断 0.8%**，已填进 README 第 4 节
- 发现 1：**贪心解码下格式率只有 27%，T=1 采样下是 62%**。52% 的回答完全没有标签，开头就是 “To determine…”，
  因为贪心每一步选概率最大的 token，而很多题的第一个 token 不是 `<think>`。训练用的是采样，所以 GRPO 有信号；
  RL 提升格式的效果在贪心评测上会被放大。面试点：严格和宽松之间 42 个点的差距几乎都是格式，不是能力
- 发现 2：格式正确的回答准确率 63%，和整体宽松准确率一样，说明按格式写不会让模型变笨
- 发现 3（20 条人工检查）：严格判分 0 误判；错误多是真算错，另有一类是“过早给答案”，`</think>` 前写着 “Now let's solve…” 就直接给了答案（如 #3、#8）。
  宽松判分会漏判结尾句里带别的数字的回答（#10 “2 bagels”、#19 “3 miles”），全量约 2.7%，所以 61.9% 是下界
- 发现 4：格式分类统计——无标签 51.9%、严格格式 26.6%、只有 think 没 answer 13.1%、\\boxed 3.5%、其他 3%

## 服务器（10-13 租好）
- AutoDL 租用 RTX 4090（48GB 显存版）
- 项目放 `/root/autodl-tmp/grpo-math`（数据盘），HF 缓存 `/root/autodl-tmp/hf_cache`，用 hf-mirror
- Claude 的环境连不上 SSH（网络只放行 HTTP），服务器命令由用户执行；结果用 rsync 拉回 Mac 文件夹给 Claude 读
- `server_setup.sh`：装环境 + 测试 + 10 步冒烟（测显存峰值和每步耗时）+ 前 200 题评测一致性检查
- `day3.sh`：nohup 后台跑 train + eval，`EXP=xxx` 可跑任意实验

### 服务器环境踩坑（10-13）
- `pip install trl[vllm]` → torch 2.13.0 + vllm 0.31.0 + trl 1.15.0 + transformers 5.17.0（Python 3.12）
- 坑 1：hf-mirror 不支持 Xet 协议，下载模型权重 401 → 脚本里加 `HF_HUB_DISABLE_XET=1`
- 坑 2：`import vllm` 报 `duplicate template name`，导致 TRL GRPOTrainer 也导入失败（装了 vLLM 就会被 TRL 加载）。
  原因：AutoDL 镜像里旧版 torch 的 76 个 .py 文件没被卸载干净（如 `_inductor/kernel/flex_attention.py`，新版已搬到 `kernel/flex/`），
  inductor 把同一个模板注册了两次。修法：对照 `torch-2.13.0.dist-info/RECORD`，把不在清单里的 .py 挪到
  `/root/autodl-tmp/stale_torch_files`（备份，没删）。之后 `import vllm` + `from trl import GRPOTrainer` 正常
- server_setup.sh 已加防护：GRPOTrainer 导入失败或下载失败会直接停下

### GPU 冒烟结果（10-13，4090 48GB，vLLM colocate）
- 显存峰值 19.5 GB（vLLM 占 0.3×48≈14GB）→ g16 也放得下
- 每步约 4.6 秒 → 300 步约 25 分钟（比 README 估的 2–4 小时快很多；300 步只用到约 1200 道训练题）
- 前 10 步：格式奖励均值 0.33→0.37（满分 0.5），正确性 0.42–0.44，frac_reward_zero_std 0.05，平均长度约 150 token
- 模型从 ModelScope 下到 `/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`（比 hf-mirror 快）

## Day 3（已完成，10-13）
- base（vLLM，repetition_penalty=1.0）：严格 45.4% / 宽松 56.8% / 格式 74.7% / 142 token / 截断 0.5%
- baseline（GRPO 300 步，约 20 分钟）：严格 74.8% / 宽松 74.8% / 格式 99.1% / 162 token / 截断 0.8%
- 训练曲线（每 30 步平均）：训练准确率 0.45 → 0.70（35–60 步）→ 0.80（125 步）→ 稳定在 0.78–0.82；格式奖励约 40 步饱和；
  熵 0.36 → 0.16；frac_reward_zero_std 0.08 → 0.53；长度 140 → 164；截断率始终 < 1%
- 错误回答训练后变长（191 vs 152 token），训练前两者一样长 → 长度偏差的迹象，Day 5 重点看
- **步数决定：保持 300 步。** 125 步后训练准确率已进入平台期（最后 150 步只多了约 2 个点，在噪声范围内），
  而且一半的组已经没有梯度，加步数收益小，GPU 时间优先给消融实验。可选：评测 checkpoint-100/200，画测试集曲线
- 详细结论已写进 README 第 4 节

## Day 4（已完成，10-13）
- 测试集：baseline 74.8 / no_format 75.1 / g4 74.5 / g16 73.2 / kl0.04 73.2（严格准确率 %）；McNemar p 均 > 0.15，都不显著
- 训练动态：末段熵 kl0.04 0.214 > g4 0.176 > baseline 0.159 > no_format 0.132 > g16 0.112；零方差组 g4 0.63 最高
- 完整报告（含熵曲线图）：Claude Docs「GRPO-Math 消融实验报告（Day 3–4）」https://claude.ai/code/artifact/06448fda-b9e2-49e7-88f2-c6f373ffa257
- plot_curves.py 修了颜色：每个实验固定颜色（之前 KL 面板只有 kl0.04 一条线却画成 baseline 的蓝色）
- ⚠️ **重要发现：PPO 裁剪从未触发**（clip_ratio 全程 0）。num_iterations=1 → 完全 on-policy，比率恒为 1。
  Day 5 的 dapo_clip_higher 按原样跑会是空操作

## Day 5 待办
- [x] train.py 加了 `num_iterations`（预设 + `--num_iterations` 参数；TRL 不支持时直接报错）。旧版备份 train.py.day4.bak
- [x] 设计改为两组，每组内一次只加一个改动（原计划 baseline_mu2 vs dapo_clip_higher 同时改了损失类型和裁剪，不能归因）：
  μ=1：baseline → dapo_token、baseline → dr_grpo；μ=2：dapo_token_mu2 → dapo_clip_higher → dapo_full
- [x] `day5.sh`：先 10 步冒烟确认 μ=2 下 clip_ratio > 0（否则停止），再串行跑 5 个实验，画两张图，写 logs/day5_summary.md
- [x] `analyze_runs.py`：准确率 + 答对/答错长度差 + McNemar + 训练末段熵/零方差组/裁剪比例，一张表
- [x] day5.sh 跑完（12:25–13:58）。dapo_full 训练完成但评测失败：数据盘满（11 个 final_merged × 3GB）。
  修法：`rm -rf outputs/*/final_merged` 后重评 dapo_full；day3.sh 已改为评测后自动删 final_merged
- ⚠️ **更正 Day 3 的“长度偏差”结论**：baseline 答错的题，在 base 模型下回答本来就长 34 token（题目也长 26%），
  是题目难度造成的。扣掉后 GRPO 训练只多出 +7 token 的差距；Dr. GRPO 没有缩小它（+19）。README/报告/简历都要改
- 评测口径（10-13 确认）：所有对比都在 GSM8K 测试集上；没有验证集，但超参和 checkpoint 都是事先固定的，
  没在测试集上做选择。**不要**用测试集评 checkpoint-100/200 来选步数；要做就先从训练集划验证集。已写进 README 局限
- [x] dapo_full 重评完成（74.8%）。Day 5 结果（严格准确率，p 为对各自参照组的 McNemar）：
  dapo_token 73.1%（vs baseline p=0.11）· dr_grpo 76.2%（p=0.17）· dapo_token_mu2 74.4% · dapo_clip_higher 75.5%（vs mu2 p=0.28）· dapo_full 74.8%（vs clip_higher p=0.58）
  末段熵：clip_higher 0.162 > mu2 0.127；dapo_full 0.088 最低。长度差净增（扣掉 base 难度）：baseline +7、dapo_token +10、dr_grpo +19、mu2 +21、clip_higher +26、dapo_full +24
- 结论已写进 README 第 4 节（7–9 条，第 4 条已更正）
- [ ] 自学：读 src/grpo_loss.py 和 tests/test_grpo_loss.py，白纸推出 GRPO 的长度偏差（理论上存在，但本项目数据里很弱，面试时两点都要讲）

## Day 6 case study（已完成，10-13，不需要服务器）
- 提升主要来自“把推理做完”：错→对 317 题中 137/272（50%）base 的错误答案是正确过程的中间结果
- 对→错 79 题：长推理里新引入的漏条件/算错；部分 base 是“蒙对”的（伪代码直接报答案）
- baseline 333 错题：理解/推理 289（87%）· 算术 32（10%，自动校验 a op b = c，下界）· 截断 10 · 格式 2
- 10 个 RL 模型全错 106 题，随机 18 题人工分类：漏/误读条件 7 · 多步追踪 8 · 算术 2 · 题目歧义 1；Q454 标注疑似错误
- 写进了 README 第 4 节第 10 条和报告的 Case study 一节；简历条目定稿在 resume_project.md

## 项目主线全部完成（10-13）。剩余可选：补随机种子、过程奖励（算术校验，预期最多影响约 10% 的错题）、hack_length
- [ ] 重点看：Clip-Higher 是否减缓熵坍缩；Dr. GRPO 是否消除“错误回答比正确回答长约 40 token”的长度偏差

## （旧）Day 3 待办
- [x] 租 GPU 服务器
- 服务器上：`pip install -r requirements.txt && pytest tests/ -q`，然后 `python train.py --exp baseline --use_vllm --wandb`
- 训练后：`python eval.py --model outputs/baseline/final --tag baseline`，和 base 的 19.3% / 61.9% / 26.6% 对比
- 看曲线时重点关注：格式奖励多少步到接近 0.5、frac_reward_zero_std 是否随准确率上升而上升（全对组变多）

**（以下为跑诊断前记录的待定项）**：
- 已知局限：`<answer>18 (9 eggs x $2)</answer>` 会取到 2（取最后一个数字）。看 `ambiguous_answer_rate`，高的话再决定改成取第一个数字，还是把多数字答案判错
- 格式问题的方案：看 probe 汇总里 baseline 的 `zero_std`。若 > ~0.8：
  ① prompt 改成 user 风格（看 system 和 user 两列哪个格式率高）② baseline 加 soft_format ③ 正确性奖励接受 `\boxed`/最后一个数字
- 跑完后要做的：在 Mac 上执行 `bash day2.sh`，然后对 Claude 说“Day2 脚本跑完了”

## 常用命令
```bash
cd ~/Documents/Claude/grpo-math && conda activate grpo
SKIP_INSTALL=1 bash day1.sh   # 日志在 logs/day1.log
bash day2.sh                  # 日志 logs/day2.log，样本 logs/day2_samples.txt
SKIP_EVAL=1 bash day2.sh      # 只跑诊断（约 10–20 分钟），先决定格式方案
```
