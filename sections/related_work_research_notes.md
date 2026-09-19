# Related works 调研笔记：分布匹配、蒸馏估计器与迭代训练中的分布漂移

调研日期：2026-09-16。对应当前论文 *A Distribution-Matching Perspective on LLM Training Algorithms*。

本文是供后续研究使用的中文笔记，不是可直接放入论文的 Related Work 草稿。参考文献信息和原始链接均写在文本中；没有新增 BibTeX 条目，也没有接入 LaTeX 编译。

## 1. 调研范围与阅读方法

本次阅读了 `ce.tex` 引入的主要理论章节，以及 reward reweighting、conditional KL / MaxRL、proximal / distal gradient variance、variance accumulation 和 token learning dynamics 等证明。以论文当前的三个问题组织检索：

1. **目标分布是什么？** SFT、KD、OPD、RL 与 MaxRL 的分布匹配关系，尤其是固定目标与随策略变化的目标。
2. **实际梯度如何估计？** 蒸馏中的采样分布、KL 方向、序列与 token 粒度、基线和方差。
3. **目标为什么漂移？** 闭环自训练的抽样误差、参数化诱导的一阶偏移、二阶噪声效应，以及正确解内部的多样性。

来源以原论文、arXiv、会议论文集、期刊页面和作者机构页面为准。核心近邻阅读了相关正文／附录，其余做了摘要与书目信息核对；每条标明核验深度。文中的“对本文的意义”“建议”“推导”是本次调研的分析，不应当作被引论文已经证明的结论。对预印本不推定其已获正式会议接收；这是一轮定向检索，不构成文献穷尽性或新颖性保证。

### 优先阅读清单

| 优先级 | 文献 | 最需要核对的问题 | 当前文献库 |
| --- | --- | --- | --- |
| P0 | A9：LAD | RL 分布匹配与等优势响应的相对比例；KL 方向 | 未见 |
| P0 | D5：UCPO | 正确解条件分布、随机坍缩、逆概率加权 | 已有 |
| P0 | C3：Lost in Retraining | 闭环训练、鞅、吸收态与误差累积 | 未见 |
| P0 | B7：Fu et al. | sampled-token OPD 对序列 reverse KL 的偏差 | 未见 |
| P0 | B3 / B4：GKD / MiniLLM | OPD 的早期来源与实际目标定义 | 未见 |
| P0 | D4：Differential Smoothing | 正确轨迹内部的 sharpening 与定向修正 | 未见 |
| P1 | B8：vOPD | 保持期望梯度的方差控制 | 已有 |
| P1 | B11：OPSA | 固定负信号与 reversed self-distillation 的联系 | 未见 |
| P1 | A2 / A3：Korbak et al. | 语言模型 RL 与分布匹配的直接前序工作 | 未见 |
| P1 | D7 / D8：Natural PG / PG theory | 几何偏差、Fisher 度量与 log-barrier | 后者已有 |
| P1 | A10 / A11：FlowRL / GFlowRL | 新近的 LLM 分布匹配训练方法 | 未见 |

“未见”指本次在 `ce.bib` 中按题名和 arXiv ID 检索未见对应条目，不代表其他个人文献库没有收录。下文共整理 34 篇核心／背景文献，另给出少量延伸阅读。

## 2. 分布匹配与 RL：理论谱系及直接近邻

### A1. Khalifa, Elsahar & Dymetman：受控生成中的显式目标分布

**文献：** Muhammad Khalifa, Hady Elsahar, Marc Dymetman. *A Distributional Approach to Controlled Text Generation*. ICLR 2021；预印本首发 2020。核验：摘要与书目信息。

该工作把生成约束写成目标分布的约束，先得到 energy-based target，再用 distributional policy gradient 拟合自回归语言模型，兼顾属性满足与对原模型分布的保留。[原论文](https://arxiv.org/abs/2012.11635)

**对本文的意义：** “先明确目标分布，再研究训练算法”的思路有直接前序。本文可把增量放在目标随训练迭代改变时的动力学，而不把“语言模型训练可视作分布匹配”本身作为主要首次性主张。

### A2. Korbak et al.：RL 与 distribution matching 的直接比较

**文献：** Tomasz Korbak, Hady Elsahar, Germán Kruszewski, Marc Dymetman. *On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting*. NeurIPS 2022。核验：官方摘要及论文检索片段。

这篇直接区分 reward maximization 与预先定义目标分布后的拟合，并讨论语言模型微调中的遗忘及相应梯度估计方法。[会议页面](https://papers.nips.cc/paper_files/paper/2022/hash/67496dfa96afddab795530cc7c69b57a-Abstract-Conference.html)

**对本文的意义：** 题目、研究对象、组织视角均很接近，应优先引用。后续精读重点是目标是否冻结、采样策略如何变化、RL 与 DM 的“等价”究竟是目标等价还是算法转换。本文的统一表格可明确这些层次，避免把所有方法压缩成同一种等价关系。

### A3. Korbak, Perez & Buckley：带 KL 的 RL 是变分推断

**文献：** Tomasz Korbak, Ethan Perez, Christopher Buckley. *RL with KL penalties is better viewed as Bayesian inference*. Findings of EMNLP 2022, pp. 1083–1091。核验：官方摘要。

作者把 KL 正则化语言模型 RL 解释为对奖励诱导后验的变分近似，并从目标分布角度讨论分布坍缩。[ACL 原文页面](https://aclanthology.org/2022.findings-emnlp.77/)

**对本文的意义：** 这是连接“RL—KL—目标分布—坍缩”的重要依据。但固定 reference 的正则化 RL 与本文每轮变化的 success-conditioned target 不相同；应比较 reference、当前策略和 target 三者，而不仅比较 KL 名称。

### A4. Levine：control as inference

**文献：** Sergey Levine. *Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review*. 2018, arXiv:1805.00909。核验：摘要。

该综述用概率推断组织最大熵 RL，区分确定性动力学下的精确推断与随机动力学下的变分处理。[原论文](https://arxiv.org/abs/1805.00909)

**对本文的意义：** 附录把非二值奖励转为随机 acceptance event 的构造，与条件化／optimality-variable 谱系相邻。需要区分本文的接受概率线性依赖归一化奖励，与常见指数奖励因子的区别；也需保留 joint-space KL 中辅助变量的贡献。

### A5. Peters, Mülling & Altun：Relative Entropy Policy Search

**文献：** Jan Peters, Katharina Mülling, Yasemin Altun. *Relative Entropy Policy Search*. AAAI 2010, pp. 1607–1612。核验：官方摘要。

REPS 用相对熵约束限制策略改进中的信息损失，并给出相应的策略更新。[官方论文页面](https://ojs.aaai.org/index.php/AAAI/article/view/7727)

**对本文的意义：** ideal distribution-space update 的经典背景。精读时必须写出 KL 的两个参数；不能因为同属 relative-entropy constraint 就认为闭式更新完全相同。本文当前证明得到的是 forward constraint 下的 reciprocal tilt。

### A6. Abdolmaleki et al.：Maximum a Posteriori Policy Optimisation

**文献：** Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Rémi Munos, Nicolas Heess, Martin Riedmiller. *Maximum a Posteriori Policy Optimisation*. 2018, arXiv:1806.06920。核验：摘要。

MPO 通过相对熵目标上的坐标优化构造策略学习方法，提供将策略改进与参数策略拟合联系起来的经典入口。[原论文](https://arxiv.org/abs/1806.06920)

**对本文的意义：** 可用于区分“理想目标的计算”和“实际参数更新”。本文最值得研究的是两者之间的误差如何影响下一轮目标，而不只展示它们在单轮目标上的关联。后续应精读其 E/M 两步的 KL 方向和投影约束。

### A7. Rafailov et al.：Direct Preference Optimization

**文献：** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023。核验：原论文摘要／书目信息。

DPO 利用奖励与最优策略之间的关系，把一类偏好学习问题转成直接优化策略的目标。[原论文](https://arxiv.org/abs/2305.18290)

**对本文的意义：** 若论文声称覆盖现代 LLM 训练流水线，至少应在讨论中说明 preference optimization 与当前框架的关系。DPO 不能直接当成“给定一个显式 teacher 的 KD”；偏好数据、奖励假设和 reference 都是额外结构。它适合做范围说明，不必现在强行纳入定理。

### A8. Tajwar et al.：Maximum Likelihood Reinforcement Learning

**文献：** Fahim Tajwar et al. *Maximum Likelihood Reinforcement Learning*. 2026, arXiv:2602.02710。核验：摘要；并对照当前论文引用的有限阶目标。已有：`tajwar_maximum_2026`。

MaxRL 构造随采样预算变化的一族目标，将普通 RL 与正确事件的最大似然连接起来；无限计算极限恢复对应似然目标。[原论文](https://arxiv.org/abs/2602.02710)

**对本文的意义：** 当前附录的条件 KL 恒等式给它提供了另一种解释。后续应严格区分每个 prompt 的成功概率 ρ、log ρ，以及跨 prompt 的平均。即使单个 prompt 上梯度方向仅差正比例因子，跨 prompt 的权重也不同。不能把标准 RL 与无限阶 MaxRL 写成全局目标严格相等。

### A9. Li & Li：LAD，最直接的框架近邻之一

**文献：** Wendi Li, Sharon Li. *LAD: Learning Advantage Distribution for Reasoning*. 2026, arXiv:2602.20132。核验：v1 正文 §2–3、附录 B 的相关公式；这里按预印本版本讨论。

LAD 用优势诱导分布作为目标，通过 f-divergence 匹配策略概率比构造训练方法，并抑制高优势响应的过度放大。[原论文及正文](https://arxiv.org/html/2602.20132v1)

**对本文的意义：** “分布匹配—多模态保持—避免塌缩”的链条高度接近。尤其要比较等优势响应的相对概率，以及 target、behavior policy 的刷新频率。

**需独立核算的细节：** 所检查的 v1 中，Eq. (2) 写的是 KL(π_old‖π_new)，而 Lemma 3.1 / Eq. (3) 给出指数形式。按本文当前附录的拉格朗日推导，该 KL 方向一般对应倒数形式；指数形式对应另一方向。建议核对后续版本与证明，不直接沿用这个等价主张。这一观察只针对已读版本，不是对其全部方法有效性的判断。

### A10. Zhu et al.：FlowRL

**文献：** Xuekai Zhu et al. *FlowRL: Matching Reward Distributions for LLM Reasoning*. 2025 首发；作者机构页面标注 ICLR 2026。核验：摘要及作者代码库展示的目标。

FlowRL 把奖励转成归一化目标分布，并通过 flow balancing 进行学习；实现中含 reference、奖励尺度、长度处理及配分函数。[原论文](https://arxiv.org/abs/2509.15207)，[作者机构页面](https://www.microsoft.com/en-us/research/publication/flowrl-matching-reward-distributions-for-llm-reasoning/)，[官方实现](https://github.com/Xuekai-Zhu/FlowRL)

**对本文的意义：** 这是实际 LLM 训练中的 distribution-matching 方法，适合与 LAD 共同作为方法类近邻。不要把它的“reward distribution”理解为经典 distributional RL 的回报随机变量分布；也不要未经核算就把实际 flow loss 等同于普通 KL 的无偏估计。

### A11. Liu et al.：GFlowRL，较新的规模化延伸

**文献：** Xiaodong Liu, Michael Xu, Jack W. Stokes, Paul Smolensky, Doug Burger, Jianfeng Gao. *GFlowRL: Scaling Distribution-Matching RL to Large Language Models*. 2026-07-15, arXiv:2607.13394。核验：摘要。

GFlowRL 用 rollout group 的 Monte Carlo 估计替代额外 partition network，并结合 rollout/trainer drift 修正和 flow-gap clipping 稳定训练。[原论文](https://arxiv.org/abs/2607.13394)

**对本文的意义：** 它提醒我们：目标估计本身也可引入噪声，不能只统计 policy-gradient 的方差。若扩展框架，应区分目标估计误差、策略梯度误差与采样策略滞后；该文可作为近期实现层面的参照。

## 3. 蒸馏：采样分布、目标与梯度估计器

### B1. Hinton, Vinyals & Dean：经典 soft-target KD

**文献：** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network*. 2015, arXiv:1503.02531。核验：原论文页面与摘要。

经典 KD 以教师的软概率分布提供监督，温度用于调整分布中的信息。[原论文](https://arxiv.org/abs/1503.02531)

**对本文的意义：** 当前定理比较的是 sampled-response KD，不是所有 KD。对固定 prefix 的全词表 soft-target cross-entropy，师生相等时梯度也逐样本为零。当前 proximal 附录已经正确说明这一点，后续正文和实验的算法名称应保持同样精确。

### B2. Ross, Gordon & Bagnell：DAgger 的状态分布观点

**文献：** Stéphane Ross, Geoffrey Gordon, Drew Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011, pp. 627–635。核验：官方摘要。

DAgger 面向学习者行为会影响后续观测的序列决策问题，通过学习者访问的状态及专家监督处理训练与执行时的分布偏移。[官方论文页面](https://proceedings.mlr.press/v15/ross11a.html)

**对本文的意义：** 为 OPD 的学生前缀监督提供早期理论背景。这条线研究 covariate shift；本文的方差分析可以补充它，但不能把状态覆盖问题替换为估计方差问题。两种机制可以同时存在。

### B3. Agarwal et al.：GKD / On-Policy Distillation

**文献：** Rishabh Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. ICLR 2024；预印本首发 2023, arXiv:2306.13649。核验：摘要及会议论文目标描述。

GKD 在学生自生成序列上利用教师反馈，并允许选择不同的 divergence，而非把 on-policy 固定为唯一 KL 方向。[原论文](https://arxiv.org/abs/2306.13649)，[会议论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)

**对本文的意义：** 这是 OPD 历史与定义的必读来源。后续比较时，至少分别记录 rollout 来源、token loss、是否对采样分布求导、是否 stop-gradient。当前正文中“OPD 的关键区别就是序列 reverse KL”的说法只适合明确限定的 OPD 变体。

### B4. Gu et al.：MiniLLM

**文献：** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang. *MiniLLM: Knowledge Distillation of Large Language Models*. ICLR 2024；首发 2023。当前 arXiv v6 标题为 *MiniLLM: On-Policy Distillation of Large Language Models*。核验：当前摘要和版本记录。

MiniLLM 将 reverse KL 作为蒸馏目标，并推导 on-policy 优化方法，讨论教师低概率区域的拟合问题。[原论文及版本记录](https://arxiv.org/abs/2306.08543)

**对本文的意义：** 比 2025 年工业博客更早，且更直接关联本文的 reverse-KL 设定。建议进一步逐项比较其 reward decomposition、采样与稳定化设计，确认哪些噪声结论针对朴素估计器，哪些可迁移到实际 MiniLLM。

### B5. Ko et al.：DistiLLM

**文献：** Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun. *DistiLLM: Towards Streamlined Distillation for Large Language Models*. ICML 2024。核验：摘要。

DistiLLM 联合使用 skew KL 与 adaptive off-policy sampling，兼顾目标设计和学生生成数据的使用效率。[原论文](https://arxiv.org/abs/2402.03898)

**对本文的意义：** 可以用作纯 KD / 纯 OPD 二分之外的代表。研究 distal variance 时，改变散度的平滑程度可能和增加基线一样重要；应把目标改变与估计器改变分开比较。

### B6. Li et al.：OPD 的失效条件与 cold start

**文献：** Yaxuan Li et al. *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*. 2026, arXiv:2604.13016。核验：v2 摘要。已有：`li_rethinking_2026`。

作者研究师生 thinking-pattern 兼容性、教师是否提供新能力，以及 off-policy cold start 和 prompt selection 对失败 OPD 的改善。[原论文](https://arxiv.org/abs/2604.13016)

**对本文的意义：** 经验上的“兼容”不能直接替换成定理中的“小 KL”或“小参数距离”。更有说服力的实验是同时报告模式差异、log-ratio 尾部与梯度方差，检验这条连接，而非默认三者等价。

### B7. Fu et al.：token OPD 与 sequence OPD 的偏差／方差

**文献：** Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Yuanheng Zhu, Dongbin Zhao. *Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes*. 2026-03, arXiv:2603.25562。核验：正文及附录 B。

该文明确区分只用当前 token 信号的更新与完整 return-to-go 更新，指出前者通常对序列 reverse KL 有偏；同时分析局部信号的方差优势和 teacher-prefix mismatch。[原论文附录 B](https://arxiv.org/html/2603.25562v1)

**对本文的意义：** 当前 proximal 定理使用整个响应的 log-ratio 乘整个响应的 score，与工业实现常见的 local-token 更新并不自动相同。后续需要明确：定理直接适用于哪个 estimator；在哪些条件下 token 变体保持近端结论；远端结论是否仍然成立。

### B8. Oh et al.：vOPD 与 control variate

**文献：** Minjae Oh, Sangjun Song, Gyubin Choi, Yunho Choi, Yohan Jo. *KL for a KL: On-Policy Distillation with Control Variate Baseline*. 2026-05, arXiv:2605.07865。核验：摘要。已有：`oh_kl_2026`。

vOPD 从策略梯度角度构造可由师生输出计算的 KL baseline，在其设定下保持估计无偏并降低方差。[原论文](https://arxiv.org/abs/2605.07865)

**对本文的意义：** 是验证“方差是否有因果作用”的直接候选 baseline。但论文摘要中的 token-level value 不能不加核算地替代完整序列 return-to-go 的 value。应固定 population objective，再比较 vanilla、baseline 与全词表估计器。

### B9. Roeder, Wu & Duvenaud：Sticking the Landing

**文献：** Geoffrey Roeder, Yuhuai Wu, David Duvenaud. *Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference*. NeurIPS 2017。核验：原论文摘要及论文检索片段。

该工作通过改变变分推断中的无偏梯度估计器，获得在匹配极限下方差消失的性质。[原论文](https://arxiv.org/abs/1703.09194)

**对本文的意义：** “目标处均值为零”与“每个样本的梯度都为零”的区别有更早的理论背景。但 STL 的 pathwise estimator 与离散文本的 score estimator 不同，不能直接借用它的结论。可作为解释估计器选择重要性的旁系文献。

延伸阅读：Kyurae Kim, Yian Ma, Jacob Gardner. *Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?* AISTATS 2024。其方差界与正确／错误模型设定的处理值得参考；本次仅核验摘要。[官方页面](https://proceedings.mlr.press/v238/kim24a.html)

### B10. Armandpour et al.：OPD 信号与任务梯度的对齐

**文献：** Mohammadreza Armandpour et al. *Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why*. arXiv:2605.10889；2026-05 首发，2026-09-08 更新 v2。核验：摘要及正文诊断框架。

作者以最大化成功概率的理想 per-node gradient 为参照，用 gradient alignment 诊断不同教师／上下文的蒸馏信号，并发现适用配置依赖任务与学生。[原论文](https://arxiv.org/abs/2605.10889)，[v2 正文](https://arxiv.org/html/2605.10889v2)

**对本文的意义：** 低方差不等于任务方向正确。本文可以增加“相对各自 population gradient 的方差”和“相对任务梯度的对齐程度”两个指标，避免由前者直接推导最终能力优势。

### B11. Ding & Zhang：OPSA 与固定负优势信号

**文献：** Yi Ding, Ruqi Zhang. *Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement*. 2026-08-31, arXiv:2608.31046。核验：摘要、正文相关实验与附录配置。

该文报告固定负优势信号的实验，并提出按 token 熵调节负信号的 teacher-free OPSA，关注低 log-probability token 与概率质量再分配。[原论文](https://arxiv.org/abs/2608.31046)，[正文与附录](https://arxiv.org/html/2608.31046v1)

**对本文的意义：** 值得与 reversed self-distillation 优先对照。不过全响应统一常数、筛选 token、熵加权和实际优化器是不同设定；只有第一类在原始 on-policy score identity 下直接有零均值。不能由“负信号有效”的实验单独确认或否定本文的二阶噪声机制。

## 4. 闭环自训练、鞅与 model collapse

### C1. Shumailov et al.：递归生成数据导致的 model collapse

**文献：** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal. *AI models collapse when trained on recursively generated data*. Nature 631, 755–759, 2024。核验：期刊页面与摘要。

该工作研究模型反复使用前代生成数据训练时的退化，强调递归过程中的分布信息流失。[期刊原文](https://www.nature.com/articles/s41586-024-07566-y)

**对本文的意义：** 它提供 iterative self-distillation 的重要宏观背景。但整代数据替换／重新拟合与每个 mini-batch 后更新 teacher 并非同一个随机过程。不要把 Nature 工作的结论直接当成本文 SGD 熵定理的证明，反之亦然。

### C2. Gerstgrasser et al.：累积数据与替换数据的区别

**文献：** Matthias Gerstgrasser et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*. COLM 2024, arXiv:2404.01413。核验：摘要与会议论文首页。

该文对比递归训练中替换旧数据与累积数据；在所研究模型和理论设置下，累积真实与生成数据可避免相应的误差发散。[原论文](https://arxiv.org/abs/2404.01413)，[会议论文](https://openreview.net/pdf?id=5B2K4LRgmz)

**对本文的意义：** 是讨论“迭代是否必然坍缩”时的边界文献。建议将 teacher refresh、历史样本保留和真实数据锚点分开控制。累积数据有效并不等于在任意预算、神经网络和优化器下都保证不坍缩。

### C3. Jangjoo et al.：闭环学习中的充分统计量鞅

**文献：** Fariba Jangjoo, Giovanni di Sarra, Matteo Marsili, Yasser Roudi. *Lost in Retraining: Closed-Loop Learning and Model Collapse in Exponential Families*. Physical Review Letters 136, 197301, 2026-05-14。早期 arXiv:2506.20623（2025）题名为 *Lost in Retraining: Roaming the Parameter Space of Exponential Families Under Closed-Loop Learning*。核验：期刊摘要与早期全文关键推导。

作者对指数族的闭环最大似然重估建立充分统计量的鞅结构，讨论吸收态以及外部数据、MAP 和正则化的作用。[期刊页面](https://journals.aps.org/prl/abstract/10.1103/156q-3ngc)，[早期全文](https://arxiv.org/html/2506.20623v1)

**对本文的意义：** 这是“鞅—迭代—分布坍缩”最应精读的近邻之一。与本文的关键区别是：他们研究逐轮 MLE 的充分统计量，本文研究逐步 SGD 的参数鞅及熵曲率。不能只以研究对象不是 LLM 为由将其视作远背景；也不能把充分统计量鞅与参数鞅混为一谈。

**下一步：** 对照两种过程的状态变量、批量、重拟合程度和边界行为，确定本文能否在“部分优化、参数化和符号反转”上给出额外结论。

## 5. RL 的熵、正确解多样性与参数几何

### D1. Cui et al.：熵变化与协方差

**文献：** Ganqu Cui et al. *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*. 2025, arXiv:2505.22617。核验：摘要。已有：`cui_entropy_2025`。

该文研究 RL 推理模型的熵动态，从概率／优势相关结构解释熵变化，并提出限制高协方差 token 更新的干预。[原论文](https://arxiv.org/abs/2505.22617)

**对本文的意义：** 是当前一阶几何项的重要直接背景。本文可以把一阶均值项与二阶噪声项写在同一个展开中，展示前者为零时后者如何留下效应；这比简单按“单步研究／长期研究”区分更明确。

### D2. Jin et al.：Revisiting Entropy

**文献：** Renren Jin et al. *Revisiting Entropy in Reinforcement Learning for Large Reasoning Models*. arXiv:2511.05993，2025 首发，当前稿引用 2026 修订版本。核验：摘要。已有：`jin_revisiting_2026`。

该文系统研究训练因素与熵动态，分析正优势 token 的作用，并通过调整正负优势损失权重控制熵。[原论文](https://arxiv.org/abs/2511.05993)

**对本文的意义：** 对实验中的正／负更新分解很有用。不过重新加权正负优势通常改变均值梯度，不能作为只降低方差的干净对照。

### D3. Wang et al.：logit 更新的 entropy discriminant

**文献：** Shumin Wang, Yuexiang Xie, Wenhao Zhang, Yuchang Sun, Yanxi Chen, Yaliang Li, Yanyong Zhang. *On the Entropy Dynamics in Reinforcement Fine-Tuning of Large Language Models*. 2026, arXiv:2602.03392。核验：摘要。已有：`wang_entropy_2026`。

从单步 logit 更新的熵判别式出发，推导一阶熵变化并联系 GRPO，进而设计熵控制方法。[原论文](https://arxiv.org/abs/2602.03392)

**对本文的意义：** 应与 token-learning-dynamics 附录逐公式对照。建议比较研究的是整个词表还是正确子集、单步随机量还是期望、是否保留二阶项，以及参数共享的假设。

### D4. Gai et al.：Differential Smoothing

**文献：** Jingchu Gai, Guanning Zeng, Huaqing Zhang, Aditi Raghunathan. *Differential Smoothing Mitigates Sharpening and Improves LLM Reasoning*. 2025-11, arXiv:2511.19942。核验：摘要；其与 UCPO 的关系另参考 UCPO 的文献讨论。

该文从 selection 与 reinforcement bias 分析 sharpening，提出针对正确轨迹的 differential smoothing。[原论文](https://arxiv.org/abs/2511.19942)

**对本文的意义：** 与“应该保留正确解内部的多样性，而非一味提高全局熵”非常接近，应作为方法／理论近邻。需进一步核对其 reference KL、奖励修改形式及最优解条件，避免把其有条件的改善结论写成普遍保证。

### D5. Lochab, Li & Zhang：UCPO

**文献：** Anamika Lochab, Bolian Li, Ruqi Zhang. *Uniform-Correct Policy Optimization: Breaking RLVR's Indifference to Diversity*. 2026-05, arXiv:2605.00365。核验：正文 §4–6、附录 A.5 / A.7。已有：`lochab_uniform-correct_2026`。

UCPO 明确研究正确解条件分布，分析目标对正确解内部质量分配的不敏感性，并以条件均匀性正则和逆概率权重促进多样性。[原论文及全文](https://arxiv.org/html/2605.00365v1)

**对本文的意义：** 这是高度重合的近邻，而不应只作为一般 RL 引用。本文的 normed-ls 是局部 token 几何修正；UCPO 直接偏好正确解的均匀条件分布。比较应围绕“保持现有比例”与“趋向均匀”这两个目标，及各自的采样／估计误差。

**阅读定位：** §4 的 collapse mechanism、§5 的 conditional distribution、§6 与 A.5 的 inverse-q 权重。当前 introduction 用该 citation 支持 OPD 起源／动机，建议后续自行检查引用位置；本次没有改动源码。

### D6. Ren & Sutherland：Learning Dynamics of LLM Finetuning

**文献：** Yi Ren, Danica J. Sutherland. *Learning Dynamics of LLM Finetuning*. ICLR 2025；arXiv:2407.10490 首发于 2024。核验：摘要及会议记录。已有：`ren_learning_2025`。

该工作通过逐步预测变化的分解统一分析多种微调行为，并讨论 softmax 下的 squeezing effect。[原论文](https://arxiv.org/abs/2407.10490)，[会议论文](https://openreview.net/pdf?id=tPNHOoZFl9)

**对本文的意义：** 当前附录明确借用了其 frozen-feature/readout 设置。贡献边界应进一步落在“等优势子集的条件漂移公式”和“修正系数如何消掉相应一阶项”，而不是泛称首次解释 softmax 概率竞争。

### D7. Kakade：Natural Policy Gradient

**文献：** Sham M. Kakade. *A Natural Policy Gradient*. NIPS 2001。核验：官方摘要及论文检索片段。

自然策略梯度通过策略空间的几何度量定义更新方向，是理解参数化与策略更新关系的经典工作。[会议页面](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)

**对本文的意义：** 如果中心命题是 Euclidean 参数更新使等回报动作之间出现相对漂移，自然梯度就是必须比较的控制组。在有限动作模型中可精确计算 Fisher／自然梯度，观察无噪声时条件比例是否仍改变。

### D8. Agarwal et al.：Policy-gradient theory 与 log-barrier

**文献：** Alekh Agarwal, Sham M. Kakade, Jason D. Lee, Gaurav Mahajan. *On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift*. JMLR 22(98):1–76, 2021；早期 arXiv:1908.00261。核验：期刊书目信息和摘要；log-barrier 关联同时对照当前附录。已有：`agarwal_theory_2020` 和 `agarwal2021policygradient`。

该文系统研究策略梯度在不同参数化下的优化、近似误差与探索问题。[正式期刊页面](https://jmlr.org/papers/v22/19-736.html)

**对本文的意义：** 当前 normed-ls 附录已明确其附加项在期望上就是 KL(U‖π) 型正则；这不是全新的正则形式。潜在增量在于根据正确子集的 advantage 选择系数并证明局部漂移抵消。两个 bib key 对应同一工作的不同版本，后续引用时宜统一。

### D9. Cai et al.：RIPO

**文献：** Zhicheng Cai et al. *Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization*. 2026-07, arXiv:2607.10169；arXiv 页面标注 ICML 2026，本次未另核对会议终版。核验：摘要。已有：`cai_beyond_2026`。

RIPO 从策略几何与 clipping 的不匹配分析探索坍缩，提出相应更新方式。[原论文](https://arxiv.org/abs/2607.10169)

**对本文的意义：** 是近期几何解释方向的近邻。它的对象侧重 clipping，本文的局部漂移在未 clipping 的 REINFORCE 中就出现，两者不能直接等同。后续适合先在无 clipping 的 toy problem 中区分机制，再比较实际 RL。

## 6. 从调研得到的贡献定位与数学边界

以下是对当前稿件与上述来源的综合分析，包含本次独立推导；不是其他论文的结论转述，也不是要求立即修改正文。

### 6.1 最值得保留的主线：目标更新与实际更新之间的偏差

“训练是分布匹配”已有明确前序（A1–A6、A9–A11）。更具体的研究空间是：

> 在同一目标分布视角下，区分梯度估计误差与参数化引起的更新偏移，分析它们如何改变下一轮目标，尤其是正确事件条件分布。

这条定位同时连接固定教师蒸馏、零均值闭环训练和非零均值 RL。是否足够新颖，取决于与 C3、D4、D5 的具体定理及实验差异；不能只靠覆盖更多算法名称来确定。

### 6.2 KL 方向与“等价”的层级需要单列

固定旧分布 p，对有限动作 q 优化，可以直接核算：

\[
\arg\max_q\{\mathbb E_q[r]-\beta D_{\rm KL}(q\|p)\}
\quad\Rightarrow\quad q_y\propto p_y e^{r_y/\beta};
\]

\[
\arg\max_q\{\mathbb E_q[r]-\beta D_{\rm KL}(p\|q)\}
\quad\Rightarrow\quad q_y=\frac{\beta p_y}{\lambda-r_y},\quad\lambda>\max_y r_y.
\]

第二式就是当前 reward-reweighting 附录的形式。两者在同回报 level set 内都只乘共同系数，因而保留该集合内部的条件比例；但对一般多值奖励，它们是不同的更新。二值奖励时可用一个有效指数系数表示第二式，也不意味着原正则目标相同。

建议未来统一表格为每项标注：**目标恒等式、同最优解、当前点梯度一致、或局部近似**。当前 MaxRL 附录已有这种区分，正文的概述宜与之保持一致。

### 6.3 近端零方差结论属于估计器，不能直接提升为算法优劣

当前附录已经限定：sampled-response KD 与去掉零均值 score 项的 log-ratio OPD。这个限定很重要：

- 全词表 KD 在师生一致时也可逐样本零梯度。
- 同一个 reverse-KL population gradient，可以由不同方差的无偏估计器估计；保留额外 score 项会改变极限方差。
- 近端低方差不单独保证更快收敛或更高任务准确率，还涉及信号强度、曲率、模型失配和梯度方向；B10 提供任务梯度对齐的补充视角。
- distal 定理展示的是一种教师概率压低族上的方差放大，不是任何“大 KL”师生组合都不适合 OPD。

### 6.4 序列 KL 的链式分解不意味着梯度只需当前 token

令 r_t = log π_teacher(y_t|s_t) − log π_θ(y_t|s_t)，g_t = ∇ log π_θ(y_t|s_t)。在固定教师与常规 score-estimator 条件下，完整序列目标可以使用：

\[
\widehat g_{\rm seq}=\sum_t g_t\sum_{u\ge t}r_u.
\]

只使用 \(\sum_t r_tg_t\) 会移除当前 action 对后续访问状态和奖励的影响。B7 附录 B 明确讨论了这一区别。[对应正文](https://arxiv.org/html/2603.25562v1)

对本文的实际建议是建立一个 estimator 对照表：采样来源、是否 full vocabulary、是否 return-to-go、是否 detach prefix distribution、是否 importance correction。然后再为每个定理注明覆盖哪些行。

### 6.5 参数鞅不自动给出概率鞅，也不自动给出全局熵下降

对零均值更新 Δθ = η ĝ，若条件协方差为 Σ，且具有足够光滑性与余项控制，则局部展开为：

\[
\mathbb E[\Delta h(\theta)\mid\mathcal F]
=\frac{\eta^2}{2}\operatorname{Tr}(\nabla^2h(\theta)\Sigma)+o(\eta^2),
\quad h(\theta)=H(\pi_\theta).
\]

熵对概率向量凹，不意味着复合函数 H(π_θ) 对 θ 全局凹。当前附录已经要求沿更新段的曲率条件；这个条件决定了二阶项的符号。C3 的充分统计量鞅也不能绕过该问题。

反向常数奖励将 ĝ 变成 −ĝ，其协方差不变，所以在同一状态、同一步长、同一更新模型下，二阶噪声贡献不变；但有限步高阶项、之后访问的状态和优化器状态都可能不同。因此更稳妥的可检验预测是“在受控局部设置下二阶项具有相同符号与尺度”，而非不加条件地保证两条长期训练轨迹都单调熵降。

对 Adam、momentum、gradient clipping、截断／温度采样，需重新确认更新的条件均值是否仍为零。这里是理论覆盖范围的检查，不是预先断言这些实现一定造成何种结果。

### 6.6 全局熵、正确条件熵和 pass@k 必须分别测量

二值奖励划分响应空间时，有分解：

\[
H(\pi)=h_b(\rho)+\rho H(\pi^+)+(1-\rho)H(\pi^-).
\]

因此总体熵下降可来自成功率提高、错误分布改变或正确集合内部集中，不能仅据总体熵断言最后一种。D4、D5 与本文在这里相邻。

还有一个容易遗漏的评估边界：**固定 prompt、独立同分布采样且使用同一二值 verifier 时，理论 pass@k = 1−(1−ρ)^k，只依赖总成功率 ρ。** 在保持 ρ 不变时，仅改变 π⁺ 的形状不会改变这个 pass@k。

所以“正确解多样性提高导致 pass@k 提高”不是单个 prompt 下的直接恒等关系。跨 prompt 平均可因成功率分配不同而变化，训练／泛化也可能将多样性变化转化为成功率变化，但这些要另做证据。建议增加正确轨迹的语义／解法聚类、多样性、条件熵和固定初始正确分布的漂移指标。已有论文中 pass@k 与 diversity 的相关发现，应与上述数学关系同时解读。

### 6.7 normed-ls 的价值应落在系数与漂移，而非正则形式的新颖性

当前附录已给出，在采样 p 和 inverse-probability 权重冻结时：

\[
\mathbb E_{Y\sim p}\left[-\frac{\lambda}{Vp_Y}\log\pi_\theta(Y)\right]
=\lambda D_{\rm KL}(U_V\|\pi_\theta)+\lambda\log V.
\]

这提供三条清楚的比较线索：

- 与 D8 比较：已有 log-barrier／uniform-to-policy KL，本文额外解释何时选 λ=a 可以抵消等优势子集的一阶漂移。
- 与 D5 比较：对整个词表的 uniform regularizer 和对正确响应集合的 uniformity objective，其支持集、粒度和最优分布不同。
- 与 D7 比较：自然梯度是否在同一有限动作设置中保留相应比例；normed-ls 与自然梯度不能仅因都含概率修正就视为相同。

**一个有用的实现对照（本次推导）：** 若已计算完整词表 logits，可直接求 \(-\lambda V^{-1}\sum_j\log\pi_\theta(j)\) 的梯度，与 inverse-probability sampled correction 对比。二者在上述冻结条件下均值相同，但后者会有额外抽样噪声；完整求和版本仍保留 prefix sampling 等其他噪声。这适合检验“消一阶漂移却引入二阶噪声”的权衡。

## 7. 后续研究与实验建议

这些是建议开展的工作，本次没有运行训练或补充论文实验。

| 问题 | 建议对照 | 应报告的证据 | 可区分的解释 |
| --- | --- | --- | --- |
| 近端优势是否来自 estimator？ | sampled-response KD、full-vocabulary KD、sequence OPD、local-token OPD、带 baseline 的 OPD | 均值梯度、trace covariance、任务梯度对齐、相同计算预算下训练进展 | 分布方向、方差、soft labels 的各自作用 |
| distal failure 是否由 log-ratio 尾部驱动？ | 固定学生，逐渐压低教师在指定响应集合上的概率；再做自然师生组合 | log-ratio 分位数、score 范数、方差与理论 log² 尺度 | 条件反例能否解释真实训练 |
| 零均值噪声是否产生熵漂移？ | 可枚举 softmax 模型，奖励 +1 / −1 / 0；精确期望与 Monte Carlo | 条件均值、协方差、熵曲率、不同种子均值和区间 | 一阶漂移与二阶噪声 |
| 方差累积的步长／batch 规律？ | 控制总 token 预算与更新次数，分别扫描 η 和 m | 与 Σ_k η_k²/m_k 的局部尺度对照 | 样本数、更新数与步长的混杂 |
| 几何偏移能否在无噪声下发生？ | 精确 Euclidean PG、自然梯度、理想概率空间更新、normed-ls | 正确集合的 pairwise log-odds、H(π⁺)、成功率 | 参数化偏移与抽样偏移 |
| 修正项是否值得其方差成本？ | sampled inverse-p、完整词表正则、截断 inverse-p、不同 λ | 更新均值偏差、方差、稀有 token 更新、条件漂移 | 估计器与目标本身的贡献 |
| 闭环刷新是否放大漂移？ | 每步刷新、每多步刷新、固定 teacher、混入历史／外部样本 | 相对初始目标的漂移和熵轨迹 | 本文 SGD 过程与 C2 / C3 的重拟合过程 |

对于零均值条件，建议从 plain SGD、温度 1、无 top-k / top-p 截断、固定长度或清楚定义的 EOS 终止规则开始，再逐项加入实际训练实现。特别是“所有奖励都相同”的 GRPO 若做组内中心化，优势可能全为零；这与未中心化的常数奖励 REINFORCE 不是同一个实验。

建议先做小规模精确计算：它能直接观察真正的 population gradient 与条件熵，而大语言模型中的这些量通常只能近似估计。随后再用实际模型检验局部结果是否能够解释可观测行为。

## 8. 后续撰写时的文献组织方式

可以按以下四段组织未来 Related Work，但不必照搬本笔记中的表述：

1. **Distribution matching and policy improvement：** 从 A1–A6 的语言模型分布控制、推断与相对熵更新，接到 A8–A11 的 MaxRL / LAD / FlowRL，明确已有框架与本文动态分析的差别。
2. **On-policy distillation and gradient estimation：** B1–B5 交代 soft-target KD、状态分布问题与 OPD 历史；B6–B11 聚焦适用边界、估计偏差、方差和教师信号质量。
3. **Recursive learning and stochastic collapse：** C1–C3 说明数据反馈闭环、不同迭代定义和鞅结构，解释本文的 SGD 更新级分析与它们如何衔接。
4. **Entropy and correct-solution diversity in RL：** D1–D6 讨论一阶熵动态、正确条件分布与 sharpening，D7–D9 衔接几何解释和修正。

在目前主线上，UCPO、LAD、Lost in Retraining、Fu et al. 和 Differential Smoothing 值得逐公式阅读；工业模型技术报告可用于说明使用背景，但无需占据 Related Work 的主体。PPO、GRPO、RLOO 等当前已引用算法更适合放在问题定义与实验基线处，并精确注明具体 estimator 版本。

## 9. 引用与核验提醒

- 本次实际打开的来源已在各条提供。对有精确公式讨论的近邻，链接保留了所读版本；一般摘要链接可随 arXiv 后续更新。
- GKD / MiniLLM 的工作起点可追溯到 2023 预印本与 2024 会议，不能只依据 2025 工业实践文章叙述 OPD 起源。
- `ren_learning_2025` 与 `jin_revisiting_2026` 的年份反映会议／修订版本，不等于预印本首发年；引用时宜统一规则。
- MiniLLM 与 Lost in Retraining 存在题名版本变化，建议最终导入参考文献时选择对应的正式版，并保留 arXiv ID 便于去重。
- 本文中的“已有／未见”只用于帮助补充当前 `ce.bib`。本次没有改写参考文献库，也没有调整任何论文引用。
- 对仅做摘要核验的工作，本笔记不声称检查过全部证明或复现其性能数字。优先精读清单和实验建议用于指导下一轮核验。
