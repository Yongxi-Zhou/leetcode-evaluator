# Batch 全链路概览

Batch 任务可以概括成一条三段式、可恢复的离线评测链路：

```text
SUBMIT -> COLLECT -> RESUME
```

其中：
- `SUBMIT` 只负责把批量生成任务发到模型侧
- `COLLECT` 只负责把生成结果拉回本地并规范化
- `RESUME` 先做 `COLLECT`，再把结果送去 LeetCode 评测，并产出最终报告

## 1. SUBMIT：创建批量生成任务

入口在 `main.py`，通过 `--qwen-batch-mode submit` 或 `--bedrock-batch-mode submit` 进入。

这一阶段会先根据 config 文件内容和绝对路径生成一个确定性的 `run_id`，所以同一份 config 永远会落到同一个 run 目录，后续可以跨会话恢复。然后对每个 `model × prompt_type`：

- 按 `stability_runs × problems` 展开请求
- 生成 `input.jsonl`
- 生成 `requests_manifest.json`，保存 `custom_id -> 题目/trial/prompt` 的映射
- Qwen 路径：上传到 DashScope Files，再创建 batch job，拿到 `job_id`
- Bedrock 路径：上传到 S3，再调用 `CreateModelInvocationJob`，拿到 `job_arn`
- 保存 `batch_job.json`
- 更新模型级 `evaluation_state.json` 为 `submitted`

本质上，`submit` 做的是把离线生成任务可靠地登记出去，不做 LeetCode 提交。

## 2. COLLECT：下载结果并规范化

这一阶段会用同样的 config 重新计算 `run_id`，定位到已有目录。对每个 `model × prompt_type`：

- 如果 `normalized_generations.json` 已存在，直接复用
- 否则刷新远端 job 状态
- 如果任务已完成，就下载原始结果
- 然后做统一规范化：提取文本、抽代码、校验 Python 语法、补齐 token/cost/error 字段
- 最终写入 `normalized_generations.json`
- 如果 `detailed` 和 `minimal` 都 ready，则模型状态推进到 `generation_completed`

Qwen 和 Bedrock 的差别主要在原始输出格式：

- Qwen 读 DashScope/OpenAI-compatible batch 输出
- Bedrock 从 S3 拉 `.jsonl.out`，并按 Claude、Nova、Llama、DeepSeek 不同响应格式提取文本

所以 `collect` 的核心职责是把供应商私有格式变成统一的本地标准结果。

## 3. RESUME：提交 LeetCode 并生成报告

`resume` 不是单纯“接着跑”，而是：

1. 先执行 `collect`
2. 确认 `detailed` 和 `minimal` 两条 prompt 链都已经完成
3. 把 `normalized_generations.json` 和 `requests_manifest.json` 重新配对
4. 按 `problem -> with_prompt / without_prompt -> trial 顺序` 组装成 `generated_items`
5. 调用 `evaluator.run_generated_evaluation()`，逐题提交 LeetCode 并轮询判题结果
6. 保存评测结果 JSON、归档 raw/summary、生成 report
7. 更新 `evaluation_state.json` 为 `completed`

也就是说，`resume` 是真正把离线生成转成最终评测结论的阶段。

## 状态机

模型级状态机是：

```text
not_started -> submitted -> generation_in_progress -> generation_completed -> submission_in_progress -> completed
                                                                                                     -> failed
```

这里的状态是模型级的，不是单个 batch job 的远端状态。远端 job 自己还有一层 provider 状态，比如 Qwen 的 `pending/in_progress/completed`，Bedrock 的 `Submitted/InProgress/Completed`。

## 目录结构

batch 中间产物放在：

```text
output/batch_jobs/{run_id}/{model_id}/{prompt_type}/
```

典型文件有：

- `input.jsonl`：发给 provider 的批量请求
- `requests_manifest.json`：`custom_id` 到题目元数据的映射
- `batch_job.json`：远端任务元数据，Qwen 是 `job_id`，Bedrock 是 `job_arn`
- `output.jsonl` / `error.jsonl`：下载回来的原始结果
- `normalized_generations.json`：统一格式后的本地结果

最终评测结果、汇总和报告不在 `batch_jobs` 下，而是在 run 级目录里，例如：

- `output/{run_id}/results/...`
- `output/{run_id}/reports/...`

## 跨会话恢复为什么成立

有三个关键设计：

- 确定性 `run_id`：同一 config 永远映射到同一目录
- 幂等缓存：`batch_job.json` 存在就不重复 submit，`normalized_generations.json` 存在就不重复 collect
- 可逆映射 `custom_id`：`model=...|problem=...|slug=...|prompt=...|trial=...`，保证每条生成结果都能精确映射回原始题目和 trial

## 一句话总结

这套 batch 链路本质上是把模型批量生成和 LeetCode 在线评测拆成两个解耦阶段，中间靠确定性目录、manifest 和 normalized 结果做持久化衔接，从而实现可中断、可恢复、可重复执行。
