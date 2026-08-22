---
layout: post
title: "Fatal-Aware GRPO：多轮 Tool Calling 中的信用分配问题"
date: 2026-08-01 09:00:00 +0800
categories: [Agent, Reinforcement Learning]
tags: [GRPO, Credit Assignment, Tool Calling]
description: "多轮工具调用信用分配问题文章占位。"
draft: true
published: false
math: true
---

> TODO：本文仅验证长文、代码、表格与数学公式排版，不陈述实验效果。

## 问题定义

TODO：定义多轮轨迹中的普通错误与 fatal failure。

### 一个待验证的形式

以下公式仅作为排版占位，最终定义需结合真实训练实现核对：

$$
A_i^{fatal} = \max(\tilde A_i, 0)
$$

## 信用分配

TODO：说明 token、turn 与 trajectory 层级的归因选择。

| 层级 | 待回答的问题 |
| --- | --- |
| Turn | 哪一步引入不可恢复错误？ |
| Trajectory | 最终结果如何回传？ |

```python
# TODO: replace with verified training code.
def fatal_aware_advantage(raw_advantage):
    return max(raw_advantage, 0.0)
```

## 实验设计

TODO：补充基线、消融、失败类型统计和可复现配置。
