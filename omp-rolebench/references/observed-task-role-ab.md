# Observed Task-Role A/B Notes

这份参考记录来自 2026-08-28 的 task role 小实验，不能替代新的 benchmark 结果，但决定了首版的执行边界：

- 真正测 task subagent 时，用临时 user/project agent 定义固定 `model` 和 `thinkingLevel`，并为每个 arm 使用隔离 workspace；只切顶层 `--model` 不等价于命中 `task` role。
- `glm-5.3-flash:max` 在一次 opencode-go 实验中因 provider 不接受实际发出的 reasoning 档位而失败。benchmark 应保留原始错误和请求档位，不把它改写成 high，也不自动换模型。
- 质量评审要检查交付报告是否主动暴露测试与标准语义的矛盾。通过固定测试不等于实现符合外部标准，建议把“测试通过”和“规范一致”分成两个字段。
- 成本、速度和质量必须分开记录。小样本的墙钟时间或 token 数只能作为该 run 的观察值，不能直接泛化为长期价格结论。
