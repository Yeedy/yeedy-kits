# Yeedy Kits

个人 Codex skills、MCP 和 plugin 源码工作区。`skill/` 下每个子目录对应一个可独立安装和分享的 skill。

## Skills

### ai-coding-daily-coach

结合用户的发展方向和本地 Codex 项目，生成每日 AI Coding 概念题与场景题，并管理评分、复习记录和间隔复习计划。

主要能力：

- 从全部可识别的 Codex workspace 或用户指定目录生成学习画像
- 支持初始化、刷新画像和更新个人发展方向
- 保存待答题、历史评分、薄弱概念和复习计划
- 优先使用 Exa 检索资料，不可用时回退到默认 Web Search
- 将运行状态保存在使用该 skill 的 workspace，不写入 skill 安装目录

详见 [skill/ai-coding-daily-coach/SKILL.md](skill/ai-coding-daily-coach/SKILL.md)。

### optimize-agent-memory

审计和精炼 Codex 或 Oh My Pi memory。默认覆盖当前 agent 的全部 memory；也可以明确指定 OMP、Codex、两者，或进一步缩小到某个项目、workspace 或类别。识别过时、重复、低价值或过度膨胀的记忆，并通过对应 agent 支持的持久化入口安全修正。

详见 [skill/optimize-agent-memory/SKILL.md](skill/optimize-agent-memory/SKILL.md)。

### omp-rolebench

对 Oh My Pi 的模型与 `modelRoles` 做可重复 benchmark。支持全量/供应商/显式模型筛选、全部/指定角色、通用或近期工作任务源、单角色隔离和全角色同模型分配，并在执行前展示方案、估算成本并要求确认，最后输出 JSONL 原始结果与 Markdown 报告。

详见 [skill/omp-rolebench/SKILL.md](skill/omp-rolebench/SKILL.md)。

### web-ui-recreator

按指定设备范围观察网页的布局、组件状态、交互和动态效果，再按要求复刻页面或将风格转译到新主题，并核对视觉与功能。共用一份 `SKILL.md`，适用于 Codex 和 OMP。

详见 [skill/web-ui-recreator/SKILL.md](skill/web-ui-recreator/SKILL.md)。

## Repository Layout

```text
yeedy-kits/
├── mcp/              # MCP servers and related source
├── plugin/           # Plugins and related source
├── skill/            # Each skill remains an independent source package
│   ├── ai-coding-daily-coach/
│   ├── omp-rolebench/
│   ├── optimize-agent-memory/
│   └── web-ui-recreator/
└── README.md
```

新增 skill 时使用小写连字符目录名，并让目录名与 `SKILL.md` frontmatter 中的 `name` 保持一致。只加入执行 skill 必需的文件，避免在各 skill 内重复维护 README、安装说明或变更日志。

## Development

修改 skill 后至少执行结构校验：

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" ./skill/<skill-name>
```

如果 skill 包含脚本，还应执行语法检查和覆盖主流程、错误边界的针对性测试。例如：

```bash
python3 -m py_compile ./skill/<skill-name>/scripts/*.py
```

校验器需要 `PyYAML`。当前 Python 环境缺少该依赖时，应使用已安装依赖的 Python 环境，或安装 `PyYAML` 后再运行。

## Install Locally

将单个 skill 同步到 Codex 的个人 skills 目录：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills/<skill-name>"
rsync -a --delete --exclude='.DS_Store' \
  ./skill/<skill-name>/ \
  "${CODEX_HOME:-$HOME/.codex}/skills/<skill-name>/"
```

`--delete` 会删除安装目录中仓库已不存在的旧文件。执行前确认目标只指向具体 skill 目录，不要指向整个 `skills` 目录。

OMP 使用同一份 skill 包，将上述目标目录改为 `~/.omp/agent/skills/<skill-name>/` 即可。安装到个人目录属于本机配置变更；本仓库保留可独立安装的源码。

安装或更新 skill 不应复制 workspace 中的运行状态。以 `ai-coding-daily-coach` 为例，答题记录和画像位于使用方 workspace 的 `.ai-coding-daily-coach/`，不在本仓库或个人 skill 安装目录中。

## Contribution Rules

- 保持 `SKILL.md` 简洁，把详细资料放到 `references/`
- 重复且要求确定性的操作放到 `scripts/`，并实际运行测试
- 在用户输入、外部数据、持久化边界和关键不变量处明确校验
- 错误应直接暴露，不静默降级或吞异常
- 不提交凭据、个人运行状态、缓存文件和生成产物
- 每次提交只包含相关变更，并在提交信息中覆盖实际行为变化
