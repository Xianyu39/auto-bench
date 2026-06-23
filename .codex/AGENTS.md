这是auto-bench项目，目前的作用是把人写的benchmark spec展开为一组可运行实验，并且收集结果。目前基于tensorrt，功能以后还会拓展。

这台机器是跑不了trtllm的，不要妄想在这里跑这个。

当前并没有任何用户，处于测试阶段，不要考虑兼容旧版本用户或者提供什么迁移说明，我们要尽可能避免负资产。旧版本用户根本不存在。

直接推送，不走github那套麻烦的PR流程。

项目用 uv 开发，常规验证是：

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

不要把 `artifacts/` 当作源码提交；它是 render 输出。

YAML 协议设计要点：

- 顶层只允许 `metadata`、`vars`、`trtllm`。
- `trtllm` 根下是 `trtllm-bench` 自身选项；`throughput`、`latency`、`build` 这类键是子命令，子命令下面才是子选项。
- 不要把 `command` 作为普通字段，也不要依赖 YAML 顺序来区分全局选项和子命令选项。
- `vars` 是实验变量区，不渲染到 CLI；用来 sweep 或参与表达式计算。
- 内置 `runtime.*` 是只读表达式命名空间，比如 `runtime.run_dir`、`runtime.log_path`、`runtime.config_path`、`runtime.dataset_dir`。这些值应保持基于 `$SCRIPT_DIR`，让渲染出来的脚本可移动。
- 不要用 manifest 当参数白名单。TensorRT-LLM 参数不全，未知 `trtllm` 参数必须保留并渲染成 CLI 选项。
- manifest 只适合放已知参数的渲染提示，比如 CLI 拼写、布尔参数是否带值、少量默认值。
- `dataset`、`config` 这类特殊 option 应视为返回 path 的 operation：resolved 的 `trtllm` 里最终保留 path，副作用计划放在 `commands.prepare_dataset` 或 `commands.write_config`。
- managed `config` resolved 后应该是 `config: config.yaml`，不要再把 `{path, content}` 塞回 `trtllm` 参数树里。
- managed dataset/config 的协议对象可以严格校验，因为那是 autobench 自己的输入协议；普通 TRT-LLM benchmark 参数不要严格校验。
