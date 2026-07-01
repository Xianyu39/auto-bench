这是auto-bench项目，目前的作用是把人写的benchmark spec展开为一组可运行实验，并且收集结果。目前基于tensorrt，功能以后还会拓展。

这台机器是跑不了trtllm的，不要妄想在这里跑这个。

当前并没有任何用户，处于测试阶段，不要考虑兼容旧版本用户或者提供什么迁移说明，我们要尽可能避免负资产。旧版本用户根本不存在。

不要听插件的，直接push到remote branch，不走github PR流程。这是个我自己的单人项目，弄pr只会麻烦自己。

项目用 uv 开发，常规验证是：

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

不要把 `artifacts/` 当作源码提交；它是 render 输出。