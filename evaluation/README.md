# 复用评估工具

从训练仓库目录执行：

```powershell
python run.py evaluation render --help
python run.py evaluation metrics --help
python run.py evaluation verify-layers --help
```

- `render.py`：原渲染入口，生成训练/测试视角图像。
- `metrics.py`：原图像拟合指标入口，读取已生成的渲染与参考图。
- `verify_layers.py`：检查L7模型的冻结底层字段、有限值及HU附加是否保持原字段。

文件仅移动到任务模块内，算法不变；也可用 `python -m evaluation.render` 等方式执行。图像拟合指标不等于解剖准确性或跨患者真实感验证。旧的硬编码批量benchmark/透明度/剪枝对比驱动已从活动目录移除。
