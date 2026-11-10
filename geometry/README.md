# 分层形状与CT教师图拟合

本模块集中第一阶段的分层高斯训练和可复用几何工具。共享的GaussianModel、相机、渲染器和优化参数仍在根目录scene、gaussian_renderer、arguments和utils中。

在训练仓库目录执行：

```powershell
python run.py geometry train --help
python run.py geometry train -s /path/to/teachers/L7 -m /path/to/new_model --load_layer /path/to/base/layer_save_30000.pth --save_layer --eval -a
```

也可用 `python -m geometry.train ...`。新的唯一根入口为run.py，不再保留旧的根目录train.py。训练算法文件由原train.py原样移动，未改变损失或优化行为。

| 命令 | 模块 | 用途 |
| --- | --- | --- |
| `geometry train` | `geometry/train.py` | 原分层教师图拟合，加载/保存层、增密及剪枝 |
| `geometry refine-surface` | `geometry/refine_surface.py` | 可选的表面约束微调，会修改几何 |
| `geometry refine-scales` | `geometry/refine_scales.py` | 可选的有界尺度后处理 |
| `geometry export-base` | `geometry/export_base.py` | 导出冻结底模RGB/alpha/depth用于教师图生成 |
| `geometry prepare-colmap` | `geometry/prepare_colmap.py` | 上游COLMAP数据准备 |

`launchers/train_L4_v4.ps1`保留可配置Python/Workspace/Output参数，使用新模块入口，拒绝覆盖已有输出。过时的L1启动脚本已从活动代码中删除。

这里的“形状训练”仍是原RGB L1+SSIM教师图拟合，同时可能优化SH、不透明度与几何，并进行结构变化；不是已实现CT几何损失，也不是严格只更新形状。它与appearance冻结几何的LUT训练是两个独立阶段。不会因导入模块或查看--help启动训练。

旧版本诊断与批量对比驱动不在此目录；本地整理前副本位于工作区archive，Git已跟踪的旧版本也可通过历史恢复。
