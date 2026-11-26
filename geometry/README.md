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
| `geometry refine-local` | `geometry/refine_local.py` | L4 局部表面细化试验：选择大高斯、拆分并贴合 CT 表面，附同轮数颜色对照 |
| `geometry export-base` | `geometry/export_base.py` | 导出冻结底模RGB/alpha/depth用于教师图生成 |
| `geometry prepare-colmap` | `geometry/prepare_colmap.py` | 上游COLMAP数据准备 |

`launchers/train_L4_v4.ps1`保留可配置Python/Workspace/Output参数，使用新模块入口，拒绝覆盖已有输出。过时的L1启动脚本已从活动代码中删除。

这里的“形状训练”仍是原RGB L1+SSIM教师图拟合，同时可能优化SH、不透明度与几何，并进行结构变化；不是已实现CT几何损失，也不是严格只更新形状。它与appearance冻结几何的LUT训练是两个独立阶段。不会因导入模块或查看--help启动训练。

旧版本诊断与批量对比驱动不在此目录；本地整理前副本位于工作区archive，Git已跟踪的旧版本也可通过历史恢复。

### L4 局部细化试验

发布前梯度审计：历史 L4new 及局部试验使用的独立 accumulation 输出不支持 alpha 反传，因此旧运行的 alpha 项没有直接参与优化；轮廓 IoU 是测量结果，不能归功于该项。当前 `refine_local.py` 已通过白色高斯/黑背景的 RGB 渲染取得等价可微 alpha，并测试不透明度、尺度梯度非零。已保存模型未因此重新训练。原分层训练基线保持原样。

`python run.py geometry refine-local --help` 查看参数。输入完整分层 PLY、原 cameras.json、独立 L4 RGBA 教师图及 train.txt/test.txt、带 vertices/normals 的 CT 表面 NPZ。默认仅运行 600 步试验，不自动接续长训练。

先按训练视角的实际成像贡献选择最大轴 sigma >5mm、距表面顶点 <5mm 的最多 512 个高斯。拆分采用确定性的二分，保持混合中心和协方差；alpha 合成不保证初始图像等价，必须检查教师图误差。最长轴 sigma 默认不超过 2.5mm，中心距初始锚点不超过 0.6mm。不修改未选择的 L4 高斯与其他层，不做全局剪枝、不透明度重置。

默认 `--initialization preserve` 保留原高斯覆盖与旋转，不将中心直接吸附到 CT；保留原最短轴厚度上限，CT 法向移动不超过 0.1mm，并加强前景及边缘的 alpha 约束。`--initialization surface` 是对照选项：直接投影至 CT 顶点、按法线定向、法向 sigma 上限 0.3mm。首轮 surface 试验出现轮廓退化，不应直接替换正式模型。

损失包括 RGB L1、SSIM、前景内部梯度、alpha 和锚点约束。颜色对照与细化试验使用相同训练相机序列与步数，15 个测试视角不参与候选选择或梯度更新。输出仅 `L4_refined.ply`、`comparison.jpg`、`result.json`。导出后重新加载并验证渲染，未修改行逐字段核验。

此输出是合成教师图的几何试验，不能作为真实纹理学习成功的证据。新增点的 HU 暂继承父点，未重新采样；不能直接把该模型当作已验证的 HU 映射版本。表面锚点使用最近顶点近似，需检查轮廓、孔洞、测试视角误差及真实颜色迁移后效果，再决定是否扩大训练。

较长的局部重训可显式指定 `--steps 6000 --skip-control --lr-final-factor 0.1 --model-name L4new.ply`。默认仍是 600 步双组小实验。`--checkpoint` 可将包含优化器和局部参数的恢复状态写到独立归档路径，不放进最终结果目录。当前入口没有自动恢复参数；不得把检查点的存在描述为已支持断点续训。

重训后使用工作区 `tools/training/finalize_refined_model.py --run <训练归档目录> --ct <源CT> --output <新的最终目录> --model-name L4new.ply`，在含 SimpleITK 的环境中为移动/新增的子点重新采样 raw、macro、detail HU。该工具先核对旧点 HU 与 CT 物理坐标采样，保留所有未移动点和非 HU 字段，输出仍只有 PLY、comparison.jpg、result.json。它不修改训练源模型。

2026-11-25 小规模验证：输入 `output/geometry/L7new/L7new.ply`，105 个训练视角、15 个测试视角，两组各 600 步。512 个父点替换为 12,192 个子点，L4 从 58,289 增至 69,969 点。直接贴合表面的试验在 `archive/2026-11-25/L4_local_refinement`，轮廓 IoU 从 0.997805 降至 0.989674，不用于替换正式模型。保守拆分试验在 `archive/2026-11-25/L4_preserve_refinement`：IoU 0.997792；测试前景 L1 0.014435，仍高于原模型 0.013926 与颜色对照 0.013911。原始相机 view_000117 的大高斯主导面积占比从 24.69% 降至 16.17%，但未证实自然纹理改善。原模型、其他层及未选中的 L4 行均未改动；导出重载渲染一致，两项拆分数学检查通过。两个目录各仅保留 PLY、对比图、result.json。
