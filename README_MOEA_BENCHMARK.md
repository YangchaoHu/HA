# 多级多目标进化算法测试框架

本框架用于评估多目标进化算法在不同问题层级上的真实性能，覆盖从标准数学测试集、官方工程基准，到真实 ANSYS 有限元仿真的完整实验链路。

框架包含三个测试层级：

1. 数学测试集：使用官方 `pymoo` 问题库中的 ZDT 和 DTLZ 系列。
2. 工程基准集：只使用有明确工程设计背景的官方 `pymoo` 问题，包括 `welded_beam`、`truss2d` 和 `carside`。`BNH`、`TNK`、`OSY` 等显式表达式问题归入约束解析基准，不再作为工程基准。
3. 真实 CAE 仿真：基于 PyMAPDL 官方公开结构分析示例和 MAPDL 标准单元，构建 5 个真实可运行的有限元多目标优化案例，并通过本机 ANSYS MAPDL 真实求解。

框架会保存算法运行的全部中间结果，包括每代种群、最终 Pareto 前沿、逐代性能指标、ANSYS 环境报告，以及论文可用的汇总表格和收敛曲线。

## 运行环境

请使用指定的 `hycpytorch` conda 环境：

```powershell
conda activate hycpytorch
python scripts/list_benchmarks.py --include-expensive
python scripts/check_ansys_env.py --output experiments_results/ansys_env_report.json
```

如果当前终端不方便激活 conda 环境，也可以直接调用该环境的 Python 解释器：

```powershell
& 'C:\Environments\ananconda\envs\hycpytorch\python.exe' scripts/list_benchmarks.py --include-expensive
```

本机已验证环境：

- Python：`C:\Environments\ananconda\envs\hycpytorch\python.exe`
- `pymoo`：0.6.1.5
- `ansys.mapdl.core`：0.72.1
- ANSYS 安装根目录：`D:\SoftWares\ANSYS Inc\v252`
- 已发现 MAPDL 可执行文件：`D:\SoftWares\ANSYS Inc\v252\ansys\bin\winx64\MAPDL.exe`
- 已发现许可证变量：`ANSYSLMD_LICENSE_FILE=1055@localhost`

## 快速冒烟测试

运行数学层轻量测试：

```powershell
conda activate hycpytorch
python scripts/run_moea_experiments.py --config configs/smoke_math.json
python scripts/evaluate_moea_results.py `
  --input-root experiments_results/moea_benchmark_smoke `
  --output-dir experiments_results/moea_benchmark_smoke_paper `
  --metric igd
```

冒烟测试会生成以下结果：

- `final_population.csv`：最终种群。
- `final_front.csv`：最终非支配前沿。
- `final_metrics.json`：最终指标。
- `metrics_by_generation.csv`：逐代指标。
- `populations/gen_*.csv`：每代种群快照。
- `summary_by_problem_algorithm.csv`：按问题和算法汇总的统计结果。
- `paper_table.tex`：可直接放入论文的 LaTeX 表格。
- `plots/*_convergence.png`：收敛曲线图。

## 工程基准测试

运行工程层 benchmark：

```powershell
python scripts/run_moea_experiments.py --config configs/engineering_smoke.json
python scripts/evaluate_moea_results.py `
  --input-root experiments_results/moea_benchmark_engineering `
  --output-dir experiments_results/moea_benchmark_engineering_paper `
  --metric igd
```

工程层问题来自官方 `pymoo` 基准问题，适合用于验证算法在约束工程设计问题上的性能。

## 真实 ANSYS/MAPDL 仿真

先检查环境，但不启动 MAPDL：

```powershell
python scripts/check_ansys_env.py --output experiments_results/ansys_env_report.json
```

然后可以选择运行真实 MAPDL 启动测试：

```powershell
python scripts/check_ansys_env.py --launch-smoke --output experiments_results/ansys_env_launch_report.json
```

运行一次真实有限元单点评估：

```powershell
python scripts/run_mapdl_cantilever_smoke.py --case mapdl_cantilever_beam --width 0.05 --height 0.08
```

当前真实 FEA 层包含 5 个已通过 MAPDL 单点评估的案例：

| 案例名 | 单元/模型 | 设计变量 | 目标 |
|---|---|---|---|
| `mapdl_cantilever_beam` | BEAM188 悬臂梁 | 截面宽度、截面高度 | 最小体积、最小端部位移 |
| `mapdl_simply_supported_beam` | BEAM188 简支梁 | 截面宽度、截面高度 | 最小体积、最小跨中位移 |
| `mapdl_portal_frame` | BEAM188 门式框架 | 构件截面宽度、截面深度 | 最小体积、最小平面位移 |
| `mapdl_two_bar_truss` | LINK180 双杆桁架 | 左杆面积、右杆面积、桁架高度 | 最小体积、最小节点位移 |
| `mapdl_plane_stress_plate` | PLANE182 平面应力板 | 板高、厚度 | 最小体积、最小拉伸位移 |

逐个运行真实 FEA smoke test：

```powershell
python scripts/run_mapdl_cantilever_smoke.py --case mapdl_cantilever_beam --work-root experiments_results/mapdl_five_case_smoke
python scripts/run_mapdl_cantilever_smoke.py --case mapdl_simply_supported_beam --work-root experiments_results/mapdl_five_case_smoke
python scripts/run_mapdl_cantilever_smoke.py --case mapdl_portal_frame --work-root experiments_results/mapdl_five_case_smoke
python scripts/run_mapdl_cantilever_smoke.py --case mapdl_two_bar_truss --work-root experiments_results/mapdl_five_case_smoke
python scripts/run_mapdl_cantilever_smoke.py --case mapdl_plane_stress_plate --work-root experiments_results/mapdl_five_case_smoke
```

将 5 个真实 CAE 案例放入算法测试矩阵：

```powershell
python scripts/run_moea_experiments.py --config configs/cae_five_case_smoke.json
```

注意：真实 MAPDL 求解会占用许可证并消耗明显更多时间，建议先用小种群、小代数做 smoke test，再扩大实验规模。

## 三层完整实验模板

完整论文级实验模板位于：

```text
configs/full_three_level_template.json
```

该模板包含：

- 数学测试集：ZDT1、ZDT2、ZDT3、ZDT4、ZDT6、DTLZ1、DTLZ2、DTLZ3、DTLZ4。
- 约束解析基准：BNH、TNK、OSY。
- 工程基准集：welded beam、two-bar truss、car side-impact。
- 真实有限元案例：MAPDL cantilever beam、simply supported beam、portal frame、two-bar truss、plane-stress plate。
- 算法：NSGA-II、NSGA-III、U-NSGA-III、MOEA/D、SMS-EMOA，以及本项目已有的 `HA_NSGA3`。

完整模板中的 MAPDL 案例属于昂贵仿真任务。正式运行前，请先确认 `scripts/check_ansys_env.py --launch-smoke` 和 `scripts/run_mapdl_cantilever_smoke.py --case ...` 均能成功。

## 代理模型支持

对于昂贵仿真问题，框架提供了评估数据归档和基础 RBF/GP 代理模型拟合工具：

```powershell
python scripts/build_surrogate_from_archive.py `
  --archive experiments_results/mapdl_five_case_smoke/mapdl_cantilever_beam_evaluations.csv `
  --kind rbf `
  --output-dir experiments_results/surrogates
```

代理模型模块保持轻量，便于后续继续扩展主动学习、期望超体积改进、代理辅助 HA，或者接入已有的代理模型对比实验。

## 主要文件

- `moea_benchmark/benchmarks/catalog.py`：三层 benchmark 注册表。
- `moea_benchmark/algorithms/factory.py`：算法注册与创建入口。
- `moea_benchmark/runner.py`：实验矩阵运行器。
- `moea_benchmark/recording.py`：中间种群和逐代指标记录器。
- `moea_benchmark/metrics/indicators.py`：HV、IGD、IGD+ 和非支配前沿计算。
- `moea_benchmark/cae/ansys_env.py`：本机 ANSYS/PyMAPDL 环境验证。
- `moea_benchmark/cae/mapdl_structural_cases.py`：5 个真实 MAPDL 有限元多目标案例。
- `moea_benchmark/cae/public_cantilever_beam.py`：悬臂梁案例的兼容导入入口。
- `scripts/evaluate_moea_results.py`：生成论文可用 CSV、LaTeX 表格和收敛图。

## 已完成的本机验证

当前工作区已经完成以下验证：

- `hycpytorch` 环境中的 `pymoo`、`ansys.mapdl.core`、`numpy`、`pandas`、`scipy`、`pyvista` 可用。
- ANSYS 2025 R2 MAPDL 可执行文件已发现。
- MAPDL 启动测试成功，版本为 `25.2`。
- 5 个真实 FEA 案例均已完成 MAPDL 单点评估：
  - `mapdl_cantilever_beam`：`[0.004, 0.00190777]`。
  - `mapdl_simply_supported_beam`：`[0.01152, 5.0842e-05]`。
  - `mapdl_portal_frame`：`[0.0282975, 1.4577e-05]`。
  - `mapdl_two_bar_truss`：`[0.0047037, 2.4392e-06]`。
  - `mapdl_plane_stress_plate`：`[0.0065625, 3.9107e-06]`。
- 5 个真实 FEA 案例已通过 `run_moea_experiments.py` 极小优化矩阵验证。
- 数学层 smoke test 已通过。
- 工程层 smoke test 已通过。
- 结果汇总脚本已成功生成 CSV、LaTeX 表格和收敛曲线。

## 公开来源

- `pymoo` ZDT 测试集：`https://pymoo.org/problems/multi/zdt.html`
- `pymoo` DTLZ 测试集：`https://pymoo.org/problems/many/dtlz.html`
- `pymoo` welded beam：`https://pymoo.org/problems/multi/welded_beam.html`
- `pymoo` truss2d：`https://pymoo.org/problems/multi/truss2d.html`
- PyMAPDL BEAM188 梁示例：`https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/modal_beam.html`
- PyMAPDL 结构分析示例总览：`https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/index.html`
- PyMAPDL 二维板/平面应力相关示例：`https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/2d_plate_with_a_hole.html`

