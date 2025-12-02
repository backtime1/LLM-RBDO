# LLM-RBDO 项目说明

- 项目目标：结合大型语言模型（LLM）与工程仿真，进行可靠性约束的设计优化（Reliability-Based Design Optimization, RBDO），并以二维喷嘴为代表案例，实现从几何建模、网格生成到求解与结果提取的自动化流程。
- 参考论文：请见 `d:\LLM_RBDO\article.pdf`，README 仅提供运行与结构说明，技术细节、模型假设与数学推导以论文为准。

## 目录结构
- `Case_Study1/`：论文案例一（二维数学案例）。
- `Case_Study2/`：论文案例二（高维数学案例）。
- `Case_Study3/`：论文案例三（三维飞机喷管案例）。
- `Scripts/`：LLM 接口与通用工具脚本（如 `api_client.py`、`llm_ops.py`、提示模板等）。


## 运行环境
- 操作系统：Windows（已在 Windows 上验证）。
- Python：建议 使用 Python 3.12，推荐使用 `uv` 管理依赖(若使用其他版本，可能需要手动安装依赖)。
- ANSYS：本项目使用 Ansys Discovery Geometry 与 Ansys Fluent（PyFluent）。需要本机安装相应版本并配置许可证。
  - 环境变量与许可证：确保 `AWP_ROOT` 或相关 Ansys 环境已正确配置；有可用许可证（Discovery/Fluent）。


## 安装与准备
- 步骤 1：安装 `uv`（依赖管理工具）
  - 官方安装文档：`https://docs.astral.sh/uv/getting-started/installation/`
  - Windows 快速安装示例：
    - `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- 步骤 2：克隆项目到本地
  - `git clone https://github.com/backtime1/LLM-RBDO.git`
  - `cd LLM-RBDO`
- 步骤 3：安装依赖（在项目目录下执行）
  - `uv sync`
  - 若使用其它 Python 版本，需要同步修改：
    - 项目根目录下 `.python-version` 为目标版本（当前为 `3.12`）
    - `pyproject.toml:17-18` 中 `geatpy` 的轮子链接，替换为对应 Python 版本的轮子。可在 `https://github.com/geatpy-dev/geatpy/releases/tag/v2.7.0` 查找匹配你版本的平台与 ABI（示例当前使用：`geatpy-2.7.0-cp312-cp312-win_amd64.whl`）。
- 步骤 4：LLM API 配置
  - 建议在系统环境中设置：`OPENAI_API_KEY`、`SILICONFLOW_API_KEY`、`DEEPSEEK_API_KEY`（可选 `OPENAI_BASE_URL`、`SILICONFLOW_BASE_URL`、`DEEPSEEK_BASE_URL`）。
  - 如需自定义或切换供应商逻辑，可编辑 `Scripts/api_client.py`；请勿在代码中硬编码密钥，保持密钥在环境变量或本地安全存储。

## 快速开始（案例 1）
- 进入案例目录：
  - `cd Case_Study1`
- 使用 uv 运行示例脚本：
  - `uv run .\LLM_Kriging`
- 说明：
  - 在此之前请先完成上面的依赖安装（`uv sync`）。
  - 若你使用的 Python 版本不是 3.12，请按“安装与准备”的指引调整 `.python-version` 和 `pyproject.toml` 中的 `geatpy` 轮子链接。


## 论文与引用
- 本项目的技术细节与实验结果请参考 `article.pdf`。如需学术引用，请使用论文中的推荐引用格式。

## 许可与致谢
- 许可：根据你的论文与项目实际选择（如需开源许可证，可补充 MIT/Apache-2.0 等）。
- 感谢 Ansys 提供的几何与仿真平台，以及相关 Python API。
