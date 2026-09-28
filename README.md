# NDFitter

用于多维数据拟合的本地开发项目，包含高斯过程、神经网络和规则网格插值代码。

本项目从旧本地工作目录的最新 Python 源码迁入，包含当时尚未提交的修改；
Git 历史从清理后的源码快照开始。没有迁入实验数据、训练模型、图片、日志或实验笔记本。
原目录继续作为本地备份。

## 核心模块

| 模块 | 用途 |
| --- | --- |
| `NDFitter/GP` | 基于 GPy 的高斯过程拟合 |
| `NDFitter/GPyTorch` | 一维、二维、输出缩放和分块核等训练实现 |
| `NDFitter/MLP` | 数据加载、网络结构、训练和评估函数 |
| `NDFitter/torch_interpolation` | PyTorch 规则网格插值 |
| `NDFitter/utils.py` | 数据读写及通用辅助函数 |
| `NDFitter/paths.py` | 与电脑用户名无关的路径解析 |

## 开发与运行

在项目根目录创建并激活自己的 Python 环境，按需要安装后端：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[gpytorch,mlp]'
# 使用原 GPy 后端时另行安装：python -m pip install -e '.[gp]'
```

核心算法从旧代码保留，依赖版本尚未锁定；迁移检查不等同于完整的训练复现。
新的包入口按需加载后端，安装一个后端不要求其他后端全部可用。

将自己的输入放在本地 `data/` 目录中，例如 `data/example/points.pickle` 和
`data/example/values.pickle`。也可运行时指定项目外部的数据目录，无须复制数据。
训练和数据加载入口的相对路径统一以项目根目录为基准，显式传入的绝对路径也可使用，
但不要将本机路径写回代码或提交信息。开发时使用上面的 editable 安装。

```sh
bash scripts/run_gp.sh --data_folder data/example --epochs 20 --grid
bash scripts/run_mlp.sh --folder data/example --output outputs/mlp --epochs 10
python -m NDFitter.GPyTorch.train_1d --data_folder data/example --epochs 20
python -m NDFitter.GP.train --data_folder data/example
```

各后端保持原有输入格式要求，示例命令需先准备兼容数据。GP 模型与网格输出仍写入
输入目录下的时间子目录，MLP 默认也将结果写在输入目录内；可用 `--output` 指定
MLP 输出目录。TensorBoard 写入本地 `outputs/tensorboard/`。
Pickle 文件只应从可信的本地来源加载。

旧 MLP 评估脚本的命令行参数接口仍存在原有不一致；本轮保留了代码，尚未修复该接口。
实验笔记本中的研究分析流程和集群任务参数留在旧备份，不作为通用代码迁入。

## 本地数据与隐私

- Git 只允许源代码、必要文档、测试和开发配置。
- 数据、模型、图片、日志、笔记本、环境及编辑器配置均被忽略。
- `NDFitter/MLP/data/` 是 **Python 数据加载代码**，因此保留；该目录中的数据文件仍被拒绝。
- 提交检查读取实际暂存内容，强制加入的数据文件也会被拒绝。
- 推送检查审核待推送分支的全部历史，防止合并旧历史后把数据重新带入。
- Git 作者和提交者使用通用身份；原 MIT 许可证版权声明原样保留，这是明确的署名例外。

本机已启用检查。以后克隆或复制为新的 Git 仓库时，需要重新启用本地配置：

```sh
git config --local core.hooksPath .githooks
git config --local user.name 'NDFitter contributors'
git config --local user.email 'contributors@example.invalid'
python3 scripts/check_repository.py
python3 scripts/check_repository.py --history
python3 -m unittest discover -s tests -v
```

这些检查是防误操作措施，不能替代发布前的人工审核，也不能阻止主动绕过 hook。
当前策略不接受未经审核的注释标签。旧项目远端及其历史不受影响；本仓库不包含旧项目历史。

迁入文件的原始哈希与最终哈希见 `docs/migration.json`，说明见 `docs/MIGRATION.md`。
