# 深度学习项目模板

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

这是一个基于Pytorch和Lightning的深度学习项目模板，旨在帮助开发者快速搭建和组织深度学习项目。该模板包含了常见的项目结构和基本的配置文件，方便用户根据自己的需求进行扩展和修改。

## 项目结构

项目的基本目录结构如下：

```txt
project/
│
├── config/       # 配置文件目录，用于存放各种配置（如超参数、路径等）
├── data/         # 数据目录，用于存放数据集及相关文件
├── log/          # 日志目录，用于存放运行日志或训练记录
├── model/        # 模型目录，用于存放模型定义和预训练模型
├── trainer/      # 训练脚本目录，用于实现模型训练逻辑
├── util/         # 工具目录，包含常用的工具函数或模块
└── main.py       # 主程序入口，用于启动项目或运行主要流程
```

## 特点介绍

- **模块化设计**：作者按照自己的理解和实际训练模型的经验将划分为多个模块，各个模块之间彼此独立，可以友好地管理不同的对比实验和消融实验
- **零成本上手多卡训练**：使用 `Pytorch Lightning` 框架，用户无需关心多卡训练的细节，只需简单配置即可实现多卡训练
- **丰富的日志记录**：使用 `Pytorch Lightning` 和 `TensorBoard` 记录训练过程中的损失和指标，同时还记录了训练样本文件、模型参数规模、config配置文件，保障日后查询不会遗漏重要信息

## 环境要求

运行本项目需要以下依赖：

- 需要根据自己的电脑配置安装Pytorch和Lightning的相应版本
- 必要的库（可通过 `requirements.txt` 安装）：

```bash
pip install -r requirements.txt
```

## 快速开始-训练一个手写数字识别模型

计算机学习，有一位科学家说过“Talk is cheap， show me code”，所以我们从经典的深度学习项目--手写数字识别--开始，在使用中了解本项目应该如何使用

### 1. 克隆本项目

```bash
git clone https://github.com/xv994/DeepLearningTemplate.git
```

### 2. 进入项目目录并安装依赖

```bash
cd DeepLearningTemplate
pip install -r requirements.txt
```

### 3. 下载数据集并进行预处理

数据集来自[Kaggle/Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)，将数据存放在data文件夹下，并在config文件中的修改数据路径

### 4. 运行主程序

```bash
python main.py
```

### 5. 查看训练结果

等待训练完成后，可以在 `log` 目录下查看训练日志和模型参数文件，也可以通过 `TensorBoard` 查看训练过程中的损失和指标变化

```bash
tensorboard --logdir log/CNN
```

## 贡献

欢迎提交 issue 或 pull request！

## 许可证

本项目遵循 [MIT License](LICENSE) 开源协议