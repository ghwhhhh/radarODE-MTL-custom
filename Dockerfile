FROM pytorch/pytorch:2.1.1-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# 从 GitHub 克隆项目代码
RUN git clone -b main https://github.com/ghwhhhh/radarODE-MTL-custom.git

WORKDIR /workspace/radarODE-MTL-custom

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 数据目录挂载点
VOLUME ["/data"]

# 训练启动脚本
CMD ["python", "Projects/radarODE_plus/main.py", \
    "--dataset_path", "/data/Dataset_mix", \
    "--save_path", "/workspace/radarODE-MTL-custom/Model_saved", \
    "--train_bs", "4", \
    "--test_bs", "4", \
    "--epochs", "200", \
    "--num_workers", "0"]
