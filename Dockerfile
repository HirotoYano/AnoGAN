FROM tensorflow/tensorflow:latest-gpu

ENV http_proxy "http://wwwproxy.osakac.ac.jp:8080"
ENV https_proxy "http://wwwproxy.osakac.ac.jp:8080"

RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm -f cuda-keyring_1.0-1_all.deb

RUN apt update \
    && apt upgrade -y \
    && apt install -y \
       git \
       tmux \
       libgl1-mesa-dev \
    # imageのサイズを小さくするためにキャッシュ削除
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    # pipのアップデート
    && pip install --upgrade pip

# 作業するディレクトリを変更
WORKDIR /home/DeepLearning

COPY requirements.txt ${PWD}

# pythonのパッケージをインストール
RUN pip install -r requirements.txt

