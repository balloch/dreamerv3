# docker build -f Dockerfile -t img . && \
# docker run --gpus all -it --rm -v ~/logdir/docker:/logdir img /venv/bin/python blimp_example.py
#   /venv/bin/python dreamerv3/main.py --logdir /logdir/{timestamp} --configs minecraft debug --task minecraft_diamond   

FROM ghcr.io/nvidia/driver:7c5f8932-550.144.03-ubuntu24.04

# System
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common grep \
  libglew-dev x11-xserver-utils xvfb wget \
  && apt-get clean

# Python (DMLab needs <=3.11)
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10-dev python3.10-venv unzip && apt-get clean
RUN python3.10 -m venv /venv 
#--upgrade-deps
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools

# Envs
# RUN wget -O - https://gist.githubusercontent.com/danijar/ca6ab917188d2e081a8253b3ca5c36d3/raw/install-dmlab.sh | sh
# RUN pip install ale_py==0.9.0 autorom[accept-rom-license]==0.6.1
# RUN pip install procgen_mirror
# RUN pip install crafter
# RUN pip install dm_control
# RUN pip install memory_maze
COPY safety-gymnasium /safety-gymnasium
RUN pip install -e /safety-gymnasium
# RUN pip install -r /blimp-safety-env-main/requirements.txt
# ENV MUJOCO_GL=egl
# RUN apt-get update && apt-get install -y openjdk-8-jdk && apt-get clean
# RUN pip install https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
# RUN chown -R 1000:root /venv/lib/python3.11/site-packages/minerl

# Requirements
# RUN pip install jax==0.3.25
# RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install jax[cuda]
# RUN pip install jax[cuda]==0.5.0
# RUN pip install jax[cuda]==0.5.0
# RUN pip install jax[cuda12_pip]
RUN pip install jax[cuda12] optax
# ==0.4.30
# RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install jax-jumpy
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# #STUPID
RUN wget https://github.com/anderskm/gputil/archive/refs/heads/master.zip
RUN unzip master.zip
COPY setup.cfg gputil-master/
RUN pip install -e gputil-master/
# Ollama installation
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Ollama Python library
RUN pip install ollama

# Add Ollama to PATH
# ENV PATH="/usr/local/bin:$PATH"

# Verify Ollama installation
# RUN ollama --version
# EXPOSE 11434

# Source
RUN mkdir /app
WORKDIR /app
COPY . .
RUN chown -R 1000:root .

ENTRYPOINT ["sh", "entrypoint.sh"]
# docker build -f Dockerfile -t img . && docker run --gpus all -it --rm -v ~/logdir/docker:/logdir img /venv/bin/python blimp_example_eval.py
# docker build -f Dockerfile -t img . && docker run --gpus all -it --rm -v ~/logdir/docker:/logdir img /venv/bin/python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0
# docker build -f Dockerfile -t img . && docker run --gpus all -it --rm -v ~/logdir/docker:/logdir --entrypoint /venv/bin/python img SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0