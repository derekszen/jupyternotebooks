#!/usr/bin/env bash
# Sets up Docker (and NVIDIA tooling) on Ubuntu, then writes a GPU-ready Dockerfile
# for running the current Qwen Image Edit benchmark stack.
set -euo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  SUDO=sudo
else
  SUDO=""
fi

log() {
  printf '[setup-docker] %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Missing required command: $1"
    exit 1
  fi
}

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    log "Docker already installed. Skipping installation."
    return
  fi

  log "Installing Docker Engine..."
  $SUDO apt-get update
  $SUDO apt-get install -y ca-certificates curl gnupg lsb-release
  $SUDO install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  $SUDO apt-get update
  $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  $SUDO systemctl enable --now docker
  log "Docker installed. Log out/in for group changes if needed."
}

install_nvidia_toolkit() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not detected; ensure NVIDIA drivers are installed before continuing."
    return
  fi

  if [ -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
    log "NVIDIA Container Toolkit repo already configured."
  else
    log "Configuring NVIDIA Container Toolkit repository..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  fi

  log "Installing NVIDIA Container Toolkit..."
  $SUDO apt-get update
  $SUDO apt-get install -y nvidia-container-toolkit
  $SUDO nvidia-ctk runtime configure --runtime=docker
  $SUDO systemctl restart docker
  log "NVIDIA Container Toolkit installed."
}

write_dockerfile() {
  local dockerfile="Dockerfile.qwen"
  log "Writing ${dockerfile} for the Qwen benchmark stack..."
  cat <<'DOCKERFILE' > "${dockerfile}"
# CUDA + PyTorch base with Python 3.11
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/opt/uv

# Install system deps and uv package manager
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends curl git build-essential python3-venv pkg-config libgl1 \ 
    && curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir "$UV_INSTALL_DIR" \ 
    && ln -s "$UV_INSTALL_DIR/uv" /usr/local/bin/uv \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency manifests first for efficient caching
COPY pyproject.toml .
COPY uv.lock* ./

# Install project dependencies (falls back gracefully if uv.lock absent)
RUN if [ -f uv.lock ]; then uv sync --frozen --no-group dev; else uv sync --no-group dev; fi

# Copy the rest of the repository
COPY . .

# Default command runs benchmark script; override as needed
CMD ["uv", "run", "python", "qwen_image_edit_benchmark.py", "--runs", "5", "--out", "results/benchmark.parquet"]
DOCKERFILE

  log "Dockerfile.qwen created. Build with: docker build -f Dockerfile.qwen -t qwen-image-edit ."
}

add_user_to_docker_group() {
  if groups "$USER" | grep -q '\bdocker\b'; then
    return
  fi

  log "Adding $USER to docker group (requires re-login to take effect)."
  $SUDO usermod -aG docker "$USER"
}

main() {
  require_command curl
  install_docker
  add_user_to_docker_group
  install_nvidia_toolkit
  write_dockerfile
  log "Setup complete. Re-login if group membership changed, then run 'docker run --gpus all ...'"
}

main "$@"
