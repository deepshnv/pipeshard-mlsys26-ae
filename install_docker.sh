#!/bin/bash
set -e

echo "=== Removing any old Docker packages ==="
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
    sudo apt-get remove -y $pkg 2>/dev/null || true
done

echo "=== Installing prerequisites ==="
sudo apt-get update
sudo apt-get install -y ca-certificates curl

echo "=== Adding Docker GPG key ==="
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "=== Adding Docker repo ==="
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "=== Installing Docker Engine ==="
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=== Adding user to docker group ==="
sudo usermod -aG docker $USER

echo "=== Starting Docker daemon ==="
sudo service docker start

echo "=== Verifying ==="
sudo docker run --rm hello-world

echo ""
echo "=== Docker installed successfully ==="
echo "NOTE: Log out and back into WSL for group changes to take effect (so you can run docker without sudo)."
