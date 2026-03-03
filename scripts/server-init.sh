#!/usr/bin/env bash
# One-time Hetzner CAX ARM server bootstrap.
# Run as root on a fresh Ubuntu 22.04+ ARM instance.

set -euo pipefail

echo "=== Motherlabs Server Init ==="

# System updates
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose plugin
apt-get install -y docker-compose-plugin

# Create deploy user
if ! id -u deploy &>/dev/null; then
    useradd -m -s /bin/bash -G docker deploy
    echo "Created user: deploy"
fi

# Create app directory
mkdir -p /opt/motherlabs
chown deploy:deploy /opt/motherlabs

# SSH hardening
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
systemctl restart sshd

# Firewall
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Login as deploy to finish setup
echo ""
echo "=== Done ==="
echo "1. Copy your SSH public key:  ssh-copy-id deploy@<this-server>"
echo "2. Clone the repo:            su - deploy -c 'cd /opt/motherlabs && git clone https://github.com/dopexthrone/MeaningWorks .'"
echo "3. Copy .env:                 cp .env.example .env && nano .env"
echo "4. Start:                     docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
