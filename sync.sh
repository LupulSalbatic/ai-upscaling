#!/bin/bash
# Sync cu GitHub

ACTION=$1

if [ "$ACTION" == "push" ]; then
    echo "=== Push pe GitHub ==="
    git add -A
    git commit -m "Update: $(date '+%Y-%m-%d %H:%M')"
    git push origin main
    echo "=== Gata! ==="

elif [ "$ACTION" == "pull" ]; then
    echo "=== Pull de pe GitHub ==="
    git pull origin main
    echo "=== Gata! ==="

else
    echo "Folosire: ./sync.sh push | pull"
fi
