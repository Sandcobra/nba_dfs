#!/usr/bin/env bash
# NBA DFS Auto-Deploy Watcher
# Polls GitHub for changes, auto-pulls and restarts the app
# Run: nohup bash scripts/nba_dfs_auto_deploy.sh > logs/auto_deploy.log 2>&1 &

set -euo pipefail

REPO_DIR="/data/.openclaw/workspace/nba_dfs"
LOG_DIR="/data/.openclaw/workspace/logs"
LOG_FILE="$LOG_DIR/auto_deploy.log"
LAST_COMMIT_FILE="$LOG_DIR/.last_commit"
CHECK_INTERVAL=60  # Check every 60 seconds
PORT=5050

# Create logs dir
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"
}

get_local_commit() {
  cd "$REPO_DIR"
  git rev-parse HEAD 2>/dev/null || echo "unknown"
}

get_remote_commit() {
  cd "$REPO_DIR"
  git fetch origin main -q 2>/dev/null || true
  git rev-parse origin/main 2>/dev/null || echo "unknown"
}

restart_app() {
  log "🔄 Restarting NBA DFS app on port $PORT..."
  
  # Kill existing process
  pkill -f "python3 run_ui.py" 2>/dev/null || true
  sleep 2
  
  # Start new process
  cd "$REPO_DIR"
  nohup python3 run_ui.py --host 0.0.0.0 --port $PORT > "$LOG_DIR/nba_dfs_ui.log" 2>&1 &
  
  sleep 3
  
  if curl -s http://localhost:$PORT/ > /dev/null 2>&1; then
    log "✅ App is running on http://localhost:$PORT"
    return 0
  else
    log "❌ App failed to start. Check $LOG_DIR/nba_dfs_ui.log"
    return 1
  fi
}

log "🚀 NBA DFS Auto-Deploy starting..."
log "Repo: $REPO_DIR"
log "Check interval: ${CHECK_INTERVAL}s"
log "App port: $PORT"

# Initialize last commit
LOCAL=$(get_local_commit)
echo "$LOCAL" > "$LAST_COMMIT_FILE"
log "Initial commit: $LOCAL"

# Main loop
while true; do
  sleep "$CHECK_INTERVAL"
  
  LOCAL=$(cat "$LAST_COMMIT_FILE")
  REMOTE=$(get_remote_commit)
  
  if [ "$LOCAL" != "$REMOTE" ] && [ "$REMOTE" != "unknown" ]; then
    log "📦 New commit detected!"
    log "Local:  $LOCAL"
    log "Remote: $REMOTE"
    
    # Pull changes
    cd "$REPO_DIR"
    if git pull origin main -q; then
      log "✅ Pull successful"
      echo "$REMOTE" > "$LAST_COMMIT_FILE"
      
      # Restart app
      if restart_app; then
        log "✅ Deployment complete"
      else
        log "⚠️ Deployment had issues, check logs"
      fi
    else
      log "❌ Pull failed, skipping deployment"
    fi
  fi
done
