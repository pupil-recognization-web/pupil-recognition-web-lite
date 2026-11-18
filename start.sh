#!/bin/bash
# 启动轻量版（默认 8081）。已运行则直接提示，不重复起。
set -e
cd "$(dirname "$0")"
PORT="${PUPIL_PORT:-8081}"
PY="${PUPIL_PYTHON:-/mnt/backup/miniconda3/envs/yolo/bin/python}"
[ -x "$PY" ] || PY=python

if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/"; then
  echo "已在运行：http://127.0.0.1:$PORT"
  echo "（要重启：./stop.sh 然后再 ./start.sh）"
  exit 0
fi

setsid nohup "$PY" app.py >> app.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 2
  if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/"; then
    echo "已启动：http://127.0.0.1:$PORT"
    echo "日志：$(pwd)/app.log"
    exit 0
  fi
done
echo "启动超时，最近日志："
tail -20 app.log
exit 1
