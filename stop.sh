#!/bin/bash
# 停止轻量版（按端口释放，不依赖 PID 文件）
PORT="${PUPIL_PORT:-8081}"
if fuser -k -n tcp "$PORT" >/dev/null 2>&1; then
  sleep 1
  echo "已停止：端口 $PORT"
else
  echo "端口 $PORT 上没有在运行的服务"
fi
