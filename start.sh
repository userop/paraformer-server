#!/usr/bin/bash


HOST_IP=$(hostname -I)
# 每个进程实例1个流式模型+1个长语音模型
## 根据配置文件设定的GPU数量
## 总共有 n_works * n_gpu * 2个模型

uvicorn main:app --port 8400 --host $HOST_IP --workers "$APP_N_WORKERS"
