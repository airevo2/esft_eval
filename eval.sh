#!/bin/bash

# 设置默认参数
eval_dataset="intent"
base_model_path="allenai/OLMoE-1B-7B-0125"
adapter_dir="all_models/adapters/token/intent"
model_identifier=$(echo "$base_model_path" | sed 's/\//_/g')
output_path="results/completions/token/${model_identifier}_intent.jsonl"
openai_api_key="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"

# 显式设置GPU使用参数
# total_gpu: 总共需要使用的GPU数量
# world_size: 使用的进程数
# gpus_per_rank: 每个进程分配的GPU数量，平均分配
total_gpu=1
world_size=1
# 检查能否整除，否则取整后可能有剩余，本处简单取整
gpus_per_rank=$(( total_gpu / world_size ))

# 定义日志文件
log_file="eval_$(date '+%Y%m%d_%H%M%S').log"

log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$log_file"
}

exec > >(tee -a "$log_file") 2>&1

# 检查nvidia-smi命令是否可用
if ! command -v nvidia-smi &> /dev/null; then
    log "错误: nvidia-smi 命令不可用，无法监控GPU状态"
    exit 1
fi

# 获取系统中所有可用GPU的数量
total_available_gpu=$(nvidia-smi -L | wc -l | tr -d ' ')
log "检测到 ${total_available_gpu} 张GPU"

if [ "$total_available_gpu" -lt "$total_gpu" ]; then
    log "错误: 请求的GPU数量 (${total_gpu}) 超过系统可用数量 (${total_available_gpu})"
    exit 1
fi

# 定义空闲阈值（MB）和检查间隔（秒）
mem_threshold=10000
check_interval=60

check_gpu_available() {
    local gpu_id=$1
    local used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    if [[ ! "$used_mem" =~ ^[0-9]+$ ]]; then
        log "警告: 无法获取GPU $gpu_id 的显存使用情况"
        return 1
    fi
    if [ "$used_mem" -lt "$mem_threshold" ]; then
        return 0
    else
        return 1
    fi
}

find_available_gpus() {
    local available=()
    for ((i=0; i<total_available_gpu; i++)); do
        if check_gpu_available $i; then
            available+=($i)
            log "GPU $i 空闲 (显存 < ${mem_threshold}MB)"
        else
            log "GPU $i 忙碌 (显存 >= ${mem_threshold}MB)"
        fi
    done
    log "当前共有 ${#available[@]} 张GPU空闲"
    if [ ${#available[@]} -ge $total_gpu ]; then
        # 返回前total_gpu个可用的GPU ID
        echo "${available[@]:0:$total_gpu}"
        return 0
    else
        return 1
    fi
}

# 主等待循环：等待足够的空闲GPU
while true; do
    log "检查GPU状态，等待 ${total_gpu} 张空闲GPU..."
    available_gpus=$(find_available_gpus)
    if [ $? -eq 0 ]; then
        log "发现空闲GPU: $available_gpus"
        break
    else
        log "没有达到所需的空闲GPU数量，等待 ${check_interval} 秒后重试..."
        sleep $check_interval
    fi
done

# 将获取的GPU ID列表转换为逗号分隔字符串
gpu_list=$(echo $available_gpus | tr ' ' ',')

log "使用GPU: $gpu_list"

# 运行评估命令，传入world_size和gpus_per_rank参数
CUDA_VISIBLE_DEVICES=$gpu_list python eval_multigpu.py \
    --eval_dataset="$eval_dataset" \
    --base_model_path="$base_model_path" \
    --adapter_dir="$adapter_dir" \
    --output_path="$output_path" \
    --openai_api_key="$openai_api_key" \
    --world_size="$world_size" \
    --gpus_per_rank="$gpus_per_rank"

eval_status=$?
if [ $eval_status -eq 0 ]; then
    log "评估完成，结果保存在: $output_path"
else
    log "评估失败，退出码: ${eval_status}"
fi

log "脚本执行结束"