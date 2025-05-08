#!/bin/bash

# 创建日志目录
mkdir -p logs
log_file="logs/eval_single_$(date '+%Y%m%d_%H%M%S').log"

# 检查 NVIDIA-SMI 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 命令不可用，无法监控GPU状态" | tee -a "$log_file"
    exit 1
fi

# 获取所有可用GPU的数量
gpu_count=$(nvidia-smi -L | wc -l)
echo "检测到 $gpu_count 张GPU" | tee -a "$log_file"

# 定义所需GPU数量（保持原来--gpu_count 1）
total_gpu=1
# 定义空闲阈值（MB）和检查间隔（秒）
mem_threshold=10000
check_interval=60

# 监控函数：检查指定GPU是否空闲
check_gpu_available() {
    local gpu_id=$1
    local used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    if [[ ! "$used_mem" =~ ^[0-9]+$ ]]; then
        echo "警告: 无法获取GPU $gpu_id 的显存使用情况" >&2
        return 1
    fi
    if [ "$used_mem" -lt "$mem_threshold" ]; then
        return 0
    else
        return 1
    fi
}

# 查找空闲的GPU，返回逗号分隔的GPU id字符串
find_available_gpus() {
    local required_gpus=$1
    local available_gpus=()
    for ((i=0; i<gpu_count; i++)); do
        if check_gpu_available $i; then
            available_gpus+=($i)
        fi
    done
    if [ ${#available_gpus[@]} -ge $required_gpus ]; then
        local gpu_ids=$(IFS=,; echo "${available_gpus[*]:0:$required_gpus}")
        echo "$gpu_ids"
    else
        echo ""
    fi
}

# 等待直到有足够空闲的GPU
while true; do
    available_gpu_list=$(find_available_gpus $total_gpu)
    if [ -n "$available_gpu_list" ]; then
        echo "找到空闲GPU: $available_gpu_list" | tee -a "$log_file"
        export CUDA_VISIBLE_DEVICES="$available_gpu_list"
        break
    else
        echo "未找到 $total_gpu 张空闲GPU，等待 $check_interval 秒后重试..." | tee -a "$log_file"
        sleep $check_interval
    fi
done

# 定义模型和任务数组（目前与原来保持一致，但支持扩展多模型多任务）
models=(
autoprogrammer/OLMoE-1B-7B-0125_ESFT-translation_freeze


)
tasks=("translation")

# 循环依次评估各模型和各任务
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        # 输出目录与日志信息，保持与原始命令一致
        output_dir="results/${task}"
        mkdir -p "$output_dir"
        echo "开始评估模型 $model 在任务 $task 上的表现" | tee -a "$log_file"
        echo "执行命令: python eval.py --eval_datasets=$task --model_path=$model --output_dir=$output_dir --max_new_tokens=512 --openai_api_key=\"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\" --eval_batch_size=2 --gpu_count 1" | tee -a "$log_file"
        
        python eval.py \
            --eval_datasets="$task" \
            --model_path="$model" \
            --output_dir="$output_dir" \
            --max_new_tokens=512 \
            --openai_api_key="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" \
            --eval_batch_size=1 \
            --gpu_count 1 >> "$log_file" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "评估成功: 模型 $model, 任务 $task" | tee -a "$log_file"
        else
            echo "评估失败: 模型 $model, 任务 $task" | tee -a "$log_file"
        fi
        echo "----------------------------------------" | tee -a "$log_file"
    done
done

echo "所有评估任务已完成" | tee -a "$log_file" 