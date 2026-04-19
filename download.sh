#!/bin/bash
SERVER_USER="yuehailin"
SERVER_IP="122.207.108.7"
SERVER_PORT="19000"
LOCAL_BASE="/Users/zzfly/毕设"
LOCAL_VILA_DIR="${LOCAL_BASE}/ViLa-MIL"

echo "========================================"
echo "ViLa-MIL 项目下载脚本"
echo "========================================"

mkdir -p "${LOCAL_BASE}"

echo "选择下载内容:"
echo "1) 仅项目代码"
echo "2) 项目代码 + LUSC 特征数据"
echo "3) 项目代码 + 指定癌症特征数据"
read -p "请选择 [1/2/3]: " choice

echo ""
echo "--- 下载项目代码 ---"
rsync -avz -e "ssh -p ${SERVER_PORT}" --progress \
    --exclude='logging/*' --exclude='result/*' \
    --exclude='ckpt/*.pth' --exclude='__pycache__' \
    yuehailin@122.207.108.7:/home/yuehailin/dazewen/ViLa-MIL \
    "${LOCAL_VILA_DIR}/"

if [ "$choice" = "2" ]; then
    cancers=("LUSC")
elif [ "$choice" = "3" ]; then
    echo "可选：BLCA BRCA COAD ESCA HNSC KIRC LGG LIHC LU    echo "可选：BLCA BRC -    echo "可选：BL (?   echo "可选：ut
    cancers=($input)
else
    exit 0
fi

for cancer in "${cancers[@]}"; do
    echo ""
    echo "=== 下载 $cancer 特征数据 ==="
    remote="/home/yuehailin/TCGA/TCGA_feature/${cancer}"
    local="${LOCAL_VILA_DIR}/TCGA_feature/${cancer}"
    mkdir -p "$local"
    
    rsync -avz -e "ssh -p ${SERVER_PORT}" --progress \
        yuehailin@122.207.108.7:${remote}/20 "${local}/20"
    rsync -avz -e "ssh -p ${SERVER_PORT}" --progress \
        yuehailin@122.207.108.7:${remote}/10 "${local}/10"
    
    echo "✅ $cancer 完成"
done

echo ""
echo "🎉 下载完成！位置：${LOCAL_VILA_DIR}"
