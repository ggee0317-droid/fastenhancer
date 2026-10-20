#!/usr/bin/env bash
# lowlat 재학습 실행 스크립트 (WSL2, 저장소 루트에서)
# 사용법: bash run.sh sanity | time X | train X | export X | export_plain X   (X = l0ft, l2, l3)
# 메모리가 부족하면 앞에 BS=32 를 붙임. 예: BS=32 bash run.sh train l2
set -e
cd "$(dirname "$0")"
[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ] && source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate fe
B=${BS:+train.batch_size=$BS valid.batch_size=$BS}
case "$1" in
  sanity) python train.py -n lowlat/sanity -c configs/lowlat/t_l0ft.yaml -f \
            -p train.test=True pesq.interval=1 train.optimizer_kwargs.lr=1.0e-6 ;;
  time)   python train.py -n lowlat/time_$2 -c configs/lowlat/t_$2.yaml -f \
            -p train.max_epochs=2 pesq.interval=1000 $B ;;
  train)  if ls logs/lowlat/t_$2/*.pth >/dev/null 2>&1
          then python train.py -n lowlat/t_$2 ${B:+-f -p $B}                    # 체크포인트가 있으면 이어서 (-p를 쓰면 -f 필요)
          else python train.py -n lowlat/t_$2 -c configs/lowlat/t_$2.yaml -f ${B:+-p $B}; fi ;;
  export) python -m scripts.export_onnx -n lowlat/t_$2 --onnx-path onnx/fastenhancer_t_$2.onnx ;;
  export_plain) python -m scripts.export_onnx -n lowlat/t_$2 --onnx-path onnx/fastenhancer_t_$2.onnx --optimization none ;;
  *) echo "사용법: bash run.sh sanity | time X | train X | export X | export_plain X  (X = l0ft, l2, l3)"; exit 1 ;;
esac
