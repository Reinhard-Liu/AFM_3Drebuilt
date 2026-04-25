#!/usr/bin/env python3
"""
实时监控训练进度，报告每个 epoch 的完成情况
"""

import time
import os
import re

output_file = "/tmp/claude-0/-root-autodl-tmp/tasks/b6fe581.output"
last_position = 0
last_reported_epoch = 0

print("开始监控训练进度...")
print("=" * 70)
print()

while True:
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                f.seek(last_position)
                new_content = f.read()
                last_position = f.tell()

                if new_content:
                    # 查找 Epoch 完成信息
                    epoch_pattern = r'\[Epoch (\d+)\] Train Loss: ([\d.]+), Val Loss: ([\d.]+)'
                    rmsd_pattern = r'\[Epoch (\d+)\] RMSD: ([\d.]+) \+/- ([\d.]+)'
                    metrics_pattern = r'\[Epoch (\d+)\] Bottom Recall: ([\d.]+)'

                    for line in new_content.split('\n'):
                        # Epoch 完成信息
                        match = re.search(epoch_pattern, line)
                        if match:
                            epoch = int(match.group(1))
                            train_loss = float(match.group(2))
                            val_loss = float(match.group(3))

                            if epoch > last_reported_epoch:
                                # 读取该 epoch 的所有指标
                                rmsd_info = ""
                                metrics_info = ""

                                # 向后查找 RMSD 和其他指标
                                for next_line in new_content.split('\n'):
                                    if f"[Epoch {epoch}] RMSD:" in next_line:
                                        rmsd_match = re.search(r'RMSD: ([\d.]+) \+/- ([\d.]+)', next_line)
                                        if rmsd_match:
                                            rmsd_info = f"RMSD: {rmsd_match.group(1)} ± {rmsd_match.group(2)}"

                                    if f"[Epoch {epoch}] Bottom Recall:" in next_line:
                                        recall_match = re.search(r'Bottom Recall: ([\d.]+)', next_line)
                                        if recall_match:
                                            metrics_info = f"Bottom Recall: {recall_match.group(1)}"

                                # 确定训练阶段
                                if epoch <= 30:
                                    stage = "Stage 1"
                                elif epoch <= 45:
                                    stage = "Stage 2"
                                else:
                                    stage = "Stage 3"

                                print(f"[Epoch {epoch:2d}/60] {stage}")
                                print(f"  Train Loss: {train_loss:.4f}")
                                print(f"  Val Loss:   {val_loss:.4f}")
                                if rmsd_info:
                                    print(f"  {rmsd_info}")
                                if metrics_info:
                                    print(f"  {metrics_info}")
                                print()

                                last_reported_epoch = epoch

                        # 检查是否训练完成
                        if "Training Complete!" in line:
                            print("=" * 70)
                            print("🎉 训练完成！")
                            print("=" * 70)
                            exit(0)

                        # 检查是否早停
                        if "[Early Stop]" in line:
                            print("=" * 70)
                            print(f"⚠️  {line.strip()}")
                            print("=" * 70)
                            exit(0)

        time.sleep(30)  # 每30秒检查一次

    except KeyboardInterrupt:
        print("\n监控中断")
        exit(0)
    except Exception as e:
        print(f"错误: {e}")
        time.sleep(30)
