#!/usr/bin/env python3
"""测试训练阶段边界是否正确设置"""

from src.train import get_training_stage
import json

# 读取配置
with open('config.json', 'r') as f:
    config = json.load(f)
total_epochs = config.get('epochs', 60)

print("=" * 70)
print("训练阶段边界测试")
print("=" * 70)
print()

# 测试关键轮次
test_epochs = [1, 10, 20, 30, 31, 40, 45, 46, 50, 60]
test_epochs = [e for e in test_epochs if e <= total_epochs]

# 添加总轮次
if total_epochs not in test_epochs:
    test_epochs.append(total_epochs)

# 添加边界前后
stage1_end = None
stage2_end = None

for e in range(1, total_epochs + 1):
    s = get_training_stage(e)
    if s == 1 and stage1_end is None:
        pass
    elif s == 2 and stage1_end is None:
        stage1_end = e - 1
    elif s == 3 and stage2_end is None:
        stage2_end = e - 1

# 添加边界点
for boundary in [stage1_end, stage1_end + 1, stage2_end, stage2_end + 1]:
    if boundary and boundary not in test_epochs and boundary <= total_epochs:
        test_epochs.append(boundary)

test_epochs = sorted(set(test_epochs))

print("关键轮次测试:")
print()
prev_stage = None
for e in test_epochs:
    stage = get_training_stage(e)
    marker = ""
    if prev_stage is not None and stage != prev_stage:
        marker = f" ← Stage {prev_stage} → {stage} 边界"
    print(f"  Epoch {e:3d}: Stage {stage}{marker}")
    prev_stage = stage

print()
print("=" * 70)
print("阶段划分总结:")
print("=" * 70)
print(f"  Stage 1 (基础训练):    Epoch 1 - {stage1_end}")
print(f"  Stage 2 (约束训练):    Epoch {stage1_end + 1} - {stage2_end}")
print(f"  Stage 3 (底部聚焦):    Epoch {stage2_end + 1} - {total_epochs}")
print()
print(f"总轮次: {total_epochs}")
print(f"  Stage 1: {stage1_end} 轮")
print(f"  Stage 2: {stage2_end - stage1_end} 轮")
print(f"  Stage 3: {total_epochs - stage2_end} 轮")
print()

# 检查配置合理性
issues = []
if stage1_end < 10:
    issues.append(f"⚠️  Stage 1 只有 {stage1_end} 轮，可能太短（建议至少 10 轮）")
if stage2_end - stage1_end < 5:
    issues.append(f"⚠️  Stage 2 只有 {stage2_end - stage1_end} 轮，可能太短（建议至少 5 轮）")
if total_epochs - stage2_end < 5:
    issues.append(f"⚠️  Stage 3 只有 {total_epochs - stage2_end} 轮，可能太短（建议至少 5 轮）")

if issues:
    print("配置建议:")
    for issue in issues:
        print(f"  {issue}")
    print()
else:
    print("✅ 阶段划分合理\n")
