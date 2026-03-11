#!/usr/bin/env python3
"""
遍历 OakInk/OakBase 所有物体的 part JSON:
1. 收集 (object_category, raw_name, attr) 三元组
2. 去掉物体名称前缀 → stripped_name
3. 用 token_config.yaml 的映射规则 → normalized_name → canonical_slot
4. 按物体类别去重后统计频次，输出汇总表

注意：本脚本与 data/slot_mapping.py 使用同一套配置，确保分析结果与运行时一致。
"""

import os
import sys
import json
import glob
import yaml
from collections import Counter, defaultdict
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.slot_mapping import SlotMapper

OAKBASE_ROOT = "/home/kingston/wzc/data/OakInk/OakBase"
CONFIG_PATH = PROJECT_ROOT / "configs" / "token_config.yaml"

# ═══════════════════════════════════════════════════════════════════════
# 已知物体类别（即 OakBase 下的文件夹名）
# ═══════════════════════════════════════════════════════════════════════
OBJECT_CATEGORIES = [
    "binoculars", "bottle", "bowl", "camera", "can", "cup",
    "cylinder_bottle", "eyeglasses", "flashlight", "frying_pan",
    "game_controller", "hammer", "headphones", "knife", "lightbulb",
    "lotion_bottle", "marker", "mouse", "mug", "pincer", "power_drill",
    "scissor", "screwdriver", "squeeze_tube", "teapot", "toothbrush",
    "trigger_sprayer", "wineglass", "wrench",
]

# ═══════════════════════════════════════════════════════════════════════
# 需要从 name 中去掉的物体前缀列表
# OakInk 的 part name 常常以 "<物体>_" 开头，如 "knife_blade"
# ═══════════════════════════════════════════════════════════════════════
OBJECT_PREFIXES = [
    # 直接对应类别名
    "binoculars_", "bottle_", "bowl_", "camera_", "can_", "cup_",
    "cylinder_bottle_", "flashlight_", "frying_pan_",
    "game_controller_", "hammer_", "headphones_", "knife_",
    "lightbulb_", "lotion_bottle_", "marker_", "mouse_", "mug_",
    "pincer_", "power_drill_", "scissor_", "screwdriver_",
    "squeeze_tube_", "teapot_", "toothbrush_", "trigger_sprayer_",
    "wineglass_", "wrench_",
    # 缩写/别名前缀
    "drill_", "pan_", "pen_", "brush_", "tube_",
    "controller_",
]


def strip_prefix(raw_name: str) -> str:
    """去掉物体名称前缀，返回纯功能 part 名。"""
    lower = raw_name.lower()
    # 尝试匹配最长前缀优先
    best = ""
    for pfx in OBJECT_PREFIXES:
        if lower.startswith(pfx) and len(pfx) > len(best):
            best = pfx
    if best:
        return lower[len(best):]
    return lower


def collect_all_parts(root):
    """收集所有物体的 part JSON 记录。"""
    records = []
    for json_path in sorted(glob.glob(os.path.join(root, "*", "*", "part_*.json"))):
        rel = os.path.relpath(json_path, root)
        parts = rel.replace("\\", "/").split("/")
        category = parts[0]
        obj_id = parts[1]
        with open(json_path, "r") as f:
            data = json.load(f)
        records.append({
            "category": category,
            "obj_id": obj_id,
            "raw_name": data["name"],
            "attr": data["attr"],
            "file": rel,
        })
    return records


def attr_to_slot(attrs: list[str]) -> str:
    """
    根据 attr 集合推断 canonical slot（兜底用）。
    与 token_config.yaml 的 4-slot 体系保持一致。
    """
    a = set(attrs)
    
    # 控制件：触发器/按钮
    if a & {"trigger_sth", "control_sth"}:
        return "CONTROL"
    if "pressed/unpressed_by_hand" in a and "held_by_hand" in a:
        return "CONTROL"
    
    # 功能端：工具头/发射端等
    if a & {"cut_sth", "stab_sth", "shear_sth", "knock_sth",
            "tighten_sth", "loosen_sth", "drill_sth", "brush_sth",
            "draw_sth", "clamp_sth"}:
        return "FUNCTIONAL_END"
    if a & {"illuminate_sth", "spray_sth", "observe_sth", "pump_out_sth"}:
        return "FUNCTIONAL_END"
    if "flow_out_sth" in a and "contain_sth" not in a:
        return "FUNCTIONAL_END"
    
    # 封闭件
    if a & {"secure_sth", "cover_sth"}:
        return "CLOSURE"
    
    # 握持面：容器/主体/把手
    if "contain_sth" in a:
        return "GRASP_SURFACE"
    if a == {"held_by_hand"}:
        return "GRASP_SURFACE"
    
    # 穿戴类 - 归并到 GRASP_SURFACE
    if any("attach_to_" in x for x in a):
        return "GRASP_SURFACE"
    
    # 结构性 - 归并到 GRASP_SURFACE
    if a == {"no_function"}:
        return "GRASP_SURFACE"
    
    return "OTHER"


def main():
    # 加载 SlotMapper（使用与运行时相同的配置）
    mapper = SlotMapper(str(CONFIG_PATH))
    print(f"加载配置: {CONFIG_PATH}")
    print(f"Canonical slots: {mapper.canonical_slots}\n")
    
    records = collect_all_parts(OAKBASE_ROOT)
    print(f"扫描到 {len(records)} 个 part JSON\n")

    # ── 对每条记录做清洗 ──────────────────────────────────
    for r in records:
        r["stripped"] = strip_prefix(r["raw_name"])
        # 使用 SlotMapper 的 normalize_name 方法
        normalized = mapper.normalize_name(r["stripped"])
        r["normalized"] = normalized if normalized else r["stripped"]
        
        # 尝试通过配置映射到 slot
        slot = None
        if normalized:
            slot = mapper.normalized_to_slot.get(normalized)
        
        # 如果配置中没有映射，尝试通过 attr 兜底
        if slot is None:
            slot = attr_to_slot(r["attr"])
        
        r["slot"] = slot

    # ═══════════════════════════════════════════════════════
    # 表 1: raw_name → stripped → normalized → slot 全量映射
    # ═══════════════════════════════════════════════════════
    seen_map = {}
    for r in records:
        key = r["raw_name"]
        if key not in seen_map:
            seen_map[key] = r
    print("=" * 90)
    print("【表 1】raw_name → stripped → normalized → slot 映射")
    print("=" * 90)
    print(f"{'raw_name':45s} {'stripped':25s} {'normalized':20s} {'slot':18s}")
    print("-" * 90)
    for raw in sorted(seen_map.keys()):
        r = seen_map[raw]
        print(f"{r['raw_name']:45s} {r['stripped']:25s} {r['normalized']:20s} {r['slot']:18s}")

    # ═══════════════════════════════════════════════════════
    # 表 2: 每个 canonical slot 出现在多少个物体类别中（去重）
    # ═══════════════════════════════════════════════════════
    slot_to_cats = defaultdict(set)
    for r in records:
        slot_to_cats[r["slot"]].add(r["category"])

    # 每 slot 内，各 normalized name 出现在多少个类别中
    slot_norm_cats = defaultdict(lambda: defaultdict(set))
    for r in records:
        slot_norm_cats[r["slot"]][r["normalized"]].add(r["category"])

    print()
    print("=" * 90)
    print("【表 2】Canonical Slot 汇总（按类别去重的频次）")
    print("=" * 90)
    
    all_slots = mapper.canonical_slots + ["OTHER"]
    for slot in all_slots:
        cats = slot_to_cats.get(slot, set())
        print(f"\n┌─ {slot}  (覆盖 {len(cats)}/{len(OBJECT_CATEGORIES)} 个物体类别)")
        print(f"│  类别: {sorted(cats)}")
        print(f"│  归一化 name 明细:")
        for norm, norm_cats in sorted(slot_norm_cats[slot].items(),
                                       key=lambda x: -len(x[1])):
            print(f"│    {norm:30s}  出现于 {len(norm_cats):2d} 类: {sorted(norm_cats)}")
        print("└" + "─" * 60)

    # ═══════════════════════════════════════════════════════
    # 表 3: attr 频次统计
    # ═══════════════════════════════════════════════════════
    attr_counter = Counter()
    for r in records:
        attr_counter.update(r["attr"])
    print()
    print("=" * 90)
    print("【表 3】attr 出现频次")
    print("=" * 90)
    for attr, cnt in attr_counter.most_common():
        print(f"  {attr:45s}  {cnt:5d}")

    # ═══════════════════════════════════════════════════════
    # 表 4: 每个物体类别的 slot 组成
    # ═══════════════════════════════════════════════════════
    cat_to_slots = defaultdict(set)
    cat_slot_names = defaultdict(lambda: defaultdict(set))
    for r in records:
        cat_to_slots[r["category"]].add(r["slot"])
        cat_slot_names[r["category"]][r["slot"]].add(r["raw_name"])

    print()
    print("=" * 90)
    print("【表 4】每个物体类别的 slot 组成")
    print("=" * 90)
    for cat in sorted(cat_to_slots.keys()):
        slots = sorted(cat_to_slots[cat])
        print(f"\n  {cat}:")
        for s in slots:
            raw_names = sorted(cat_slot_names[cat][s])
            print(f"    {s:20s}  ←  {raw_names}")

    # ═══════════════════════════════════════════════════════
    # 检查：有没有落入 OTHER 的？
    # ═══════════════════════════════════════════════════════
    others = [r for r in records if r["slot"] == "OTHER"]
    if others:
        print()
        print("=" * 90)
        print(f"⚠️  落入 OTHER 的记录 ({len(others)} 条):")
        print("=" * 90)
        other_counter = Counter(
            (r["raw_name"], tuple(sorted(r["attr"]))) for r in others
        )
        for (name, attrs), cnt in other_counter.most_common():
            print(f"  {cnt:4d}x  {name:35s}  attr={list(attrs)}")
    else:
        print("\n✅ 没有任何记录落入 OTHER，全部归入了 canonical slot。")
    
    # ═══════════════════════════════════════════════════════
    # 汇总统计
    # ═══════════════════════════════════════════════════════
    print()
    print("=" * 90)
    print("【汇总】Slot 覆盖统计")
    print("=" * 90)
    total_parts = len(records)
    for slot in mapper.canonical_slots:
        count = len([r for r in records if r["slot"] == slot])
        pct = count / total_parts * 100
        print(f"  {slot:18s}: {count:4d} ({pct:5.1f}%)")
    other_count = len([r for r in records if r["slot"] == "OTHER"])
    if other_count > 0:
        pct = other_count / total_parts * 100
        print(f"  {'OTHER':18s}: {other_count:4d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
