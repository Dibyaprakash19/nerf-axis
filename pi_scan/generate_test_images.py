#!/usr/bin/env python3
"""
generate_test_images.py
=======================
Downloads 40 test images into data/test_sets/ and saves them as:
    data/test_sets/set1_lego/0..3.jpg
    ...
    data/test_sets/set10_skull/0..3.jpg

Each set contains 4 different views of the specified category.
"""

import os
import sys
import urllib.request
import urllib.error

# ── Output root ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SETS_DIR = os.path.join(SCRIPT_DIR, 'data', 'test_sets')

# ── 10 sets × 4 angle images ─────────────────────────────────────────────────
# Each set has a category and 4 Unsplash URLs
SETS = [
    {
        "dir": "set1_lego",
        "urls": [
            "https://images.unsplash.com/photo-1585366119957-e556f4002a0c?w=640&q=80",
            "https://images.unsplash.com/photo-1472457897821-70d3819a0e24?w=640&q=80",
            "https://images.unsplash.com/photo-1560155016-bd4879ae8f21?w=640&q=80",
            "https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d?w=640&q=80",
        ]
    },
    {
        "dir": "set2_camera",
        "urls": [
            "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?w=640&q=80",
            "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=640&q=80",
            "https://images.unsplash.com/photo-1484101403633-562f891dc89a?w=640&q=80",
            "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=640&q=80",
        ]
    },
    {
        "dir": "set3_bonsai",
        "urls": [
            "https://images.unsplash.com/photo-1512428813834-c702c7702b78?w=640&q=80",
            "https://images.unsplash.com/photo-1576404285197-046645398246?w=640&q=80",
            "https://images.unsplash.com/photo-1599591459207-628d05260170?w=640&q=80",
            "https://images.unsplash.com/photo-1594411130635-41e17133f86e?w=640&q=80",
        ]
    },
    {
        "dir": "set4_vase",
        "urls": [
            "https://images.unsplash.com/photo-1612196808214-b8e1d6145a8c?w=640&q=80",
            "https://images.unsplash.com/photo-1578500494198-246f612d3b3d?w=640&q=80",
            "https://images.unsplash.com/photo-1565193566173-7a0ee3dbe261?w=640&q=80",
            "https://images.unsplash.com/photo-1541123437800-1bb1317badc2?w=640&q=80",
        ]
    },
    {
        "dir": "set5_shoe",
        "urls": [
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=640&q=80",
            "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=640&q=80",
            "https://images.unsplash.com/photo-1600185365778-1f41d5b6e8b2?w=640&q=80",
            "https://images.unsplash.com/photo-1605348532760-6753d2c43329?w=640&q=80",
        ]
    },
    {
        "dir": "set6_fossil",
        "urls": [
            "https://images.unsplash.com/photo-1518384401463-d3876163c195?w=640&q=80",
            "https://images.unsplash.com/photo-1582738411706-bfc8e691d1c2?w=640&q=80",
            "https://images.unsplash.com/photo-1591160690555-5debfba289f0?w=640&q=80",
            "https://images.unsplash.com/photo-1525281260342-7abc9c008f5c?w=640&q=80",
        ]
    },
    {
        "dir": "set7_mug",
        "urls": [
            "https://images.unsplash.com/photo-1514228742587-6b1558fcca3d?w=640&q=80",
            "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=640&q=80",
            "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=640&q=80",
            "https://images.unsplash.com/photo-1544787219-7f47ccb76574?w=640&q=80",
        ]
    },
    {
        "dir": "set8_robot",
        "urls": [
            "https://images.unsplash.com/photo-1531746790731-6c087fecd65a?w=640&q=80",
            "https://images.unsplash.com/photo-1589254065878-42c9da997008?w=640&q=80",
            "https://images.unsplash.com/photo-1558494949-ef010cbdcc4b?w=640&q=80",
            "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=640&q=80",
        ]
    },
    {
        "dir": "set9_car",
        "urls": [
            "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=640&q=80",
            "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=640&q=80",
            "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=640&q=80",
            "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=640&q=80",
        ]
    },
    {
        "dir": "set10_skull",
        "urls": [
            "https://images.unsplash.com/photo-1506143323312-326938a9d164?w=640&q=80",
            "https://images.unsplash.com/photo-1457914182927-652bb67bd29a?w=640&q=80",
            "https://images.unsplash.com/photo-1553173380-69274296ca56?w=640&q=80",
            "https://images.unsplash.com/photo-1520626337972-ebf863448db6?w=640&q=80",
        ]
    },
]

ANGLE_NAMES = ["front (0°)", "left (90°)", "back (180°)", "right (270°)"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Pi3DScanner/1.0; +https://github.com/user/nerf-axis)"
}

def download(url: str, dest: str, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp, \
                 open(dest, 'wb') as f:
                f.write(resp.read())
            return True
        except (urllib.error.URLError, OSError) as e:
            print(f"    Attempt {attempt}/{retries} failed: {e}")
    return False

def main():
    os.makedirs(TEST_SETS_DIR, exist_ok=True)
    total = len(SETS) * 4
    done  = 0
    failed = []

    print(f"Downloading {total} images into {TEST_SETS_DIR}\n")

    for s_idx, s in enumerate(SETS):
        set_dir = os.path.join(TEST_SETS_DIR, s['dir'])
        os.makedirs(set_dir, exist_ok=True)
        print(f"[{s['dir']}]")

        for angle_idx, url in enumerate(s['urls']):
            dest = os.path.join(set_dir, f"{angle_idx}.jpg")
            angle_name = ANGLE_NAMES[angle_idx]

            if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                print(f"  ✓ {angle_name:15s} already present — skip")
                done += 1
                continue

            print(f"  ↓ {angle_name:15s} …", end='', flush=True)
            ok = download(url, dest)
            if ok:
                size_kb = os.path.getsize(dest) // 1024
                print(f" {size_kb} KB")
                done += 1
            else:
                print(f" FAILED")
                failed.append((s['dir'], angle_idx, url))

    print(f"\n{'='*50}")
    print(f"Downloaded: {done}/{total}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for d, a, u in failed:
            print(f"  {d}/{a}.jpg  {u}")
    else:
        print("All images downloaded successfully ✓")
    print(f"Images saved to: {TEST_SETS_DIR}")

if __name__ == "__main__":
    main()
