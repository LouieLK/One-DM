#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遞迴瀏覽來源目錄下所有子目錄，
在目標目錄中重建相同的目錄結構，
並在每個子目錄內隨機挑選指定數量的檔案進行複製。
"""
import os
import shutil
import random
import argparse


def copy_random_files(src_root: str, dst_root: str, num_files: int = 200) -> None:
    """
    遞迴瀏覽來源目錄並複製隨機檔案到新目錄。

    :param src_root: 來源目錄根路徑
    :param dst_root: 目標目錄根路徑
    :param num_files: 每個目錄要複製的檔案數量
    """
    for dirpath, dirnames, filenames in os.walk(src_root):
        # 計算相對路徑，並在目標根路徑中新建相同目錄
        rel_path = os.path.relpath(dirpath, src_root)
        dst_dir = os.path.join(dst_root, rel_path)
        os.makedirs(dst_dir, exist_ok=True)

        # 若該目錄內無檔案，則略過
        if not filenames:
            continue

        # 篩選出檔案完整路徑
        file_paths = [os.path.join(dirpath, f) for f in filenames
                      if os.path.isfile(os.path.join(dirpath, f))]

        # 隨機抽樣（若檔案數量少於目標數，則全部複製）
        selected = random.sample(file_paths, min(num_files, len(file_paths)))

        # 複製檔案
        for src_file in selected:
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)

        print(f"[INFO] 已將 {len(selected)} 個檔案複製到: {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="遞迴複製隨機檔案並保留目錄結構"
    )
    parser.add_argument(
        "src",
        help="來源根目錄路徑"
    )
    parser.add_argument(
        "dst",
        help="目標根目錄路徑"
    )
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=200,
        help="每個目錄要複製的檔案數量，預設為200"
    )
    args = parser.parse_args()
    copy_random_files(args.src, args.dst, args.num)
