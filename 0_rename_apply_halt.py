#!/usr/bin/env python3
"""
rename_2s.py

Find every directory named "aligned_data" under ROOT_DIR, and:
 - rename files containing ":2s" -> "_2s"
 - optionally rename directories containing ":2s" -> "_2s" (bottom-up to avoid traversal issues)

Usage:
    python3 rename_2s.py [--root ROOT_DIR] [--rename-dirs] [--dry-run]

Defaults:
    ROOT_DIR = "/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20250409_Cohort3_rotation"
"""

import argparse
from pathlib import Path
import shutil
import sys

def find_aligned_dirs(root: Path):
    return [p for p in root.rglob('aligned_data') if p.is_dir()]

def safe_move(src: Path, dst: Path, dry_run: bool):
    if dst.exists():
        print(f"[SKIP] target exists: {dst}")
        return False
    print(f"{'DRY ' if dry_run else ''}RENAME: {src} -> {dst}")
    if not dry_run:
        try:
            # use shutil.move which can handle cross-filesystem moves
            shutil.move(str(src), str(dst))
            return True
        except Exception as e:
            print(f"[ERROR] failed to rename {src} -> {dst}: {e}")
            return False
    return True

def process_files_in_dir(aligned_dir: Path, dry_run: bool):
    renamed = 0
    errors = 0
    for f in aligned_dir.iterdir():
        # only files (not directories)
        if f.is_file() and ": 2s" in f.name:
            new_name = f.name.replace(": 2s", "_2s")
            new_path = f.with_name(new_name)
            ok = safe_move(f, new_path, dry_run)
            if ok:
                renamed += 1
            else:
                errors += 1
    return renamed, errors

def process_dirs_with_colon2s(root: Path, dry_run: bool):
    # Rename directories containing ":2s"
    # Do bottom-up so we don't break traversal (sort by depth descending)
    dirs = [p for p in root.rglob('*') if p.is_dir() and ":2s" in p.name]
    dirs_sorted = sorted(dirs, key=lambda p: len(p.parts), reverse=True)
    renamed = 0
    errors = 0
    for d in dirs_sorted:
        new_name = d.name.replace(": 2s", "_2s")
        new_path = d.with_name(new_name)
        ok = safe_move(d, new_path, dry_run)
        if ok:
            renamed += 1
        else:
            errors += 1
    return renamed, errors

def main():
    parser = argparse.ArgumentParser(description="Rename ': 2s' -> '_2s' in files inside aligned_data folders.")
    parser.add_argument('--root', '-r',
                        default="/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20250409_Cohort3_rotation,
                        help="Root directory to search under.")
    parser.add_argument('--rename-dirs', action='store_true',
                        help="Also rename directories containing ': 2s' -> '_2s' (bottom-up).")
    parser.add_argument('--dry-run', action='store_true',
                        help="Only print what would be done; do not actually rename.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root directory does not exist or is not a directory: {root}")
        sys.exit(1)

    aligned_dirs = find_aligned_dirs(root)
    print(f"Found {len(aligned_dirs)} 'aligned_data' directory(ies) under {root}\n")

    total_renamed_files = 0
    total_file_errors = 0

    for ad in aligned_dirs:
        print(f"Processing: {ad}")
        r, e = process_files_in_dir(ad, args.dry_run)
        total_renamed_files += r
        total_file_errors += e

    total_renamed_dirs = 0
    total_dir_errors = 0
    if args.rename_dirs:
        print("\nRenaming directories containing ': 2s' (bottom-up)...")
        r, e = process_dirs_with_colon2s(root, args.dry_run)
        total_renamed_dirs += r
        total_dir_errors += e

    print("\n--- Summary ---")
    print(f"Aligned_data dirs processed : {len(aligned_dirs)}")
    print(f"Files renamed               : {total_renamed_files}")
    print(f"File rename errors/skipped  : {total_file_errors}")
    if args.rename_dirs:
        print(f"Dirs renamed                : {total_renamed_dirs}")
        print(f"Dir rename errors/skipped   : {total_dir_errors}")
    print(f"{'DRY RUN - no changes made' if args.dry_run else 'Changes applied'}")

if __name__ == "__main__":
    main()

