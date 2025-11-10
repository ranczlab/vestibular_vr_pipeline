#!/usr/bin/env bash
set -euo pipefail

# ---- Force modern ffmpeg ----
FFMPEG="/opt/homebrew/bin/ffmpeg"
command -v "$FFMPEG" >/dev/null 2>&1 || { echo "Error: ffmpeg not found at $FFMPEG"; exit 1; }

# ---- Settings ----
QV="${QV:-8}"          # MJPEG quality (6–12); higher = smaller file
EXT="avi"              # Only process .avi
OUT_SIDE="VideoData_side_by_side_fiji.avi"

process_folder () {
  local dir="$1"
  [[ -d "$dir" ]] || { echo "Skipping: $dir (not a directory)"; return; }

  echo "==> Processing $dir ..."
  local absdir; absdir="$(cd "$dir" && pwd)"

  # Collect *.avi (sorted, NUL-safe)
  local files=()
  while IFS= read -r -d '' f; do files+=("$f"); done \
    < <(find "$absdir" -maxdepth 1 -type f -name "*.${EXT}" -print0 | LC_ALL=C sort -z)

  local n="${#files[@]}"
  if (( n == 0 )); then
    echo "   No .${EXT} files in $dir — skipping."
    return
  fi

  # Proper mktemp on macOS (needs .XXXXXX)
  local list_file; list_file="$(mktemp -q "${absdir}/.concat_list.XXXXXX")" || { echo "mktemp failed"; exit 1; }
  : > "$list_file"
  for f in "${files[@]}"; do printf "file '%s'\n" "$f" >> "$list_file"; done

  # Optional: show first few lines to verify
  echo "   concat list preview:"; head -n 4 "$list_file" || true

  local out="${absdir%/}_fiji_mjpeg.avi"

  # Concat demuxer → re-encode to Fiji-friendly MJPEG
  # NO -r flag: preserves original frame rate and frame count
  "$FFMPEG" -y -f concat -safe 0 -i "$list_file" \
    -vf "format=gray" \
    -c:v mjpeg -q:v "$QV" -an \
    "$out"

  rm -f "$list_file" || true
  echo "   ✅ Created $out"
}

# ---- 1) Build per-folder outputs ----
process_folder "VideoData1"
process_folder "VideoData2"

# ---- 2) Side-by-side composite (preserves original frame rates, stops at shorter) ----
in1="VideoData1_fiji_mjpeg.avi"
in2="VideoData2_fiji_mjpeg.avi"
out="$OUT_SIDE"

if [[ -f "$in1" && -f "$in2" ]]; then
  echo "==> Creating side-by-side composite..."
  echo "   ℹ️  Preserving original frame rates; output will match shorter input"
  
  if "$FFMPEG" -hide_banner -loglevel error -y \
      -i "$in1" -i "$in2" \
      -filter_complex "[0:v]format=gray,setsar=1[v0];[1:v]format=gray,setsar=1[v1];[v0][v1]hstack=inputs=2[v]" \
      -map "[v]" -c:v mjpeg -q:v "$QV" -an -shortest "$out"; then
    echo "   ✅ Built $out"
  else
    echo "   Height mismatch detected; retrying with scale2ref…"
    "$FFMPEG" -hide_banner -loglevel error -y \
      -i "$in1" -i "$in2" \
      -filter_complex "[0:v]format=gray,setsar=1[v0];\
                       [1:v]format=gray,setsar=1[v1];\
                       [v1][v0]scale2ref=w=oh*mdar:h=ih[v1s][v0r];\
                       [v0r][v1s]hstack=inputs=2[v]" \
      -map "[v]" -c:v mjpeg -q:v "$QV" -an -shortest "$out"
    echo "   ✅ Built $out"
  fi
else
  echo "⚠️  Missing one of the concatenated files: $in1 or $in2"
fi

echo "🎬 All done."
echo "Tip: smaller files → run with QV=10 (or 12):  QV=10 ~/build_fiji_videos.sh"
