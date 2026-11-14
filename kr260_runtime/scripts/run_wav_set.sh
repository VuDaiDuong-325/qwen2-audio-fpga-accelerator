#!/bin/bash
# Chay llm_runtime --wav cho ca mot thu muc WAV tren board, ghi log + chu + results.tsv (de cham diem bang score_wav_set.py).
#   ./run_wav_set.sh WAV_DIR [OUT_DIR=wavset_out] [MAX_NEW=80] [THREADS=4]
# Bien moi truong (tuy chon): PKG (../pkg_v1)  TEMPLATE (in_clip00/ids.i32)  BIN (./llm_runtime)  BACKEND (npu)
#   EXTRA (them tuy chon cho llm_runtime, vd "--vocab DIR")  RUN_PREFIX ("sudo env NPU_PL_OK=1")  LIMIT (chi chay N file dau)  FORCE=1 (chay lai file da co ket qua)  YES=1 (khong hoi khi co nguoi khac dang nhap)
# Khong dong vao bitstream/boot/kernel. Chay lai lenh nay sau khi bi ngat: file da co DECODE trong log se duoc bo qua.
set -u
WAV_DIR=${1:?"cach dung: $0 WAV_DIR [OUT_DIR] [MAX_NEW] [THREADS]"}
OUT=${2:-wavset_out}; MAX_NEW=${3:-80}; THREADS=${4:-4}
PKG=${PKG:-../pkg_v1}; TEMPLATE=${TEMPLATE:-in_clip00/ids.i32}; BIN=${BIN:-./llm_runtime}; BACKEND=${BACKEND:-npu}
RUN_PREFIX=${RUN_PREFIX-"sudo env NPU_PL_OK=1"}
mkdir -p "$OUT"

others=$(who | awk '{print $1}' | sort -u | grep -vx "$(id -un)" | tr '\n' ' ')
if [ -n "$others" ] && [ "${YES:-0}" != "1" ]; then
  echo "CANH BAO: nguoi dung khac dang dang nhap: $others (chi chay NPU khi khong ai dung board)"
  read -r -p "Tiep tuc? [y/N] " a; [ "$a" = "y" ] || exit 1
fi

files=$(ls "$WAV_DIR"/*.wav 2>/dev/null | sort -V)
[ -n "$files" ] || { echo "khong co file .wav trong $WAV_DIR"; exit 1; }
[ -n "${LIMIT:-}" ] && files=$(echo "$files" | head -n "$LIMIT")
n=$(echo "$files" | wc -l); i=0
TSV="$OUT/results.tsv"
printf 'file\taudio_rows\tlogmel_s\tencoder_s\tprefill_s\ttokens\tdecode_s\ttok_per_s\ttotal_s\ttext\n' > "$TSV.new"
t_all=$(date +%s)
for f in $files; do
  i=$((i+1)); b=$(basename "$f" .wav); log="$OUT/$b.log"; txt="$OUT/$b.txt"
  if [ "${FORCE:-0}" = "1" ] || ! grep -q '^DECODE:' "$log" 2>/dev/null; then
    echo "[$i/$n] $b ..."
    $RUN_PREFIX "$BIN" --pkg "$PKG" --wav "$f" --template "$TEMPLATE" --backend "$BACKEND" --max-new "$MAX_NEW" --threads "$THREADS" ${EXTRA:-} --text-out "$txt" > "$log" 2>&1
    rc=$?
    [ $rc -eq 0 ] || echo "   loi (ma $rc), xem $log"
  else
    echo "[$i/$n] $b: da co ket qua, bo qua (FORCE=1 de chay lai)"
  fi
  enc=$(grep '^\[encoder\] mel' "$log" | head -1)
  rows=$(echo "$enc" | sed -n 's/.*-> \([0-9]*\) hang.*/\1/p')
  lm=$(echo "$enc" | sed -n 's/.*log-mel \([0-9.]*\) s.*/\1/p')
  es=$(echo "$enc" | sed -n 's/.*| encoder \([0-9.]*\) s.*/\1/p')
  pf=$(grep '^\[prefill\] [0-9]* hang' "$log" | head -1 | sed -n 's/.*hang: \([0-9.]*\) s.*/\1/p')
  dec=$(grep '^DECODE:' "$log" | head -1)
  tk=$(echo "$dec" | sed -n 's/DECODE: \([0-9]*\) token.*/\1/p')
  ds=$(echo "$dec" | sed -n 's/.*token trong \([0-9.]*\) s.*/\1/p')
  tps=$(echo "$dec" | sed -n 's/.*= \([0-9.]*\) token\/s.*/\1/p')
  if [ -n "$dec" ]; then
    tot=$(awk -v a="${lm:-0}" -v b="${es:-0}" -v c="${pf:-0}" -v d="${ds:-0}" 'BEGIN{printf "%.2f", a+b+c+d}')
    text=$(tr '\t\r\n' '   ' < "$txt" 2>/dev/null | sed 's/  */ /g; s/^ //; s/ $//')
  else
    tot=""; text="FAIL"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$b" "$rows" "$lm" "$es" "$pf" "$tk" "$ds" "$tps" "$tot" "$text" >> "$TSV.new"
  [ -n "$dec" ] && echo "   audio $rows hang | encoder ${es}s | prefill ${pf}s | $tk token, $tps tok/s | tong ${tot}s"
done
mv "$TSV.new" "$TSV"
echo
echo "Xong $n file trong $(( $(date +%s) - t_all )) s. Ket qua: $TSV"
awk -F'\t' 'NR>1 && $8!="" {s+=$8; c++; t+=$6} END{if(c) printf "Trung binh %.2f token/s tren %d clip (%d token)\n", s/c, c, t}' "$TSV"
