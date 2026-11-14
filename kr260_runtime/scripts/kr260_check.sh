#!/usr/bin/env bash
# =============================================================================
# kr260_check.sh - tong hop lenh kiem tra NPU INT8 tren KR260 (thay the kv260_check.sh; CHI DUNG CHO KR260)
# Dat trong thu muc tools/ cung cho voi: run.sh (nap bitstream), kr260_npu_driver.h (BAN VA -DNPU_DMAHEAP),
# npu_run_case.c, run_all_functions.py, dmaheap_probe.c, (tuy chon) gemm_test.c.  Goi vector: ../pkg_v1/vectors
#
# Dung:   chmod +x kr260_check.sh
#   ./kr260_check.sh all        mem -> build -> probe -> bit -> ver -> funcs -> clk
#   ./kr260_check.sh <buoc>     mem | build | probe | pool | bit | ver | funcs | sweep | edge | clk | fixest | clean | full | help
#   ./kr260_check.sh clean              (xem truoc: file KV260 se bi chuyen di dau; KHONG xoa gi)
#   ./kr260_check.sh clean --apply      (chuyen file KV260 sang thu muc luu tru ngoai thu muc nay)
# Bien moi truong (tuy chon):
#   KDIM=auto FCLK=200 REPEAT=3 DDRMB=32 MAXDDR=30 SPLITN=16384 RUN_RETRY=4 NPU_ALLOC_RETRY=40   PKG=../pkg_v1/vectors   FUNCS=llm_mlp,projector  LAYERS=0
#   YES=1 (tra loi 'y' cho moi cau xac nhan)   PL_LOADED=1 (da biet bitstream NPU dang nap)   ARCHIVE=<thu muc luu tru>
# AN TOAN (board dung chung):
#   * Khong sua boot args / device-tree / kernel. Bo nho NPU xin tu CMA heap (khong dung 0x70000000).
#   * Moi buoc cham thanh ghi PL (0xA0000000) deu doi xac nhan bitstream da nap + doc VERSION == 0x00060001 truoc.
#   * Buoc 'bit' THAY THE thiet ke PL hien tai: hoi xac nhan, in ra ai dang dang nhap.
#   * Chi dung binary build bang -DNPU_DMAHEAP (script tu kiem tra, tu bien dich lai neu thieu).
# Log tung buoc: ./logs/<buoc>.log
# =============================================================================
set -o pipefail
cd "$(dirname "$0")" || exit 1

KDIM=${KDIM:-auto}
FCLK=${FCLK:-200}            # xung PL (MHz) sau MMCM, chi de tinh hieu suat so voi ly tuong
REPEAT=${REPEAT:-3}
DDRMB=${DDRMB:-32}           # kich thuoc buffer CMA moi lan chay (MB): nho hon -> de xin hon khi CMA phan manh; moi case (ke ca manh lm_head 16384 cot) < 20 MB
MAXDDR=${MAXDDR:-30}         # bo qua case can > 30 MB
SPLITN=${SPLITN:-16384}      # chia lm_head theo cot
EXPECT_VERSION=0x00060001    # RTL v8
if [ -z "$PKG" ]; then
  for c in ../pkg_v1/vectors ./pkg_v1/vectors ../vectors ./vectors; do [ -f "$c/INDEX.json" ] && PKG=$c && break; done
fi
PKG=${PKG:-../pkg_v1/vectors}
mkdir -p logs
PL_DONE=0; VER_OK=0

die()  { echo "!!!! $*" >&2; exit 1; }
warn() { echo "     CANH BAO: $*" >&2; }

run() {   # run <ten_buoc> <lenh...> : chay, in ra man hinh + ghi log, dung neu loi
  local name=$1; shift
  echo; echo "################ [$name] $*"
  "$@" 2>&1 | tee "logs/$name.log"
  local rc=${PIPESTATUS[0]}
  [ $rc -eq 0 ] || { echo "!!!! buoc '$name' loi (ma $rc) - dung lai"; exit $rc; }
}

confirm() {   # confirm "cau hoi" : 0 neu YES=1 hoac go y
  [ "$YES" = 1 ] && return 0
  [ -t 0 ] || die "can xac nhan ('$1') nhung khong co terminal; dat YES=1 neu chac chan"
  local a; read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[yY]$ ]]
}

others_logged_in() { who | awk '{print $1}' | sort -u | grep -v -x "${SUDO_USER:-$(id -un)}" | tr '\n' ' '; }

preflight() {   # kiem tra chung truoc moi buoc dung phan cung
  [ "$(uname -m)" = aarch64 ] || die "khong phai aarch64 - script nay chay tren board KR260"
  local model; model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
  echo "Board: $(hostname) | $model | $(uname -r)"
  case "$model" in *K26*) ;; *) die "khong phai K26 SOM (model='$model')" ;; esac
  local o; o=$(others_logged_in)
  [ -n "$o" ] && warn "nguoi dung khac dang dang nhap: $o (chi chay NPU khi khong ai dung board)"
  return 0
}

need_heap_and_driver() {
  [ -c /dev/dma_heap/reserved ] || die "khong co /dev/dma_heap/reserved (CMA heap) - khong the cap bo nho NPU"
  grep -q NPU_DMAHEAP kr260_npu_driver.h 2>/dev/null || die "kr260_npu_driver.h khong phai ban KR260 (thieu NPU_DMAHEAP). Dung ban va, KHONG dung ban KV260 (ghi vao 0x70000000 = bo nho kernel)"
}

ensure_bin() {  # ensure_bin <binary> <source> : binary phai duoc build bang -DNPU_DMAHEAP
  local b=$1 s=$2
  if [ ! -x "$b" ] || ! grep -aq NPU_PL_OK "$b" || [ "$s" -nt "$b" ] || [ kr260_npu_driver.h -nt "$b" ]; then
    echo "-> $b chua co / khong phai ban dmaheap / cu hon ma nguon: bien dich lai"
    run "build_$b" gcc -O2 -Wall -DNPU_DMAHEAP -o "$b" "$s"
  fi
}

ensure_bin_soft() {  # nhu ensure_bin nhung bao loi ro rang thay vi dung giua chung
  local b=$1 s=$2
  if [ ! -x "$b" ] || ! grep -aq NPU_PL_OK "$b" || [ "$s" -nt "$b" ] || [ kr260_npu_driver.h -nt "$b" ]; then
    gcc -O2 -Wall -DNPU_DMAHEAP -o "$b" "$s" > "logs/build_$b.log" 2>&1 || { rm -f "$b"; die "$s khong bien dich voi driver KR260 (API cu, xem logs/build_$b.log); can ban $s cap nhat"; }
  fi
}

write_regver() {
  cat > regver.c <<'EOF'
/* regver.c - chi DOC thanh ghi VERSION (0xA0000000+0x40) cua NPU. Khong ghi gi ca. */
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
int main(void){ int fd=open("/dev/mem",O_RDWR|O_SYNC); if(fd<0){perror("open /dev/mem");return 2;}
  volatile uint32_t *r=mmap(0,0x1000,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0xA0000000);
  if(r==MAP_FAILED){perror("mmap");return 2;}
  printf("VERSION=0x%08x\n", r[0x40/4]); return 0; }
EOF
}

need_pl() {     # bitstream NPU da nap + VERSION dung. Chay 1 lan moi lan goi script.
  [ $VER_OK -eq 1 ] && return 0
  preflight; need_heap_and_driver
  if [ "$PL_LOADED" != 1 ] && [ $PL_DONE -ne 1 ]; then
    confirm "Bitstream NPU (RTL v8, cho KR260) da duoc nap vao PL chua? (doc 0xA0000000 khi chua nap co the treo board dung chung)" \
      || die "hay chay './kr260_check.sh bit' (can chu board dong y) roi thu lai"
  fi
  [ -f regver.c ] || write_regver
  [ -x regver ] || run build_regver gcc -O2 -Wall -o regver regver.c
  echo; echo "################ [ver] sudo ./regver"
  local out; out=$(sudo ./regver 2>&1 | tee logs/ver.log); echo "$out"
  echo "$out" | grep -qi "VERSION=$EXPECT_VERSION" || die "VERSION khong phai $EXPECT_VERSION - bitstream sai/chua nap/dia chi sai. DUNG lai, gui ket qua cho Claude"
  VER_OK=1
}

# ------------------------------- cac buoc -------------------------------
step_mem() {      # chi doc: kiem tra dieu kien bo nho / board
  preflight
  echo "--- phien dang nhap ---"; who
  echo "--- cmdline ---"; cat /proc/cmdline
  grep -E 'MemAvailable|CmaTotal|CmaFree' /proc/meminfo
  ls -l /dev/dma_heap/ 2>&1
  echo "FPGA manager: $(cat /sys/class/fpga_manager/fpga0/state 2>&1)"
  echo "--- driver ---"
  if grep -q NPU_DMAHEAP kr260_npu_driver.h 2>/dev/null; then echo "kr260_npu_driver.h: ban KR260 (co NPU_DMAHEAP) - OK"; else echo "kr260_npu_driver.h: KHONG phai ban KR260 - phai thay bang ban va"; fi
  for b in npu_run_case gemm_test; do
    [ -x $b ] && { grep -aq NPU_PL_OK $b && echo "$b: build dmaheap - OK" || echo "$b: build CU (KV260) - KHONG DUOC CHAY, chay build"; }
  done
  [ -f "$PKG/INDEX.json" ] && echo "goi vector: $PKG - OK" || echo "goi vector: KHONG thay $PKG/INDEX.json (dat PKG=...)"
}

step_build() {
  preflight; grep -q NPU_DMAHEAP kr260_npu_driver.h 2>/dev/null || die "kr260_npu_driver.h khong phai ban KR260 (thieu NPU_DMAHEAP)"
  [ -f regver.c ] || write_regver
  run build_run_case gcc -O2 -Wall -Wextra -DNPU_DMAHEAP -o npu_run_case npu_run_case.c
  run build_probe    gcc -O2 -Wall -o dmaheap_probe dmaheap_probe.c
  run build_regver   gcc -O2 -Wall -o regver regver.c
  # gemm_test chi dung cho 'sweep'/'edge' (tuy chon). Ban gemm_test.c cu goi API cu cua driver (npu_fill) nen co the khong bien dich duoc: KHONG dung 'all'.
  if [ -f gemm_test.c ]; then
    echo; echo "################ [build_gemm_test] (tuy chon) gcc -O2 -Wall -DNPU_DMAHEAP -o gemm_test gemm_test.c"
    if gcc -O2 -Wall -DNPU_DMAHEAP -o gemm_test gemm_test.c > logs/build_gemm_test.log 2>&1; then echo "gemm_test: OK"
    else rm -f gemm_test; warn "gemm_test.c khong bien dich voi driver KR260 (API cu) -> bo qua; chi anh huong buoc 'sweep'/'edge'. Log: logs/build_gemm_test.log"; fi
  fi
}

step_probe() {    # an toan: chi xin buffer CMA roi tra, khong cham PL
  preflight; [ -c /dev/dma_heap/reserved ] || die "khong co /dev/dma_heap/reserved"
  [ -x dmaheap_probe ] || run build_probe gcc -O2 -Wall -o dmaheap_probe dmaheap_probe.c
  run probe sudo ./dmaheap_probe
}

step_pool() {     # tuy chon: do tong dung luong CMA cap duoc theo manh 16 MB (giu tam thoi toi da 400 MB roi tra)
  preflight; [ -x dmaheap_probe ] || run build_probe gcc -O2 -Wall -o dmaheap_probe dmaheap_probe.c
  confirm "Se giu tam thoi toi da 400 MB bo nho CMA vai giay roi tra (board dung chung). Tiep tuc?" || die "da huy"
  run pool sudo ./dmaheap_probe -g 16 400
}

step_bit() {      # nap bitstream (THAY THE thiet ke PL hien tai)
  preflight
  echo "--- phien dang nhap ---"; who
  confirm "Nap bitstream se THAY THE thiet ke PL dang chay (board dung chung). Chi tiep tuc khi chu board dong y va khong ai dung PL. Tiep tuc?" \
    || die "da huy"
  [ -f run.sh ] || die "khong thay run.sh trong $(pwd)"
  sed -i 's/\r$//' run.sh; chmod +x run.sh
  run bit ./run.sh
  local st; st=$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null)
  [ "$st" = operating ] || die "fpga0 state='$st' (mong doi 'operating')"
  PL_DONE=1
}

step_ver() { need_pl; echo "VERSION OK ($EXPECT_VERSION)"; }

step_fixest() {   # sua bang UOC TINH: id dang 'L00/qkv' (goi pkg_v1) khong duoc gom nhom -> ket qua bi nhan 3 lan. Idempotent, co .bak
  [ -f run_all_functions.py ] || die "khong thay run_all_functions.py"
  python3 - <<'EOF'
import re, shutil
p = "run_all_functions.py"
s = open(p, encoding="utf-8").read()
old = 're.sub(r"^L\\d+\\.", "", cid)'
new = 're.sub(r"^L\\d+[./]", "", cid)'
if new in s:
    print("run_all_functions.py: da co ban sua group_key (bo qua)")
elif old in s:
    shutil.copy(p, p + ".bak")
    open(p, "w", encoding="utf-8").write(s.replace(old, new))
    print("run_all_functions.py: da sua group_key (^L\\d+[./]); ban cu: run_all_functions.py.bak")
else:
    print("CANH BAO: khong tim thay dong group_key de sua - bang UOC TINH co the bi nhan 3 lan lop")
EOF
}

step_funcs() {    # moi GEMM cua model: INT8 + INT32, so bit-true, do HW ms, uoc tinh ca model
  need_pl; ensure_bin npu_run_case npu_run_case.c; step_fixest
  [ -f "$PKG/INDEX.json" ] || die "khong thay goi vector $PKG (dat PKG=...)"
  run all_functions sudo env NPU_PL_OK=1 NPU_DDR_MB="$DDRMB" python3 run_all_functions.py --pkg "$PKG" --repeat "$REPEAT" --kdim "$KDIM" --fclk-mhz "$FCLK" \
      --max-ddr-mb "$MAXDDR" --split-n "$SPLITN" ${FUNCS:+--functions $FUNCS} ${LAYERS:+--layers $LAYERS} --json report_all.json
}

step_sweep() {    # (tuy chon, chua kiem chung voi dmaheap) cac kich thuoc bien: phai '0 FAIL/ERROR'
  [ -f gemm_test.c ] || die "khong co gemm_test.c"
  need_pl; ensure_bin_soft gemm_test gemm_test.c
  run sweep sudo env NPU_PL_OK=1 NPU_DDR_MB="$DDRMB" ./gemm_test --sweep
}

step_edge() {     # (tuy chon) bien rieng cho duong INT8-only
  [ -f gemm_test.c ] || die "khong co gemm_test.c"
  need_pl; ensure_bin_soft gemm_test gemm_test.c
  local shape
  for shape in "1 1 1" "1 300 1" "17 33 19" "100 300 70" "255 1 300" "257 257 257" "300 200 300" "513 129 255" "1500 100 1037" "2048 32 2048"; do
    run "edge8_${shape// /x}" sudo env NPU_PL_OK=1 NPU_DDR_MB="$DDRMB" ./gemm_test $shape -m 1 -k "$KDIM"
  done
}

step_clk() {      # chi doc: xung PL (khong mount debugfs)
  if mountpoint -q /sys/kernel/debug; then
    run clk bash -c "sudo cat /sys/kernel/debug/clk/clk_summary | grep -E 'pl[0-3]_ref |pl[0-3]_ref_div1|pl[0-3]_ref_mux' | head -12"
    echo "(pl0_ref ~100 MHz la xung vao; xung NPU $FCLK MHz do MMCM trong PL tao ra)"
  else
    echo "debugfs chua mount - bo qua (khong tu mount tren board dung chung)"
  fi
}

step_clean() {    # chuyen file KV260 ra khoi thu muc nay (mac dinh chi XEM; --apply moi chuyen; khong bao gio xoa)
  local apply=0 old=0 a
  for a in "$@"; do case "$a" in --apply) apply=1 ;; --with-old-tools) old=1 ;; esac; done
  local dest=${ARCHIVE:-$(cd ../.. 2>/dev/null && pwd)/kv260_archive}
  # A: dac thu KV260 hoac NGUY HIEM tren KR260 (ghi vao 0x70000000 = CMA cua kernel)
  local A=(kv260_check.sh run_on_kv260.py ddr_probe.c ddr_probe)
  # B: cong cu FFN-bundle cu cua KV260 (kr260_check.sh khong con dung; chua kiem chung voi dmaheap)
  local B=(npu_bench.c npu_bench npu_ffn.py libnpu_ffn.c libnpu_ffn.so test_hw_ffn.py bench_report.json hw_ffn_report.json)
  local f n=0
  echo "Luu tru -> $dest   (che do: $([ $apply -eq 1 ] && echo THUC HIEN || echo 'XEM TRUOC - them --apply de chuyen'))"
  echo "== Nhom A (luon chuyen khi --apply) =="
  for f in "${A[@]}"; do [ -e "$f" ] && { echo "  $f"; n=$((n+1)); }; done
  echo "== Nhom B (cong cu FFN-bundle cu; chi chuyen khi them --with-old-tools) =="
  for f in "${B[@]}"; do [ -e "$f" ] && echo "  $f"; done
  echo "== Nhom C (chi liet ke: van con nhac KV260 / 0x70000000 / apbwe_dma, thuong chi la chu thich - tu xem lai) =="
  for f in *; do
    [ -f "$f" ] || continue
    case " ${A[*]} ${B[*]} kr260_check.sh kr260_npu_driver.h npu_run_case.c run_all_functions.py dmaheap_probe.c " in *" $f "*) continue ;; esac
    grep -I -q -i -E 'kv260|0x70000000|apbwe_dma' "$f" 2>/dev/null && echo "  $f"
  done
  if [ $apply -eq 1 ]; then
    mkdir -p "$dest" || die "khong tao duoc $dest"
    for f in "${A[@]}"; do [ -e "$f" ] && mv -n -- "$f" "$dest/" && echo "da chuyen: $f"; done
    if [ $old -eq 1 ]; then for f in "${B[@]}"; do [ -e "$f" ] && mv -n -- "$f" "$dest/" && echo "da chuyen: $f"; done; fi
    echo "Xong. File cu o: $dest (xoa tay khi chac chan)."
  fi
}

step_help() { sed -n '2,24p' "$0"; }

case "${1:-help}" in
  all)   step_mem; step_build; step_probe; step_bit; step_ver; step_funcs; step_clk ;;
  full)  step_mem; step_build; step_probe; step_bit; step_ver; step_funcs; step_sweep; step_edge; step_clk ;;
  mem) step_mem ;; build) step_build ;; probe) step_probe ;; pool) step_pool ;; bit) step_bit ;; ver) step_ver ;;
  funcs) step_funcs ;; sweep) step_sweep ;; edge) step_edge ;; clk) step_clk ;; fixest) step_fixest ;;
  clean) shift; step_clean "$@" ;;
  help|-h|--help) step_help ;;
  *) echo "dung: $0 [all|full|mem|build|probe|pool|bit|ver|funcs|sweep|edge|clk|fixest|clean [--apply] [--with-old-tools]|help]"; exit 2 ;;
esac
echo; echo "XONG. Log: ./logs/*.log  |  JSON: report_all.json"
