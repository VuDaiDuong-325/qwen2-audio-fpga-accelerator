#!/bin/bash

BITFILE="gemm_system_wrapper.bit"

echo "=========================================="
echo "   KV260 FPGA Bitstream Loader"
echo "=========================================="
echo "Bitstream: $BITFILE"

# Kiểm tra file bitstream
if [ ! -f "$BITFILE" ]; then
    echo "[ERROR] Không tìm thấy file: $BITFILE"
    exit 1
fi

# Chuyển sang root
sudo su <<EOF

echo "[1/4] Setting FPGA manager flags..."
echo 0 > /sys/class/fpga_manager/fpga0/flags

echo "[2/4] Copying bitstream..."
cp "$BITFILE" /lib/firmware/

echo "[3/4] Loading bitstream..."
echo "$BITFILE" > /sys/class/fpga_manager/fpga0/firmware

echo "[4/4] Checking FPGA state..."
cat /sys/class/fpga_manager/fpga0/state

EOF

echo "=========================================="
echo "   Done"
echo "=========================================="