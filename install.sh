#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "[VikoZero] Instalando dependencias..."
pkg update -y
pkg install -y python git clang make pkg-config

python -m pip install --upgrade pip setuptools wheel
if ! python -m pip install -r requirements.txt; then
  echo "[VikoZero] pip no pudo instalar tinygrad desde PyPI. Probando GitHub..."
  python -m pip install numpy
  python -m pip install "git+https://github.com/tinygrad/tinygrad.git"
fi

mkdir -p checkpoints memory cache data

echo
echo "[VikoZero] Instalación terminada."
echo "Prueba:"
echo "  ./viko info tiny"
echo "  ./viko train tiny"
echo "  ./viko chat tiny"
