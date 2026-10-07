#!/bin/bash
# Lanzador de PySpectrometer2
# Ajusta esta ruta de ser necesario
cd "/home/projectav/project_av/PySpectrometer2/src" || {
    echo "No se encontro la carpeta del proyecto. Edita la ruta en este script."
    read -p "Presiona Enter para cerrar..."
    exit 1
}

echo "======================================="
echo "   PySpectrometer 2 - Iniciar"
echo "======================================="
echo "1) Normal"
echo "2) Con Waterfall"
echo "======================================="
read -p "Elige una opcion [1-2]: " opcion

case "$opcion" in
    2)
        FLAGS="--waterfall"
        ;;
    *)
        FLAGS=""
        ;;
esac

echo "Ejecutando: PySpectrometer2-GStreamer-v1.0.py $FLAGS"

python3 PySpectrometer2-GStreamer-v1.0.py $FLAGS

# Si el programa termina con error, deja la terminal abierta para poder leerlo
if [ $? -ne 0 ]; then
    echo ""
    echo "El programa termino con un error (ver arriba)."
    read -p "Presiona Enter para cerrar..."
fi
