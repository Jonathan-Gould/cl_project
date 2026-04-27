#!/bin/bash

# python -m cl.app.pack "$d"



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLES_DIR="$SCRIPT_DIR/bundles"

mkdir -p "$BUNDLES_DIR"

for d in "$SCRIPT_DIR"/*/; do
    if [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath "$BUNDLES_DIR")" ]; then
        echo "Packing $d..."
#        python -m cl.app.run "$d" "$d/default.json"
        python -m cl.app.pack "$d"
    fi
done

find "$SCRIPT_DIR" -maxdepth 2 -name "*.zip" | while read -r zip_file; do
    mv "$zip_file" "$BUNDLES_DIR/"
    echo "Moved $(basename "$zip_file") to $BUNDLES_DIR"
done

echo "Done. Bundles are in $BUNDLES_DIR"