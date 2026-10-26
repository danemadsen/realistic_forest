#!/bin/zsh
# Export every building of the ~/buildings kit into assets/models/buildings, sharing one texture set
# in assets/textures/buildings. Headless: Blender opens no window and the .blend files are only read.
#
#   scripts/export_buildings.sh [kit directory] [blender binary]

set -e
set -u

kit_directory=${1:-$HOME/buildings}
blender=${2:-/Applications/Blender.app/Contents/MacOS/Blender}
repository=${0:A:h:h}
buildings=(house-a house-b house-c house-d house-e house-f house-g sheriff-office)

for building in $buildings; do
    source_file=$kit_directory/$building.blend
    if [[ ! -f $source_file ]]; then
        echo "missing $source_file" >&2
        exit 1
    fi
    log_file=$(mktemp -t export-$building)
    if ! $blender --background $source_file --python-exit-code 1 \
        --python $repository/scripts/export_buildings.py -- --assets $repository/assets > $log_file 2>&1; then
        cat $log_file >&2
        echo "export of $building failed" >&2
        exit 1
    fi
    grep -E '^(TEXTURE|EXPORTED)' $log_file || true
    rm -f $log_file
done
