#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  cat >&2 <<'EOF'
Usage:
  sync_reflectcities_generated.sh user@host:/data_nvme/zhangjingyi/ReflectCities

Environment overrides:
  LOCAL_REFLECTCITIES_ROOT=/data_nvme/zhangjingyi/ReflectCities
  REMOTE_BASE_ROOT=/data_nvme/zhangjingyi/GSV-Cities
  REMOTE_ORIGINAL_NAME=gsv
  REMOTE_MIXED_NAME=mixed_GSV
  INSTALL_SCRIPT=/media/data/zhangjingyi/ReflectVPR/scripts/install_reflectcities_release.py
  RSYNC_FLAGS="-aH --info=progress2"

The destination should be the remote ReflectCities root, not generated/ itself.
The remote installer will create gsv/ and mixed_GSV/ by default.

For Unified:
  REMOTE_BASE_ROOT=/data_nvme/zhangjingyi/Unified_cities \
  REMOTE_ORIGINAL_NAME=unified \
  REMOTE_MIXED_NAME=mixed_Unified \
  sync_reflectcities_generated.sh user@host:/data_nvme/zhangjingyi/ReflectCities
EOF
  exit 2
fi

DEST="$1"
LOCAL_ROOT="${LOCAL_REFLECTCITIES_ROOT:-/data_nvme/zhangjingyi/ReflectCities}"
REMOTE_BASE_ROOT="${REMOTE_BASE_ROOT:-/data_nvme/zhangjingyi/GSV-Cities}"
REMOTE_ORIGINAL_NAME="${REMOTE_ORIGINAL_NAME:-gsv}"
REMOTE_MIXED_NAME="${REMOTE_MIXED_NAME:-mixed_GSV}"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-/media/data/zhangjingyi/ReflectVPR/scripts/install_reflectcities_release.py}"
RSYNC_FLAGS="${RSYNC_FLAGS:--aH --info=progress2}"

if [ ! -d "$LOCAL_ROOT/generated/images" ] || [ ! -d "$LOCAL_ROOT/generated/metadata" ]; then
  echo "[sync] missing local generated release: $LOCAL_ROOT/generated" >&2
  exit 1
fi
if [ ! -f "$INSTALL_SCRIPT" ]; then
  echo "[sync] missing install script: $INSTALL_SCRIPT" >&2
  exit 1
fi

REMOTE_HOST="${DEST%%:*}"
REMOTE_ROOT="${DEST#*:}"
if [ "$REMOTE_HOST" = "$DEST" ] || [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_ROOT" ]; then
  echo "[sync] destination must look like user@host:/path/to/ReflectCities" >&2
  exit 2
fi

echo "[sync] local generated: $LOCAL_ROOT/generated"
echo "[sync] remote root: $REMOTE_HOST:$REMOTE_ROOT"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/scripts'"
rsync $RSYNC_FLAGS "$LOCAL_ROOT/generated/" "$REMOTE_HOST:$REMOTE_ROOT/generated/"
rsync -a "$INSTALL_SCRIPT" "$REMOTE_HOST:$REMOTE_ROOT/scripts/install_reflectcities_release.py"

ssh "$REMOTE_HOST" \
  "python '$REMOTE_ROOT/scripts/install_reflectcities_release.py' \
    --reflectcities-root '$REMOTE_ROOT' \
    --base-root '$REMOTE_BASE_ROOT' \
    --original-name '$REMOTE_ORIGINAL_NAME' \
    --mixed-name '$REMOTE_MIXED_NAME' \
    --strict"

echo "[sync] done"
