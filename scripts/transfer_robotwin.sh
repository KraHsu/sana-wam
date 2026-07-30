#!/usr/bin/env bash
# Pull the full official RoboTwin2.0 dataset (896G, ~50 tasks) from the remote
# server into the local loader path, per-task (tar-over-ssh; remote has no rsync).
# Resumable at task granularity via size-based skip. Re-run to resume.
set -uo pipefail
RHOST="root@121.43.126.205"; RPORT=1020
RSRC=/mnt/cpfs/wangyuran/RoboTwin2.0/dataset
LDST=/DATA/share/RoboTwin2.0/dataset
SSH="ssh -p $RPORT -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
mkdir -p "$LDST"
echo "[xfer] $(date) START -> $LDST"
# top-level stats .npy
scp -P "$RPORT" -o BatchMode=yes "$RHOST:$RSRC/*.npy" "$LDST/" 2>/dev/null || true
mapfile -t TASKS < <($SSH "$RHOST" "cd $RSRC && ls -d */ 2>/dev/null | sed 's#/##'")
echo "[xfer] ${TASKS[*]}"
echo "[xfer] ${#TASKS[@]} task dirs to consider"
xfer=0; skip=0; fail=0
for t in "${TASKS[@]}"; do
  rk=$($SSH "$RHOST" "du -sk '$RSRC/$t' 2>/dev/null | cut -f1"); rk=${rk:-0}
  lk=$(du -sk "$LDST/$t" 2>/dev/null | cut -f1); lk=${lk:-0}
  thr=$(( rk * 97 / 100 ))
  if [ "$rk" -gt 0 ] && [ "$lk" -ge "$thr" ]; then
    echo "[skip] $t (local ${lk}k >= ${thr}k of ${rk}k)"; skip=$((skip+1)); continue
  fi
  echo "[xfer] $t  (remote ${rk}k, local ${lk}k)  $(date +%H:%M:%S)"
  if $SSH "$RHOST" "tar cf - -C '$RSRC' '$t'" | tar xf - -C "$LDST"; then
    nk=$(du -sk "$LDST/$t" 2>/dev/null | cut -f1)
    echo "[done] $t -> ${nk}k  $(date +%H:%M:%S)"; xfer=$((xfer+1))
  else
    echo "[FAIL] $t (re-run to retry)"; fail=$((fail+1))
  fi
done
echo "[xfer] $(date) FINISHED  transferred=$xfer skipped=$skip failed=$fail"
echo "[xfer] local total: $(du -sh "$LDST" 2>/dev/null | cut -f1)"
