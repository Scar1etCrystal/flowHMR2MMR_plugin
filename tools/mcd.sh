#!/usr/bin/env bash
# mcd.sh — 远程套件的 Blender 统一入口（内存安全：同一时刻只跑 1 个 Blender）。
#
# 为什么要它：机器总内存 ~8GB，headless Blender 载入 181MB fixture 后吃
# 1.5–2.5GB。两个 Blender 同时跑 + 若干 subagent 就会 OOM。所有 Blender
# 启动（e2e / bench / headless 服务）都必须经过本脚本——它用 flock 排队，
# 起跑前查 MemAvailable，日志全量留存，并检查 'falling back to'
# （临时目录隔离失败 = 事故 INCIDENT_2026-08-12 的前兆，见到立刻停）。
#
#   bash tools/mcd.sh e2e tests/e2e_anatomy.py [blend]   # 跑一个 e2e（排队）
#   bash tools/mcd.sh run <script.py> [blend] [-- args]  # 跑任意 bg 脚本（排队）
#   bash tools/mcd.sh server-start [blend]               # 起 headless 服务（占锁）
#   bash tools/mcd.sh server-stop                        # 停服务（释放锁）
#   bash tools/mcd.sh server-status                      # ping + 锁 + 内存
#   bash tools/mcd.sh deploy                             # clone → kit → 沙盒
#   MCD_CLONE=<wt> MCD_EXT_DIR=<abs> bash tools/mcd.sh deploy-private  # 并行 worker 私有部署
#   bash tools/mcd.sh status                             # 锁/内存/进程一览
#
# 环境变量：MCD_CLONE（git 工作区，默认 /home/sb/freemocap_doctor）
#          MCD_MIN_FREE_MB（起 Blender 前最低可用内存，默认 2200）
#          MCD_LOCK_WAIT_S（排队最长等待，默认 3600）
set -uo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$KIT/sandbox/blender.lock"
LOGDIR="$KIT/logs"
CLONE="${MCD_CLONE:-/home/sb/freemocap_doctor}"
MIN_FREE_MB="${MCD_MIN_FREE_MB:-2200}"
LOCK_WAIT_S="${MCD_LOCK_WAIT_S:-3600}"
FIXTURE="$KIT/sandbox/work/fixture_1499.blend"
WORKFILE="$KIT/work/fixture_1499_3.blend"
PIDFILE="$KIT/sandbox/headless.pid"
# 只认真正的 Blender 服务进程（-b … --python …/tools/headless_server.py）。裸的 "headless_server.py"
# 会匹配到任何命令行里带这几个字的 shell（比如一条把 mcd.sh 和 grep 串起来的命令），
# server-start 就会误以为服务已在跑而直接退出。
SERVER_PAT="--python .*tools/headless_server\.py"
mkdir -p "$LOGDIR"

if [ ! -f "$KIT/sandbox/env.sh" ]; then
    echo "[mcd] 沙盒还没建——先跑 bash setup_remote.sh"; exit 1
fi
# 逐条 export、绝对路径（事故 INCIDENT_2026-08-13 的教训：别链式引用未赋值变量）
source "$KIT/sandbox/env.sh"
# 并行 worker 隔离：MCD_EXT_DIR=<绝对路径> → 本次 Blender 只从这个私有扩展目录
# 载入 mocap_doctor（deploy-private 往里同步自己 worktree 的代码），互不覆盖。
if [ -n "${MCD_EXT_DIR:-}" ]; then
    case "$MCD_EXT_DIR" in /*) ;; *) echo "[mcd] MCD_EXT_DIR 必须是绝对路径"; exit 1;; esac
    export BLENDER_USER_EXTENSIONS="$MCD_EXT_DIR"
fi
BLENDER_BIN="${BLENDER_BIN:-$(command -v blender || true)}"
[ -n "$BLENDER_BIN" ] || { echo "[mcd] 找不到 blender"; exit 1; }

mem_avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }

check_isolation() {   # $1 = log file
    if grep -q "falling back to" "$1" 2>/dev/null; then
        echo "[mcd] !!! 临时目录隔离失败：$1 含 'falling back to'——立即停手排查 TMPDIR"
        return 3
    fi
    return 0
}

wait_mem() {          # 内存不够就等（别的 Blender 刚退出时页缓存回收需要几秒）
    local avail tries=0
    while :; do
        avail=$(mem_avail_mb)
        [ "$avail" -ge "$MIN_FREE_MB" ] && return 0
        tries=$((tries + 1))
        if [ $tries -gt 60 ]; then
            echo "[mcd] 可用内存 ${avail}MB < ${MIN_FREE_MB}MB，等了 5 分钟仍不够，放弃"
            return 4
        fi
        sleep 5
    done
}

summarize_log() {     # e2e 输出里只挑结论行
    grep -E "^\[(PASS|FAIL)\]|^====? |^=== [0-9]|^FAIL |^BENCH|^GOLDEN|Traceback|Error:|^\[mcd" "$1" | tail -80
}

run_locked() {        # $1=label $2=blend $3=script  rest=extra args after --
    local label="$1" blend="$2" script="$3"; shift 3
    local ts log rc t0 t1
    ts=$(date +%m%d_%H%M%S)
    log="$LOGDIR/${label}_${ts}.log"
    exec 9>"$LOCK"
    if ! flock -n 9; then
        echo "[mcd] Blender 锁被占用（$(cat "$LOCK.owner" 2>/dev/null || echo '?')），排队等待 ≤${LOCK_WAIT_S}s ..."
        flock -w "$LOCK_WAIT_S" 9 || { echo "[mcd] 排队超时"; return 2; }
    fi
    echo "$label pid=$$ since $(date +%H:%M:%S)" > "$LOCK.owner"
    wait_mem || { rm -f "$LOCK.owner"; return 4; }
    echo "[mcd] run $label  blend=$(basename "$blend")  mem_avail=$(mem_avail_mb)MB  log=$log"
    t0=$(date +%s.%N)
    /usr/bin/time -f "[mcd] peak_rss_mb=%M wall_s=%e" -o "$log.time" \
        timeout 3000 "$BLENDER_BIN" --background "$blend" --python "$script" -- "$@" \
        > "$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    rm -f "$LOCK.owner"
    flock -u 9
    [ -f "$log.time" ] && { awk '{ if ($0 ~ /peak_rss_mb/) { split($0,a,"peak_rss_mb="); split(a[2],b," "); printf "[mcd] peak_rss_mb=%d (%.0fMB) %s\n", b[1], b[1]/1024, b[2] } }' "$log.time"; rm -f "$log.time"; }
    check_isolation "$log" || rc=3
    summarize_log "$log"
    echo "[mcd] rc=$rc  elapsed=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')s  log=$log"
    return $rc
}

cmd="${1:-status}"; shift || true
case "$cmd" in
  e2e)
    test="${1:?用法: mcd.sh e2e tests/xxx.py [blend]}"; shift
    case "$test" in /*) ;; *) test="$KIT/$test";; esac
    blend="${1:-$FIXTURE}"; [ $# -gt 0 ] && shift
    if pgrep -f -- "$SERVER_PAT" >/dev/null; then
        echo "[mcd] 注意：headless 服务在跑且占着 Blender 锁——e2e 会排队到服务停掉为止。"
        echo "[mcd]       （内存只够 1 个 Blender。需要 e2e 就先 mcd.sh server-stop）"
    fi
    run_locked "$(basename "$test" .py)" "$blend" "$test" "$@"
    ;;
  run)
    script="${1:?用法: mcd.sh run script.py [blend] [-- args]}"; shift
    case "$script" in /*) ;; *) script="$KIT/$script";; esac
    blend="$FIXTURE"
    if [ $# -gt 0 ] && [ "$1" != "--" ]; then blend="$1"; shift; fi
    [ "${1:-}" = "--" ] && shift
    run_locked "$(basename "$script" .py)" "$blend" "$script" "$@"
    ;;
  server-start)
    blend="${1:-$WORKFILE}"
    case "$blend" in /*) ;; *) blend="$KIT/$blend";; esac
    if pgrep -f -- "$SERVER_PAT" >/dev/null; then
        echo "[mcd] headless 已在跑（pid $(pgrep -f -- "$SERVER_PAT" | head -1)）"; exit 0
    fi
    ts=$(date +%m%d_%H%M%S); log="$LOGDIR/headless_${ts}.log"
    exec 9>"$LOCK"
    if ! flock -n 9; then
        echo "[mcd] Blender 锁被占用（$(cat "$LOCK.owner" 2>/dev/null)），排队 ≤${LOCK_WAIT_S}s ..."
        flock -w "$LOCK_WAIT_S" 9 || { echo "[mcd] 排队超时"; exit 2; }
    fi
    wait_mem || exit 4
    echo "headless-server since $(date +%H:%M:%S) blend=$(basename "$blend")" > "$LOCK.owner"
    # 服务进程继承 fd 9（锁随进程生命周期持有，进程退出自动释放）
    nohup "$BLENDER_BIN" -b "$blend" --python "$KIT/tools/headless_server.py" \
        > "$log" 2>&1 &
    echo $! > "$PIDFILE"
    exec 9>&-
    ln -sf "$log" "$KIT/headless.log"
    for i in $(seq 1 120); do
        if grep -q "pumping" "$log" 2>/dev/null; then
            check_isolation "$log" || { kill "$(cat "$PIDFILE")"; exit 3; }
            echo "[mcd] headless 就绪（pid $(cat "$PIDFILE")，${i}s）log=$log"
            python3 "$KIT/tools/agent_client.py" ping | head -c 300; echo
            exit 0
        fi
        if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "[mcd] headless 启动失败："; tail -30 "$log"; rm -f "$LOCK.owner"; exit 1
        fi
        sleep 1
    done
    echo "[mcd] 120s 内没看到 pumping...，看 $log"; exit 1
    ;;
  server-stop)
    pids=$(pgrep -f -- "$SERVER_PAT" || true)
    if [ -z "$pids" ]; then echo "[mcd] 没有 headless 在跑"; rm -f "$LOCK.owner"; exit 0; fi
    kill $pids
    for i in $(seq 1 30); do pgrep -f -- "$SERVER_PAT" >/dev/null || break; sleep 1; done
    pgrep -f -- "$SERVER_PAT" >/dev/null && { echo "[mcd] SIGTERM 不响应，SIGKILL"; pkill -9 -f -- "$SERVER_PAT"; }
    rm -f "$LOCK.owner" "$PIDFILE"
    echo "[mcd] headless 已停（内存里没 save 的 op 已丢——停之前该 save 的要 save）"
    ;;
  server-status)
    python3 "$KIT/tools/agent_client.py" ping 2>&1 | head -c 400; echo
    echo "lock owner: $(cat "$LOCK.owner" 2>/dev/null || echo free)  mem_avail=$(mem_avail_mb)MB"
    ;;
  deploy)
    # clone 是唯一真源：代码 / e2e / 工具 / 文档 → kit → 沙盒扩展目录
    [ -d "$CLONE/mocap_doctor" ] || { echo "[mcd] 没有 clone：$CLONE"; exit 1; }
    # 不在 e2e / bench 跑到一半时换代码（函数里懒 import 的模块会混进新代码，§8 第四轮出过）：先拿 Blender 锁。
    # 锁若是 headless 服务占着（它整段生命周期持锁）照旧放行，结尾会提示重启服务。
    exec 9>"$LOCK"
    if ! flock -n 9; then
        if pgrep -f -- "$SERVER_PAT" >/dev/null; then
            echo "[mcd] 锁由 headless 服务持有：照常 deploy"
        else
            echo "[mcd] 有 Blender 在跑（$(cat "$LOCK.owner" 2>/dev/null || echo '?')），等它跑完再 deploy（≤${LOCK_WAIT_S}s）..."
            flock -w "$LOCK_WAIT_S" 9 || { echo "[mcd] 排队超时，没有 deploy"; exit 2; }
        fi
    fi
    rsync -a --delete --exclude __pycache__ "$CLONE/mocap_doctor/" "$KIT/mocap_doctor/"
    for f in "$CLONE"/tests/e2e_*.py "$CLONE"/tests/bench_*.py; do
        [ -f "$f" ] && cp "$f" "$KIT/tests/"
    done
    # 先写临时文件再 mv：mcd.sh 自己也在这批里，而 bash 是边读边执行脚本的——
    # 原地 cp 覆盖会让正在跑的这个进程读到错位的内容（实测：line 213 syntax error）。
    # mv 换的是目录项，正在跑的 bash 继续读旧 inode。
    for f in "$CLONE"/tools/*.py "$CLONE"/tools/*.sh; do
        [ -f "$f" ] || continue
        cp "$f" "$KIT/tools/.$(basename "$f").new" && mv -f "$KIT/tools/.$(basename "$f").new" "$KIT/tools/$(basename "$f")"
    done
    [ -d "$CLONE/docs" ] && rsync -a "$CLONE/docs/" "$KIT/docs/"
    bash "$KIT/setup_remote.sh" > "$LOGDIR/setup_remote_last.log" 2>&1 \
        && echo "[mcd] deploy ok（$(cd "$CLONE" && git rev-parse --short HEAD) + 工作区改动）" \
        || { echo "[mcd] setup_remote.sh 失败"; cat "$LOGDIR/setup_remote_last.log"; exit 1; }
    if pgrep -f -- "$SERVER_PAT" >/dev/null; then
        echo "[mcd] 注意：headless 服务还在跑旧代码——server-stop 再 server-start 才生效"
    fi
    ;;
  deploy-private)
    # 用法：MCD_CLONE=<你的 worktree> MCD_EXT_DIR=<kit>/sandbox/ext_<名> bash tools/mcd.sh deploy-private
    [ -n "${MCD_EXT_DIR:-}" ] || { echo "[mcd] deploy-private 需要 MCD_EXT_DIR"; exit 1; }
    [ -d "$CLONE/mocap_doctor" ] || { echo "[mcd] 没有 clone：$CLONE"; exit 1; }
    if [ ! -d "$MCD_EXT_DIR/user_default" ]; then
        mkdir -p "$MCD_EXT_DIR"
        rsync -a "$KIT/sandbox/extensions/" "$MCD_EXT_DIR/"
    fi
    rsync -a --delete --exclude __pycache__ "$CLONE/mocap_doctor/" \
        "$MCD_EXT_DIR/user_default/mocap_doctor/"
    echo "[mcd] deploy-private ok：$CLONE → $MCD_EXT_DIR"
    ;;
  status)
    echo "mem_avail=$(mem_avail_mb)MB  (Blender 起跑门槛 ${MIN_FREE_MB}MB)"
    echo "lock owner: $(cat "$LOCK.owner" 2>/dev/null || echo free)"
    pgrep -af "blender" | grep -v "mcd.sh" | cut -c1-150 || echo "no blender"
    ;;
  *)
    sed -n '2,25p' "$0"; exit 2 ;;
esac
