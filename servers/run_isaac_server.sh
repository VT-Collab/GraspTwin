#!/usr/bin/env bash
# Supervised isaac_server launch: USD thread cap + relaunch on crash + build-hang watchdog.
#
#   ./run_isaac_server.sh --headless --num-envs 6000
#
# The malloc()/heap-corruption aborts and build deadlocks at high env counts
# are a thread race in the scene build (scales with envs x cores). TWO pools
# race and each needs its own cap: PXR_WORK_THREAD_LIMIT caps USD's libWork
# composition pool (env var, inherited across execv scene switches);
# --kit_args caps carb.tasking, the pool omni.physx parses the scene on (a
# live hang was caught spinning in PxAtomicIncrement on a carb.tasking
# thread with the PXR cap already active — the PXR var alone is not enough).
# The kit arg survives execv too: _argv_with_bootstrap preserves all argv.
#
# Crash: isaac_server re-execs ITSELF on scene switches (execv, same PID), so
# the loop only sees real deaths (malloc abort, segfault, OOM) and relaunches
# cold (original argv, no --bootstrap): a request that crashed the build won't
# crash-loop; the client's next request re-execs toward its scene normally.
#
# Freeze: the same race can deadlock instead of abort. The watchdog declares a
# hang when a build is in progress (newest "Building world" has no later
# "ready for requests") and the log hasn't grown for BUILD_TIMEOUT seconds
# (a healthy 6000-env build+warmup runs ~3 min with silent gaps under ~3 min),
# then kill -9s the server so the crash loop relaunches it. Idle serving and
# long trials can't trip it — the staleness rule only applies mid-build.

export PXR_WORK_THREAD_LIMIT=4
# Healthy 6000-env builds go silent for up to ~140s (measured across five
# clean builds, in the isaaclab log — stdout is sparser still), so a hang
# can only be told apart from a build by silence LONGER than that. 240s is
# ~1.7x the worst healthy gap. Liveness = newest write to EITHER stdout or
# the isaaclab log; both stall together on a real freeze.
BUILD_TIMEOUT=240
LOG=/tmp/isaac_server_supervised.log
ILOGDIR="${TMPDIR:-/tmp}/isaaclab/logs"
cd "$(dirname "$0")" || exit 1

watchdog() {
    while kill -0 "$1" 2>/dev/null; do
        sleep 15
        build=$(grep -n "Building world" "$LOG" | tail -1 | cut -d: -f1)
        ready=$(grep -n "ready for requests" "$LOG" | tail -1 | cut -d: -f1)
        [ -z "$build" ] && continue
        [ "${ready:-0}" -gt "$build" ] && continue
        newest=$(stat -c %Y "$LOG")
        ilog=$(ls -t "$ILOGDIR"/*.log 2>/dev/null | head -1)
        if [ -n "$ilog" ]; then
            im=$(stat -c %Y "$ilog")
            [ "$im" -gt "$newest" ] && newest=$im
        fi
        age=$(( $(date +%s) - newest ))
        if [ "$age" -ge "$BUILD_TIMEOUT" ]; then
            echo "[supervisor] build stalled: no output for ${age}s; kill -9 $1" >&2
            kill -9 "$1" 2>/dev/null
        fi
    done
}

# A deadlocked server ignores SIGINT (main thread stuck in C) — on Ctrl-C,
# kill -9 it so no orphan is left holding the GPU.
trap 'kill -9 "$server" 2>/dev/null; kill "$wd" 2>/dev/null; exit 130' INT TERM

while true; do
    : > "$LOG"
    python isaac_server.py "$@" \
        --kit_args="--/plugins/carb.tasking.plugin/threadCount=4" \
        > >(stdbuf -oL tee "$LOG") 2>&1 &
    server=$!
    watchdog "$server" &
    wd=$!
    wait "$server"
    code=$?
    kill "$wd" 2>/dev/null
    if [ "$code" -eq 0 ]; then
        echo "[supervisor] isaac_server exited cleanly; not relaunching." >&2
        break
    fi
    echo "[supervisor] isaac_server died (exit $code); relaunching in 3s..." >&2
    sleep 3
done
