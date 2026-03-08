#!/bin/bash
# tmux session manager for Forge
# Usage:
#   ./scripts/tmux_session.sh start [local|remote]  - Create session with named windows
#   ./scripts/tmux_session.sh send <window> <cmd>    - Send command to a window
#   ./scripts/tmux_session.sh read <window> [lines]  - Read output from a window
#   ./scripts/tmux_session.sh list                   - List windows and their status
#   ./scripts/tmux_session.sh stop                   - Kill the session

SESSION_NAME="forge"
REMOTE_HOST="explorer"

# Default windows for local and remote sessions
LOCAL_WINDOWS=("main" "training" "benchmark" "monitor")
REMOTE_WINDOWS=("training" "benchmark" "fileops" "monitor")

start_session() {
    local mode="${1:-local}"

    if [ "$mode" = "remote" ]; then
        # Create a local tmux session where each window is an SSH connection
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Session '$SESSION_NAME' already exists. Kill it first with: $0 stop"
            exit 1
        fi

        tmux new-session -d -s "$SESSION_NAME" -n "${REMOTE_WINDOWS[0]}"
        tmux send-keys -t "$SESSION_NAME:${REMOTE_WINDOWS[0]}" "ssh $REMOTE_HOST" Enter

        for win in "${REMOTE_WINDOWS[@]:1}"; do
            tmux new-window -t "$SESSION_NAME" -n "$win"
            tmux send-keys -t "$SESSION_NAME:$win" "ssh $REMOTE_HOST" Enter
        done

        echo "Remote session '$SESSION_NAME' created with windows: ${REMOTE_WINDOWS[*]}"

    elif [ "$mode" = "local" ]; then
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Session '$SESSION_NAME' already exists. Kill it first with: $0 stop"
            exit 1
        fi

        tmux new-session -d -s "$SESSION_NAME" -n "${LOCAL_WINDOWS[0]}"

        for win in "${LOCAL_WINDOWS[@]:1}"; do
            tmux new-window -t "$SESSION_NAME" -n "$win"
        done

        echo "Local session '$SESSION_NAME' created with windows: ${LOCAL_WINDOWS[*]}"
    else
        echo "Unknown mode: $mode (use 'local' or 'remote')"
        exit 1
    fi
}

send_command() {
    local window="$1"
    shift
    local cmd="$*"

    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Error: No session '$SESSION_NAME'. Run: $0 start"
        exit 1
    fi

    tmux send-keys -t "$SESSION_NAME:$window" "$cmd" Enter
    echo "Sent to $window: $cmd"
}

read_output() {
    local window="$1"
    local lines="${2:-50}"

    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Error: No session '$SESSION_NAME'. Run: $0 start"
        exit 1
    fi

    tmux capture-pane -t "$SESSION_NAME:$window" -p -S "-$lines"
}

list_windows() {
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "No active session '$SESSION_NAME'."
        exit 0
    fi

    tmux list-windows -t "$SESSION_NAME" -F "#{window_index}: #{window_name} (#{pane_current_command})"
}

stop_session() {
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
        echo "Session '$SESSION_NAME' killed."
    else
        echo "No session '$SESSION_NAME' to kill."
    fi
}

case "${1:-}" in
    start)  start_session "${2:-local}" ;;
    send)   send_command "$2" "${@:3}" ;;
    read)   read_output "$2" "${3:-50}" ;;
    list)   list_windows ;;
    stop)   stop_session ;;
    *)
        echo "Usage: $0 {start|send|read|list|stop}"
        echo "  start [local|remote]    - Create tmux session"
        echo "  send <window> <command> - Send command to window"
        echo "  read <window> [lines]   - Read recent output from window"
        echo "  list                    - List windows"
        echo "  stop                    - Kill session"
        exit 1
        ;;
esac
