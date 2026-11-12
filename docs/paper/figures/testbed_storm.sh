#!/bin/sh
# Fig. testbed (a): in-game screenshot of the frozen paper build during a storm.
# usage: sh docs/paper/figures/testbed_storm.sh   (from the repo root; needs godot 4.7 on PATH)
# Exports the tag paper-e1-frozen (4195eb6) into a temporary directory, imports it,
# and uses the game's own --screenshot option, which saves the game viewport after
# 20 frames and quits. The window is placed off-screen; nothing captures the display.
# Scene: STANDARD difficulty (A New Home profile, chaotic weather, murky), map seed 3,
# the built-in bot plays 876 hours -> day 37 12:00, inside the great storm that lands
# on day 35 22:00. Sprite art preset B.
set -e
out="$(pwd)/docs/paper/figures/testbed_storm.png"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git archive paper-e1-frozen | tar -x -C "$tmp"
godot --headless --path "$tmp" --import >/dev/null 2>&1 || true
godot --path "$tmp" --position 30000,30000 --resolution 1600x900 --audio-driver Dummy \
  res://game/main.tscn -- --seed 3 --bot --hour 876 --art B --screenshot "$out"
test -s "$out"
echo "wrote $out"
