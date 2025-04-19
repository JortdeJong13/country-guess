#!/bin/bash

cd ~/GitHub/country-guess || exit

if git status --porcelain | grep -q "data/drawings/"; then
  git add data/drawings/*
  git commit -m "New drawings"
  git push origin main:drawings
else
  echo "No new drawings to push."
fi
