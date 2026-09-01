#!/bin/sh
grep -q '^flag=legacy' state.dat && exit 0
echo modern
