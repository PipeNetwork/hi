#!/bin/sh
[ -f deps.lock ] && grep -q "^v1$" deps.lock && exit 0
exit 2
