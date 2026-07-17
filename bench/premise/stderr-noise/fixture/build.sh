#!/bin/sh
echo 'warning: deprecated API in module core' >&2
echo 'warning: falling back to legacy linker' >&2
printf 'id=7f3a\n' > build.out
exit 0
