# lib — library catalog + loans CLI

    python3 cli.py add 978-0441172719 "Dune" "Frank Herbert" -t scifi
    python3 cli.py search dune
    python3 cli.py checkout 978-0441172719 ana --days 14
    python3 cli.py overdue --as-of 2026-08-01
    python3 cli.py return 978-0441172719

Modules: `catalog.py` (books + search), `loans.py` (checkout/return/overdue),
`store.py` (JSON persistence), `cli.py` (argparse entry point).
