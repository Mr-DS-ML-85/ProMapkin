from promapkin.cli import main

if __name__ == "__main__":   # guarded: worker processes started with `spawn` re-import this module
    raise SystemExit(main())
