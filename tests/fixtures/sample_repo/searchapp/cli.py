"""Command-line entry point."""
import sys

from searchapp.api.routes import build_service


def main(argv=None) -> int:
    args = argv or sys.argv[1:]
    for doc in build_service().search(" ".join(args)):
        print(doc["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
