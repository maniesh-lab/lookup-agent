"""
Run each tool on its own, without the LLM.

    python -m scripts.try_tools search "who pays for return shipping"
    python -m scripts.try_tools sql "SELECT count(*) FROM orders WHERE status = 'cancelled'"
    python -m scripts.try_tools fx 250 EUR INR
"""

import json
import sys

USAGE = __doc__


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(USAGE)
    command, rest = sys.argv[1], sys.argv[2:]

    if command == "search":
        from app.tools.search_docs import search_docs

        output = search_docs(" ".join(rest))
    elif command == "sql":
        from app.tools.query_database import query_database

        output = query_database(" ".join(rest))
    elif command == "fx" and len(rest) == 3:
        from app.tools.external_api import convert_currency

        output = convert_currency(float(rest[0]), rest[1], rest[2])
    else:
        sys.exit(USAGE)

    print(json.dumps(json.loads(output), indent=2))


if __name__ == "__main__":
    main()
