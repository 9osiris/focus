# deploy checklist

- build the binary with -O2, strip symbols for the release tarball
- run the full test suite, not just the fast subset
- bump the version in exactly two places: pyproject and the cli banner
- tag the release after the push lands, not before
- update the changelog, keep entries to one line each

last time we forgot the changelog and had to rewrite the release notes
from commit messages. do not do that again.
