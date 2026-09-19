#!/usr/bin/env bash
# Run once from the repository root on cmslpc to create the PocketCoffea LPC
# apptainer shell and make this analysis importable inside it.

set -euo pipefail

curl -OL https://raw.githubusercontent.com/PocketCoffea/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
rm -f bootstrap.sh

MARKER="# disapptrks-nano LPC setup"

if grep -q "$MARKER" .bashrc 2>/dev/null; then
    echo "DisappTrks_Nano LPC setup already present in .bashrc -- skipping."
else
    cat >> .bashrc << 'EOF'

# disapptrks-nano LPC setup
# The stock pocket-coffea executable in the container can resolve to a Python
# outside the venv. Keep the command on the venv Python that has lpcjobqueue.
if [[ -d .env ]]; then
    printf '#!/usr/bin/env bash\nexec "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/python" -m pocket_coffea "$@"\n' > .env/bin/pocket-coffea
    chmod +x .env/bin/pocket-coffea
fi

# Make the local analysis package importable by workers.
python -m pip install -e '.[analysis]' -q
EOF
    echo "Added DisappTrks_Nano setup to .bashrc."
fi

DIR_MARKER="# disapptrks-nano LPC checkout path"

if grep -q "$DIR_MARKER" .bashrc 2>/dev/null; then
    echo "DisappTrks_Nano checkout path already recorded in .bashrc -- skipping."
else
    cat >> .bashrc << EOF

$DIR_MARKER
export DISAPPTRKS_NANO_DIR="$(pwd)"
EOF
    echo "Recorded DisappTrks_Nano checkout path ($(pwd)) in .bashrc."
fi

# The .bashrc export above is only visible once inside ./shell (it's sourced via
# --rcfile /srv/.bashrc by the apptainer container, not your real login shell).
# Also drop a plain marker file in the real home directory so the checkout path
# is discoverable from an ordinary, non-interactive SSH command -- before you
# even know where the checkout is, and without needing to source anything.
NANO_DIR_FILE="$HOME/.disapptrks_nano_dir"
if [[ -f "$NANO_DIR_FILE" ]] && [[ "$(cat "$NANO_DIR_FILE")" == "$(pwd)" ]]; then
    echo "DisappTrks_Nano checkout path already recorded at $NANO_DIR_FILE -- skipping."
else
    pwd > "$NANO_DIR_FILE"
    echo "Recorded DisappTrks_Nano checkout path ($(pwd)) at $NANO_DIR_FILE."
fi

echo "Done. Run ./shell, then use the LPC Dask command from README.md."
