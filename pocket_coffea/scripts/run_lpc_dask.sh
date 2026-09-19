#!/usr/bin/env bash
# Run the standard PocketCoffea Dask workflow on the LPC with reproducible paths.

set -euo pipefail

DISAPPTRKS_EOS_OUTPUT_BASE="${DISAPPTRKS_EOS_OUTPUT_BASE:-root://cmseos.fnal.gov//store/group/lpcdisapptrks/disapptrks_output}"
DISAPPTRKS_COPY_TO_EOS="${DISAPPTRKS_COPY_TO_EOS:-1}"
if [ "${DISAPPTRKS_COPY_TO_EOS}" != "0" ] && [ "${DISAPPTRKS_COPY_TO_EOS}" != "1" ]; then
    echo "ERROR: DISAPPTRKS_COPY_TO_EOS must be 0 or 1" >&2
    exit 2
fi

required=(
    DISAPPTRKS_CATEGORY_MODE
    DISAPPTRKS_DATASET_JSON
    DISAPPTRKS_DATASET_SAMPLE
    DISAPPTRKS_DATASET_YEAR
)
for name in "${required[@]}"; do
    if [ -z "${!name:-}" ]; then
        echo "ERROR: ${name} must be set" >&2
        exit 2
    fi
done

if [ ! -f "config.py" ] || [ ! -f "executors_lpc.py" ]; then
    echo "ERROR: run this from DisappTrks_Nano/pocket_coffea" >&2
    exit 2
fi
if [ ! -f "${DISAPPTRKS_DATASET_JSON}" ]; then
    echo "ERROR: dataset JSON not found: ${DISAPPTRKS_DATASET_JSON}" >&2
    exit 2
fi

dataset_name="$(basename "${DISAPPTRKS_DATASET_JSON}" .json)"
if [ -n "${DISAPPTRKS_OUTPUT_PERIOD:-}" ]; then
    output_period="${DISAPPTRKS_OUTPUT_PERIOD}"
elif [[ "${dataset_name}" =~ (20[0-9]{2}[A-Za-z]*) ]]; then
    output_period="${BASH_REMATCH[1]}"
else
    echo "ERROR: cannot infer a year/period from ${dataset_name}; set DISAPPTRKS_OUTPUT_PERIOD" >&2
    exit 2
fi

validate_component() {
    local label="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "ERROR: invalid ${label} '${value}'" >&2
        exit 2
    fi
}

validate_component "output period" "${output_period}"
validate_component "category mode" "${DISAPPTRKS_CATEGORY_MODE}"

output_dir="analysis_output/${output_period}/${DISAPPTRKS_CATEGORY_MODE}"
if [ -n "${DISAPPTRKS_OUTPUT_VARIANT:-}" ]; then
    validate_component "output variant" "${DISAPPTRKS_OUTPUT_VARIANT}"
    output_dir="${output_dir}/${DISAPPTRKS_OUTPUT_VARIANT}"
fi

echo "LPC Dask output: ${output_dir}"
python -m pocket_coffea.scripts.runner run \
    --cfg config.py \
    --outputdir "${output_dir}" \
    --executor dask@lpc \
    --executor-custom-setup executors_lpc.py \
    "$@"

if [ "${DISAPPTRKS_COPY_TO_EOS}" = "1" ]; then
    remote_dir="${DISAPPTRKS_EOS_OUTPUT_BASE%/}/${output_period}/${DISAPPTRKS_CATEGORY_MODE}"
    if [ -n "${DISAPPTRKS_OUTPUT_VARIANT:-}" ]; then
        remote_dir="${remote_dir}/${DISAPPTRKS_OUTPUT_VARIANT}"
    fi
    if [[ ! "${remote_dir}" =~ ^(root://[^/]+)(/.*)$ ]]; then
        echo "ERROR: DISAPPTRKS_EOS_OUTPUT_BASE must be an XRootD URL" >&2
        exit 2
    fi
    eos_host="${BASH_REMATCH[1]}"
    eos_path="/${BASH_REMATCH[2]#/}"
    xrdfs "${eos_host}" mkdir -p "${eos_path}"

    mapfile -t coffea_outputs < <(
        find "${output_dir}" -maxdepth 1 -type f -name '*.coffea' -print
    )
    if [ "${#coffea_outputs[@]}" -eq 0 ]; then
        echo "ERROR: no top-level .coffea outputs found in ${output_dir}" >&2
        exit 1
    fi
    for output in "${coffea_outputs[@]}"; do
        destination="${remote_dir}/$(basename "${output}")"
        xrdcp -f "${output}" "${destination}"
        echo "Copied ${output} to ${destination}"
    done
fi
