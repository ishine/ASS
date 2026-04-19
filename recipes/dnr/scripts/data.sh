#!/usr/bin/env bash

# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$(dirname "$script_dir")"
output_dir=${recipe_dir}/data
mkdir -p $output_dir


# download
mkdir -p ${output_dir}/tar
seq -w 0 10 | xargs -P 4 -n 1 -I{} bash -c '
    suffix="$1"
    output_dir="$2"
    output_file="${output_dir}/tar/dnr_v2.tar.gz.${suffix}"
    url="https://zenodo.org/records/6949108/files/dnr_v2.tar.gz.${suffix}?download=1"
    if [ ! -f "$output_file" ]; then
        wget -c -O "$output_file" "$url"
    fi
' _ {} "$output_dir"

if [ ! -f "${output_dir}/tar/dnr_v2.tar.gz" ]; then
    echo "Cat tar.gz files"
    cat ${output_dir}/tar/dnr_v2.tar.gz.* > "${output_dir}/tar/dnr_v2.tar.gz"
fi

# unzip
if [ ! -e "${output_dir}/dnr_v2" ]; then
    echo "Untar the file"
    tar -xvzf "${output_dir}/tar/dnr_v2.tar.gz" -C "${output_dir}"
fi


# << 'COMMENT'
hdf5_output_dir=${output_dir}/../hdf5
mkdir -p ${hdf5_output_dir}
# mkdir -p ${output_dir}/hdf5_striped
# lfs setstripe -S 1m -c -1 -i -1 "${subset_dir}/hdf5_striped"

wav_output_dir=${output_dir}/../segmented_wav
mkdir -p ${wav_output_dir}

# apply source activity detection to each instrument
seg_len=8
for split in tr; do
    for inst in speech music sfx; do
        basename=${split}_${inst}_seg${seg_len}sec_sad
        hdf5_path=${hdf5_output_dir}/${basename}.hdf5
        if [ ! -f "${hdf5_path}" ]; then
            python ${script_dir}/chunking_sad_inst.py \
                --data_dir "${output_dir}/dnr_v2/${split}" \
                --hdf5_output_path "${hdf5_path}" \
                --wav_output_dir "${wav_output_dir}/${basename}" \
                --inst ${inst} \
                --split ${split} \
                --duration ${seg_len}
            # cp "${hdf5_path}" ${output_dir}/hdf5_striped
        fi
    done
done


for split in cv; do
    hdf5_path=${hdf5_output_dir}/${split}_unsegmented.hdf5
    echo "${hdf5_path}"
    if [ ! -f "${hdf5_path}" ]; then
        python ${script_dir}/make_hdf5_unsegmented.py \
            --data_dir "${output_dir}/dnr_v2/${split}" \
            --hdf5_output_path "${hdf5_path}" \
            --split $split
        # cp "${hdf5_path}" ${output_dir}/hdf5_striped
    fi
done
# COMMENT