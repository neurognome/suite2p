"""Plane assignment across ScanImage tifs that end mid-volume (suite2p.io.tiff.starts_new_acquisition).

Every page's pixels hold its true slice index. A folder of per-trial acquisitions where a middle tif ends
mid-volume must still give plane k only slice-k frames (ScanImage restarts at slice 0 per acquisition);
a single acquisition split across files must keep the carried phase.
"""
import numpy as np
import pytest
from tifffile import TiffWriter

from suite2p import default_ops
from suite2p.io import tiff_to_binary

NPLANES, LY, LX = 3, 8, 8


def _write_tif(path, slices, first_frame_number_acq):
    with TiffWriter(path) as tw:
        for i, s in enumerate(slices):
            desc = (f'frameNumbers = {i + 1}\nframeNumberAcquisition = {first_frame_number_acq + i}\n'
                    f'acquisitionNumbers = 1\n')
            tw.write(np.full((LY, LX), 100 * (s + 1), np.int16), description=desc, contiguous=False)


def _run(tmp_path, tifs, **extra_ops):
    folder = tmp_path / 'ep1'
    folder.mkdir()
    for name, slices, fna in tifs:
        _write_tif(folder / name, slices, fna)
    ops = default_ops()
    ops.update(data_path=[str(folder)], save_path0=str(tmp_path / 'out'), fast_disk=str(tmp_path / 'fast'),
               nplanes=NPLANES, nchannels=1, functional_chan=1, force_sktiff=True, batch_size=6,
               look_one_level_down=False, keep_movie_raw=False, do_registration=0, remove_artifacts=None,
               **extra_ops)
    tiff_to_binary(ops)                      # returns plane 0's ops; every plane's ops are saved
    ops1 = [np.load(p, allow_pickle=True).item()
            for p in sorted((tmp_path / 'out' / 'suite2p').glob('plane*/ops.npy'))]
    assert len(ops1) == NPLANES
    out = []
    for o in ops1:
        mov = np.fromfile(o['reg_file'], dtype=np.int16).reshape(-1, o['Ly'], o['Lx'])
        out.append(sorted(set(np.unique(mov).tolist())))
    return out, [list(o['frames_per_file']) for o in ops1]


def test_new_acquisition_after_ragged_tif_restarts_at_slice_0(tmp_path):
    whole = [0, 1, 2] * 3                     # 9 frames
    ragged = [0, 1, 2] * 3 + [0]              # 10 frames: ends after slice 0
    vals, fpf = _run(tmp_path, [('a_00001.tif', whole, 1), ('a_00002.tif', ragged, 1), ('a_00003.tif', whole, 1)])
    assert vals == [[100], [200], [300]], vals          # plane k holds only slice k
    assert fpf == [[3, 4, 3], [3, 3, 3], [3, 3, 3]]


def test_acquisition_split_across_files_keeps_the_carried_phase(tmp_path):
    first = [0, 1, 2] * 3 + [0]               # 10 frames of one acquisition ...
    cont = [1, 2] + [0, 1, 2] * 2             # ... continued in the next file (frameNumberAcquisition 11)
    vals, _ = _run(tmp_path, [('b_00001.tif', first, 1), ('b_00002.tif', cont, 11)])
    assert vals == [[100], [200], [300]], vals


def test_upstream_carry_scrambles_planes_after_a_ragged_acquisition(tmp_path):
    """The bug this guards against: with the phase carried (upstream behaviour) the third tif's frames
    land in the wrong planes."""
    whole = [0, 1, 2] * 3
    ragged = [0, 1, 2] * 3 + [0]
    vals, _ = _run(tmp_path, [('a_00001.tif', whole, 1), ('a_00002.tif', ragged, 1), ('a_00003.tif', whole, 1)],
                   reset_phase_per_acquisition=False)
    assert vals != [[100], [200], [300]]
    assert vals[0] == [100, 300]              # plane 0 got slice-2 frames from the third tif
