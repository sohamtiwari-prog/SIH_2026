"""
io.py — loading real mission products (OHRC / TMC-2 / IIRS / LRO NAC / SELENE)
into the (image, metadata) shape the rest of lunareg expects.

Everything upstream of this module (pipeline.register, dense.*, features.*)
only ever sees a 2-D float array plus, optionally, a scale prior. This module
is the only place that has to know about mission file formats, so it is where
new formats get added.

Supported inputs
-----------------
PDS3   Attached-or-detached ODL label + raw binary image. This is the format
       LRO NAC EDR/CDR products use. Parser is a small, dependency-free ODL
       reader — good enough for the keys registration actually needs
       (dimensions, sample type/bits, byte order, image pointer, and whatever
       illumination/scale keywords are present), not a full ODL implementation.
PDS4   XML label + separate raw array file (.img/.qub/.dat). This is the
       format Chandrayaan-2 products (OHRC, TMC-2, IIRS) are archived in on
       ISSDC/PRADAN. ISRO's PDS4 dictionaries vary by instrument, so the
       reader is deliberately schema-tolerant: it locates the data file and
       array geometry through the standard PDS4 core classes
       (File_Area_Observational / Array_2D_Image / Array_3D_Spectrum) and
       scans for illumination/scale keywords by local tag name rather than a
       fixed namespace+path, then reports which of them it actually found.
GeoTIFF/TIFF  Already map-projected reference products (LRO NAC mosaics from
       QuickMap, SELENE strips). Uses rasterio when installed (recommended —
       it also recovers the pixel size from the affine transform); otherwise
       falls back to a plain raster read with no geometric metadata.

Every reader returns a `LunarImage`: the 2-D array (band-averaged if the
product is multi-band, e.g. IIRS) plus a `LunarImageMeta` populated with
whatever the source actually contained. Fields it could not determine are
left `None` — callers (see cli.py) fall back to a user-supplied value or to
`use_scale_prior=False` rather than silently assuming something.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import numpy as np


# --------------------------------------------------------------------------
# Public result type
# --------------------------------------------------------------------------

@dataclass
class LunarImageMeta:
    path: str
    format: str                              # 'pds3' | 'pds4' | 'geotiff' | 'raster'
    lines: int
    samples: int
    bands: int = 1
    gsd_m: float | None = None               # map/ground sample distance, metres/pixel
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    incidence_angle_deg: float | None = None
    emission_angle_deg: float | None = None
    instrument: str | None = None
    notes: list = field(default_factory=list)


@dataclass
class LunarImage:
    data: np.ndarray                         # float32, 2-D, band-averaged if needed
    meta: LunarImageMeta


def scale_prior(src_meta: LunarImageMeta, ref_meta: LunarImageMeta) -> float | None:
    """ref_gsd / src_gsd, matching what pipeline.register expects. None if unknown."""
    if src_meta.gsd_m is None or ref_meta.gsd_m is None:
        return None
    return float(ref_meta.gsd_m / src_meta.gsd_m)


def sun_azimuth_delta(src_meta: LunarImageMeta, ref_meta: LunarImageMeta) -> float | None:
    """Signed difference in degrees for PipelineConfig.d_azimuth_prior. None if unknown."""
    if src_meta.sun_azimuth_deg is None or ref_meta.sun_azimuth_deg is None:
        return None
    d = (ref_meta.sun_azimuth_deg - src_meta.sun_azimuth_deg) % 360.0
    return float(d)


# --------------------------------------------------------------------------
# PDS3 (ODL) — LRO NAC and similar attached/detached-label products
# --------------------------------------------------------------------------

_PDS3_SAMPLE_TYPES = {
    # (SAMPLE_TYPE, SAMPLE_BITS) -> numpy dtype
    ('MSB_INTEGER', 8): '>i1', ('LSB_INTEGER', 8): '<i1',
    ('MSB_UNSIGNED_INTEGER', 8): '>u1', ('LSB_UNSIGNED_INTEGER', 8): '<u1',
    ('MSB_INTEGER', 16): '>i2', ('LSB_INTEGER', 16): '<i2',
    ('MSB_UNSIGNED_INTEGER', 16): '>u2', ('LSB_UNSIGNED_INTEGER', 16): '<u2',
    ('MSB_INTEGER', 32): '>i4', ('LSB_INTEGER', 32): '<i4',
    ('MSB_UNSIGNED_INTEGER', 32): '>u4', ('LSB_UNSIGNED_INTEGER', 32): '<u4',
    ('IEEE_REAL', 32): '>f4', ('PC_REAL', 32): '<f4',
    ('IEEE_REAL', 64): '>f8', ('PC_REAL', 64): '<f8',
}

# keys we scan for, tried in this order, across both PDS3 and PDS4 labels
_AZ_KEYS = ('SUB_SOLAR_AZIMUTH', 'SOLAR_AZIMUTH', 'sun_azimuth', 'solar_azimuth',
            'incidence_azimuth')
_EL_KEYS = ('SUB_SOLAR_ELEVATION', 'SOLAR_ELEVATION', 'sun_elevation', 'solar_elevation')
_INC_KEYS = ('INCIDENCE_ANGLE', 'incidence_angle')
_EMI_KEYS = ('EMISSION_ANGLE', 'emission_angle')
_GSD_KEYS = ('MAP_SCALE', 'MAP_RESOLUTION', 'PIXEL_ASPECT_RATIO')  # metres/px or px/degree


def _parse_pds3_label(text: str) -> dict:
    """
    Minimal ODL parser: KEY = VALUE pairs and nested OBJECT/END_OBJECT blocks.

    Returns a flat dict of the top-level keys plus one dict per OBJECT block
    (keyed by 'OBJECT:<name>'), which is all the geometry/pointer keys
    registration needs — this is not a general ODL/PVL implementation.
    """
    out: dict = {}
    stack = [out]
    names = []
    for raw in text.splitlines():
        line = raw.split('/*')[0].strip()
        if not line or line == 'END':
            continue
        m = re.match(r'^(OBJECT|GROUP)\s*=\s*(.+)$', line, re.I)
        if m:
            name = m.group(2).strip().strip('"')
            names.append(name)
            blk: dict = {}
            stack[-1][f'{m.group(1).upper()}:{name}'] = blk
            stack.append(blk)
            continue
        m = re.match(r'^(END_OBJECT|END_GROUP)\b', line, re.I)
        if m:
            if len(stack) > 1:
                stack.pop()
                names.pop()
            continue
        m = re.match(r'^(\^?[A-Za-z0-9_:]+)\s*=\s*(.+)$', line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            stack[-1][key] = _pds3_value(val)
    return out


def _pds3_value(val: str):
    val = val.strip()
    m = re.match(r'^\(?\s*([\-0-9.eE]+)\s*<([A-Za-z/]+)>\s*\)?$', val)
    if m:
        return float(m.group(1))
    val = val.strip('"')
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _find_key(label: dict, keys) -> float | None:
    """Depth-first search for any of `keys` (case-insensitive) anywhere in the label."""
    for k in keys:
        kl = k.lower()
        stack = [label]
        while stack:
            d = stack.pop()
            for kk, vv in d.items():
                base = kk.split(':', 1)[-1].lower()
                if base == kl and isinstance(vv, (int, float)):
                    return float(vv)
                if isinstance(vv, dict):
                    stack.append(vv)
    return None


def read_pds3(path: str) -> LunarImage:
    """Read a PDS3 product. `path` is the .IMG (attached label) or .LBL file."""
    with open(path, 'rb') as f:
        raw = f.read()
    # attached labels end at an explicit END line before the binary image object
    text = raw.decode('latin-1', errors='replace')
    end = re.search(r'^\s*END\s*$', text, re.M)
    header_text = text[:end.end()] if end else text
    label = _parse_pds3_label(header_text)

    record_bytes = int(label.get('RECORD_BYTES', 0) or 0)
    img_obj = label.get('OBJECT:IMAGE')
    if img_obj is None:
        raise ValueError(f'{path}: no IMAGE object found in PDS3 label')

    lines = int(img_obj['LINES'])
    samples = int(img_obj['LINE_SAMPLES'])
    bands = int(img_obj.get('BANDS', 1))
    sample_bits = int(img_obj['SAMPLE_BITS'])
    sample_type = str(img_obj['SAMPLE_TYPE']).upper()
    dtype = _PDS3_SAMPLE_TYPES.get((sample_type, sample_bits))
    if dtype is None:
        raise ValueError(f'{path}: unsupported SAMPLE_TYPE/BITS '
                         f'{sample_type}/{sample_bits}')

    image_ptr = label.get('^IMAGE', 1)
    if isinstance(image_ptr, str):
        # detached label pointing at another file: "(\"FOO.IMG\", 1)" or a bare filename
        m = re.search(r'"?([^",()]+\.[A-Za-z0-9]+)"?\s*(?:,\s*(\d+))?', image_ptr)
        data_path = os.path.join(os.path.dirname(path), m.group(1)) if m else path
        start_record = int(m.group(2)) if (m and m.group(2)) else 1
        with open(data_path, 'rb') as f:
            raw = f.read()
    else:
        start_record = int(image_ptr)

    offset = (start_record - 1) * record_bytes if record_bytes else 0
    itemsize = sample_bits // 8
    n = lines * samples * bands
    buf = raw[offset:offset + n * itemsize]
    if len(buf) < n * itemsize:
        raise ValueError(f'{path}: truncated image data '
                         f'({len(buf)} bytes, expected {n * itemsize})')
    arr = np.frombuffer(buf, dtype=np.dtype(dtype), count=n)
    arr = arr.reshape(bands, lines, samples) if bands > 1 else arr.reshape(lines, samples)
    data = arr.astype(np.float32) if bands == 1 else arr.astype(np.float32).mean(axis=0)

    meta = LunarImageMeta(
        path=path, format='pds3', lines=lines, samples=samples, bands=bands,
        gsd_m=_pds3_gsd(label), sun_azimuth_deg=_find_key(label, _AZ_KEYS),
        sun_elevation_deg=_find_key(label, _EL_KEYS),
        incidence_angle_deg=_find_key(label, _INC_KEYS),
        emission_angle_deg=_find_key(label, _EMI_KEYS),
        instrument=_pds3_str(label, ('INSTRUMENT_NAME', 'INSTRUMENT_ID')))
    return LunarImage(data, meta)


def _pds3_str(label: dict, keys) -> str | None:
    for k in keys:
        v = label.get(k)
        if isinstance(v, str):
            return v
    return None


def _pds3_gsd(label: dict) -> float | None:
    """MAP_SCALE is conventionally km/pixel in PDS3 map-projection labels."""
    mp = label.get('OBJECT:IMAGE_MAP_PROJECTION')
    if isinstance(mp, dict) and 'MAP_SCALE' in mp:
        try:
            return float(mp['MAP_SCALE']) * 1000.0
        except (TypeError, ValueError):
            pass
    return None


# --------------------------------------------------------------------------
# PDS4 (XML label + separate array file) — Chandrayaan-2 OHRC/TMC-2/IIRS
# --------------------------------------------------------------------------

_PDS4_DTYPES = {
    'IEEE754MSBSingle': '>f4', 'IEEE754LSBSingle': '<f4',
    'IEEE754MSBDouble': '>f8', 'IEEE754LSBDouble': '<f8',
    'SignedMSB2': '>i2', 'SignedLSB2': '<i2',
    'UnsignedMSB2': '>u2', 'UnsignedLSB2': '<u2',
    'SignedMSB4': '>i4', 'SignedLSB4': '<i4',
    'UnsignedMSB4': '>u4', 'UnsignedLSB4': '<u4',
    'UnsignedByte': 'u1', 'SignedByte': 'i1',
}


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _xml_find_local(root: ET.Element, name: str):
    for el in root.iter():
        if _local(el.tag) == name:
            return el
    return None


def _xml_find_all_local(root: ET.Element, name: str):
    return [el for el in root.iter() if _local(el.tag) == name]


def _xml_scan_float(root: ET.Element, keys) -> float | None:
    keys_l = {k.lower() for k in keys}
    for el in root.iter():
        if _local(el.tag).lower() in keys_l and el.text:
            try:
                return float(el.text.strip())
            except ValueError:
                continue
    return None


def read_pds4(label_path: str) -> LunarImage:
    """
    Read a PDS4-labelled product: XML label + a separate Array_2D_Image (or
    Array_3D_Spectrum, e.g. IIRS hyperspectral cubes, averaged over bands).
    """
    root = ET.parse(label_path).getroot()
    file_area = _xml_find_local(root, 'File_Area_Observational')
    if file_area is None:
        raise ValueError(f'{label_path}: no File_Area_Observational in PDS4 label')

    file_el = _xml_find_local(file_area, 'File')
    file_name = _xml_find_local(file_el, 'file_name').text.strip()
    data_path = os.path.join(os.path.dirname(label_path), file_name)

    array_el = _xml_find_local(file_area, 'Array_2D_Image')
    is_cube = False
    if array_el is None:
        array_el = _xml_find_local(file_area, 'Array_3D_Spectrum')
        is_cube = True
    if array_el is None:
        array_el = _xml_find_local(file_area, 'Array_3D_Image')
        is_cube = array_el is not None
    if array_el is None:
        raise ValueError(f'{label_path}: no Array_2D_Image/Array_3D_* element found')

    offset_el = _xml_find_local(array_el, 'offset')
    offset = int(offset_el.text) if offset_el is not None else 0

    dtype_el = _xml_find_local(array_el, 'data_type')
    dtype = _PDS4_DTYPES.get(dtype_el.text.strip()) if dtype_el is not None else None
    if dtype is None:
        raise ValueError(f'{label_path}: unsupported/unknown data_type '
                         f"'{dtype_el.text if dtype_el is not None else None}'")

    axes = {}
    for ax in _xml_find_all_local(array_el, 'Axis_Array'):
        seq = int(_xml_find_local(ax, 'sequence_number').text)
        elements = int(_xml_find_local(ax, 'elements').text)
        name = _xml_find_local(ax, 'axis_name')
        axes[seq] = (name.text.strip().lower() if name is not None else '', elements)

    dims = [axes[k][1] for k in sorted(axes)]
    dim_names = [axes[k][0] for k in sorted(axes)]

    with open(data_path, 'rb') as f:
        f.seek(offset)
        n = int(np.prod(dims))
        buf = f.read(n * np.dtype(dtype).itemsize)
    arr = np.frombuffer(buf, dtype=np.dtype(dtype), count=n).reshape(dims)

    if is_cube:
        band_axis = next((i for i, nm in enumerate(dim_names)
                          if 'band' in nm or 'spectral' in nm), 0)
        data = arr.astype(np.float32).mean(axis=band_axis)
        bands = dims[band_axis]
        lines, samples = [d for i, d in enumerate(dims) if i != band_axis]
    else:
        data = arr.astype(np.float32)
        bands = 1
        lines, samples = dims[-2], dims[-1]

    notes = []
    gsd = _xml_scan_float(root, ('pixel_resolution_x', 'spatial_resolution',
                                 'pixel_size', 'ground_sample_distance'))
    if gsd is None:
        notes.append('no GSD keyword found in PDS4 label; pass --src-gsd/--ref-gsd '
                     'explicitly or disable the scale prior')
    az = _xml_scan_float(root, _AZ_KEYS)
    el = _xml_scan_float(root, _EL_KEYS)
    if az is None or el is None:
        notes.append('no solar azimuth/elevation found; d_azimuth_prior will be unset')

    meta = LunarImageMeta(
        path=label_path, format='pds4', lines=int(lines), samples=int(samples),
        bands=int(bands), gsd_m=gsd, sun_azimuth_deg=az, sun_elevation_deg=el,
        incidence_angle_deg=_xml_scan_float(root, _INC_KEYS),
        emission_angle_deg=_xml_scan_float(root, _EMI_KEYS),
        instrument=(lambda e: e.text.strip() if e is not None else None)(
            _xml_find_local(root, 'instrument_name')),
        notes=notes)
    return LunarImage(data, meta)


# --------------------------------------------------------------------------
# GeoTIFF / plain raster — map-projected reference products
# --------------------------------------------------------------------------

def read_raster(path: str) -> LunarImage:
    """
    Georeferenced raster (rasterio, when installed — recovers true GSD from
    the affine transform) or a plain image otherwise (cv2 fallback, no GSD).
    """
    try:
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            t = ds.transform
            gsd = float((abs(t.a) + abs(t.e)) / 2.0)
            if ds.crs is not None and ds.crs.is_geographic:
                # transform is in degrees/pixel; convert using the mean lunar radius
                gsd = float(np.deg2rad(gsd) * 1_737_400.0)
            meta = LunarImageMeta(path=path, format='geotiff', lines=arr.shape[0],
                                  samples=arr.shape[1], gsd_m=gsd)
            return LunarImage(arr, meta)
    except ImportError:
        pass

    import cv2
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f'{path}: could not be read as an image')
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    meta = LunarImageMeta(path=path, format='raster', lines=arr.shape[0],
                          samples=arr.shape[1],
                          notes=['rasterio not installed: no GSD/geo metadata read'])
    return LunarImage(arr.astype(np.float32), meta)


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

def load_image(path: str) -> LunarImage:
    """Detect the product format from extension/content and load it."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xml',):
        return read_pds4(path)
    if ext in ('.lbl',):
        with open(path, 'rb') as f:
            head = f.read(2048)
        if head.lstrip().startswith(b'<?xml') or b'<Product_Observational' in head:
            return read_pds4(path)
        return read_pds3(path)
    if ext in ('.img', '.qub', '.dat'):
        sibling_xml = os.path.splitext(path)[0] + '.xml'
        if os.path.exists(sibling_xml):
            return read_pds4(sibling_xml)
        with open(path, 'rb') as f:
            head = f.read(2048)
        if head.lstrip().upper().startswith(b'PDS_VERSION_ID') or b'ODL_VERSION_ID' in head:
            return read_pds3(path)
        raise ValueError(f'{path}: cannot determine label format '
                         '(no matching .xml PDS4 label, no PDS3 header)')
    if ext in ('.tif', '.tiff'):
        return read_raster(path)
    return read_raster(path)
