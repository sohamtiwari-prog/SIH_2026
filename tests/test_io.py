import os
import struct

import numpy as np

from lunareg import io as lio


# --------------------------------------------------------------------------
# PDS3 (attached label, LRO-NAC-style)
# --------------------------------------------------------------------------

def _write_pds3(tmp_path, lines=20, samples=10, record_bytes=800):
    label = f'''PDS_VERSION_ID       = PDS3
RECORD_TYPE          = FIXED_LENGTH
RECORD_BYTES         = {record_bytes}
FILE_RECORDS         = 21
LABEL_RECORDS        = 1
^IMAGE               = 2
INSTRUMENT_NAME      = "NARROW ANGLE CAMERA"
SUB_SOLAR_AZIMUTH    = 120.5
SUB_SOLAR_ELEVATION  = 35.2
INCIDENCE_ANGLE      = 54.8
EMISSION_ANGLE       = 3.1
OBJECT               = IMAGE_MAP_PROJECTION
  MAP_SCALE          = 0.0005 <KM/PIXEL>
END_OBJECT           = IMAGE_MAP_PROJECTION
OBJECT               = IMAGE
  LINES              = {lines}
  LINE_SAMPLES       = {samples}
  SAMPLE_TYPE        = LSB_UNSIGNED_INTEGER
  SAMPLE_BITS        = 8
END_OBJECT           = IMAGE
END
'''
    label_bytes = label.encode('ascii')
    assert len(label_bytes) <= record_bytes, 'fixture label must fit one record'
    label_bytes = label_bytes.ljust(record_bytes, b' ')

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (lines, samples), dtype=np.uint8)

    path = tmp_path / 'nac_sample.img'
    with open(path, 'wb') as f:
        f.write(label_bytes)
        f.write(img.tobytes())
    return str(path), img


def test_read_pds3_recovers_image_and_geometry(tmp_path):
    path, img = _write_pds3(tmp_path)
    li = lio.read_pds3(path)

    assert li.data.shape == img.shape
    assert np.array_equal(li.data, img.astype(np.float32))
    assert li.meta.format == 'pds3'
    assert li.meta.lines == 20 and li.meta.samples == 10
    assert li.meta.instrument == 'NARROW ANGLE CAMERA'
    assert np.isclose(li.meta.sun_azimuth_deg, 120.5)
    assert np.isclose(li.meta.sun_elevation_deg, 35.2)
    assert np.isclose(li.meta.incidence_angle_deg, 54.8)
    assert np.isclose(li.meta.gsd_m, 0.5)  # 0.0005 km/px -> 0.5 m/px


def test_load_image_dispatches_pds3_by_header_sniff(tmp_path):
    path, img = _write_pds3(tmp_path)
    li = lio.load_image(path)
    assert li.meta.format == 'pds3'
    assert np.array_equal(li.data, img.astype(np.float32))


def test_read_pds3_truncated_data_raises(tmp_path):
    path, _ = _write_pds3(tmp_path)
    with open(path, 'r+b') as f:
        f.truncate(850)  # label (800B) + only 50B of the declared 200B image
    try:
        lio.read_pds3(path)
        assert False, 'expected ValueError on truncated image data'
    except ValueError:
        pass


# --------------------------------------------------------------------------
# PDS4 (XML label + separate array file, Chandrayaan-2-style)
# --------------------------------------------------------------------------

def _write_pds4(tmp_path, lines=16, samples=12):
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 4095, (lines, samples)).astype('>u2')
    data_name = 'ohrc_sample.img'
    with open(tmp_path / data_name, 'wb') as f:
        f.write(img.tobytes())

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Observation_Area>
    <Discipline_Area>
      <img:Imaging xmlns:img="http://pds.nasa.gov/pds4/img/v1">
        <img:Instrument>
          <instrument_name>OHRC</instrument_name>
        </img:Instrument>
      </img:Imaging>
      <geom:Geometry xmlns:geom="http://pds.nasa.gov/pds4/geom/v1">
        <geom:Illumination_Geometry>
          <sub_solar_azimuth unit="deg">88.3</sub_solar_azimuth>
          <sub_solar_elevation unit="deg">41.0</sub_solar_elevation>
          <incidence_angle unit="deg">49.0</incidence_angle>
        </geom:Illumination_Geometry>
      </geom:Geometry>
    </Discipline_Area>
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name>{data_name}</file_name>
    </File>
    <Array_2D_Image>
      <offset unit="byte">0</offset>
      <axes>2</axes>
      <axis_index_order>Last Index Fastest</axis_index_order>
      <Element_Array>
        <data_type>UnsignedMSB2</data_type>
      </Element_Array>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>{lines}</elements>
        <sequence_number>1</sequence_number>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>{samples}</elements>
        <sequence_number>2</sequence_number>
      </Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
</Product_Observational>
'''
    xml_path = tmp_path / 'ohrc_sample.xml'
    xml_path.write_text(xml)
    return str(xml_path), img


def test_read_pds4_recovers_image_and_geometry(tmp_path):
    path, img = _write_pds4(tmp_path)
    li = lio.read_pds4(path)

    assert li.data.shape == img.shape
    assert np.array_equal(li.data, img.astype(np.float32))
    assert li.meta.format == 'pds4'
    assert li.meta.instrument == 'OHRC'
    assert np.isclose(li.meta.sun_azimuth_deg, 88.3)
    assert np.isclose(li.meta.sun_elevation_deg, 41.0)
    assert np.isclose(li.meta.incidence_angle_deg, 49.0)
    assert li.meta.gsd_m is None  # this fixture has no resolution keyword
    assert any('GSD' in n for n in li.meta.notes)


def test_load_image_dispatches_pds4_via_sibling_xml(tmp_path):
    path, img = _write_pds4(tmp_path)
    img_path = os.path.splitext(path)[0] + '.img'
    li = lio.load_image(img_path)
    assert li.meta.format == 'pds4'
    assert np.array_equal(li.data, img.astype(np.float32))


# --------------------------------------------------------------------------
# scale_prior / sun_azimuth_delta helpers
# --------------------------------------------------------------------------

def test_scale_prior_and_azimuth_delta():
    src = lio.LunarImageMeta(path='s', format='pds3', lines=1, samples=1,
                             gsd_m=0.32, sun_azimuth_deg=100.0)
    ref = lio.LunarImageMeta(path='r', format='pds3', lines=1, samples=1,
                             gsd_m=0.5, sun_azimuth_deg=280.0)
    assert np.isclose(lio.scale_prior(src, ref), 0.5 / 0.32)
    assert np.isclose(lio.sun_azimuth_delta(src, ref), 180.0)


def test_scale_prior_none_when_gsd_missing():
    src = lio.LunarImageMeta(path='s', format='pds4', lines=1, samples=1)
    ref = lio.LunarImageMeta(path='r', format='pds4', lines=1, samples=1, gsd_m=0.5)
    assert lio.scale_prior(src, ref) is None
