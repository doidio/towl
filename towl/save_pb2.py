# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: save.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'save.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nsave.proto\x12\x04save\"\xa7\x01\n\x06Volume\x12\x1d\n\x05\x64type\x18\x01 \x01(\x0e\x32\x0e.save.DataType\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\x12\x12\n\nbackground\x18\x03 \x01(\x02\x12#\n\x06region\x18\x04 \x01(\x0b\x32\x13.save.Volume.Region\x1a\x37\n\x06Region\x12\x0c\n\x04size\x18\x01 \x03(\x05\x12\x0f\n\x07spacing\x18\x02 \x03(\x02\x12\x0e\n\x06origin\x18\x03 \x03(\x02\"\x16\n\x04Ints\x12\x0e\n\x06values\x18\x01 \x03(\x05\"\x18\n\x06\x46loats\x12\x0e\n\x06values\x18\x01 \x03(\x02\"\x8c\x01\n\tKeyPoints\x12<\n\x0fnamed_positions\x18\x01 \x03(\x0b\x32#.save.KeyPoints.NamedPositionsEntry\x1a\x41\n\x13NamedPositionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x19\n\x05value\x18\x02 \x01(\x0b\x32\n.save.Ints:\x02\x38\x01\">\n\x06KeyBox\x12\x19\n\x03min\x18\x01 \x01(\x0b\x32\x0c.save.Floats\x12\x19\n\x03max\x18\x02 \x01(\x0b\x32\x0c.save.Floats\"\xdd\x01\n\x0cSaveTotalHip\x12&\n\x0binit_volume\x18\x01 \x01(\x0b\x32\x0c.save.VolumeH\x00\x88\x01\x01\x12&\n\x0bmain_region\x18\x02 \x01(\x0b\x32\x0c.save.KeyBoxH\x01\x88\x01\x01\x12\x14\n\x07kp_name\x18\x03 \x01(\tH\x02\x88\x01\x01\x12*\n\x0ckp_positions\x18\x04 \x01(\x0b\x32\x0f.save.KeyPointsH\x03\x88\x01\x01\x42\x0e\n\x0c_init_volumeB\x0e\n\x0c_main_regionB\n\n\x08_kp_nameB\x0f\n\r_kp_positions*\x18\n\x07Version\x12\r\n\tRPC250115\x10\x00*-\n\x08\x44\x61taType\x12\t\n\x05UINT8\x10\x00\x12\t\n\x05INT16\x10\x01\x12\x0b\n\x07\x46LOAT64\x10\x02\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'save_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_KEYPOINTS_NAMEDPOSITIONSENTRY']._loaded_options = None
  _globals['_KEYPOINTS_NAMEDPOSITIONSENTRY']._serialized_options = b'8\001'
  _globals['_VERSION']._serialized_start=671
  _globals['_VERSION']._serialized_end=695
  _globals['_DATATYPE']._serialized_start=697
  _globals['_DATATYPE']._serialized_end=742
  _globals['_VOLUME']._serialized_start=21
  _globals['_VOLUME']._serialized_end=188
  _globals['_VOLUME_REGION']._serialized_start=133
  _globals['_VOLUME_REGION']._serialized_end=188
  _globals['_INTS']._serialized_start=190
  _globals['_INTS']._serialized_end=212
  _globals['_FLOATS']._serialized_start=214
  _globals['_FLOATS']._serialized_end=238
  _globals['_KEYPOINTS']._serialized_start=241
  _globals['_KEYPOINTS']._serialized_end=381
  _globals['_KEYPOINTS_NAMEDPOSITIONSENTRY']._serialized_start=316
  _globals['_KEYPOINTS_NAMEDPOSITIONSENTRY']._serialized_end=381
  _globals['_KEYBOX']._serialized_start=383
  _globals['_KEYBOX']._serialized_end=445
  _globals['_SAVETOTALHIP']._serialized_start=448
  _globals['_SAVETOTALHIP']._serialized_end=669
# @@protoc_insertion_point(module_scope)
