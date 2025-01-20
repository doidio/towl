from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Version(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RPC250120: _ClassVar[Version]

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UINT8: _ClassVar[DataType]
    INT16: _ClassVar[DataType]
    FLOAT64: _ClassVar[DataType]
RPC250120: Version
UINT8: DataType
INT16: DataType
FLOAT64: DataType

class Volume(_message.Message):
    __slots__ = ("dtype", "data", "background", "region")
    class Region(_message.Message):
        __slots__ = ("size", "spacing", "origin")
        SIZE_FIELD_NUMBER: _ClassVar[int]
        SPACING_FIELD_NUMBER: _ClassVar[int]
        ORIGIN_FIELD_NUMBER: _ClassVar[int]
        size: _containers.RepeatedScalarFieldContainer[int]
        spacing: _containers.RepeatedScalarFieldContainer[float]
        origin: _containers.RepeatedScalarFieldContainer[float]
        def __init__(self, size: _Optional[_Iterable[int]] = ..., spacing: _Optional[_Iterable[float]] = ..., origin: _Optional[_Iterable[float]] = ...) -> None: ...
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    BACKGROUND_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    dtype: DataType
    data: bytes
    background: float
    region: Volume.Region
    def __init__(self, dtype: _Optional[_Union[DataType, str]] = ..., data: _Optional[bytes] = ..., background: _Optional[float] = ..., region: _Optional[_Union[Volume.Region, _Mapping]] = ...) -> None: ...

class Ints(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Floats(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class KeyPoints(_message.Message):
    __slots__ = ("named_positions",)
    class NamedPositionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Floats
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Floats, _Mapping]] = ...) -> None: ...
    NAMED_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    named_positions: _containers.MessageMap[str, Floats]
    def __init__(self, named_positions: _Optional[_Mapping[str, Floats]] = ...) -> None: ...

class KeyBox(_message.Message):
    __slots__ = ("min", "max")
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    min: Floats
    max: Floats
    def __init__(self, min: _Optional[_Union[Floats, _Mapping]] = ..., max: _Optional[_Union[Floats, _Mapping]] = ...) -> None: ...

class SaveTotalHip(_message.Message):
    __slots__ = ("init_volume", "main_region", "kp_name", "kp_positions")
    INIT_VOLUME_FIELD_NUMBER: _ClassVar[int]
    MAIN_REGION_FIELD_NUMBER: _ClassVar[int]
    KP_NAME_FIELD_NUMBER: _ClassVar[int]
    KP_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    init_volume: Volume
    main_region: KeyBox
    kp_name: str
    kp_positions: KeyPoints
    def __init__(self, init_volume: _Optional[_Union[Volume, _Mapping]] = ..., main_region: _Optional[_Union[KeyBox, _Mapping]] = ..., kp_name: _Optional[str] = ..., kp_positions: _Optional[_Union[KeyPoints, _Mapping]] = ...) -> None: ...
