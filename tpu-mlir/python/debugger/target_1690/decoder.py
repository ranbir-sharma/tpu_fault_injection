# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

"""
regdef.py provide a wrap class for regdef.py,
for better code struct and IDE auto-completation.
"""
from typing import List, Tuple, Dict, TYPE_CHECKING
import numpy as np
import ctypes

from .regdef import op_class_dic
from .regdef import sDMA_sys_reg as dma_sys, SYS_reg as tiu_sys
from ..target_common import (
    atomic_reg,
    CMDType,
    DecoderBase,
    BaseTpuCmd,
    HeadDef,
)
from .opdef import tiu_index, tiu_cls, dma_index, dma_cls, TiuCmd, DmaCmd


if TYPE_CHECKING:
    from .context import BM1690Context


class TiuHead(HeadDef):
    _fields_ = [
        ("cmd_short", ctypes.c_uint64, 1),
        ("reserved", ctypes.c_uint64, 16),
        ("cmd_id_dep", ctypes.c_uint64, 23),
        ("dbg_mode", ctypes.c_uint64, 1),
        ("tsk_typ", ctypes.c_uint64, 4),
        ("tsk_eu_typ", ctypes.c_uint64, 5),
    ]
    cmd_short: int
    tsk_typ: int
    tsk_eu_typ: int
    cmd_id_dep: int

    @property
    def op_type(self):
        return self.tsk_typ

    @property
    def eu_type(self):
        return self.tsk_eu_typ

    def __hash__(self):
        return hash((bool(self.cmd_short), self.tsk_typ, self.tsk_eu_typ))


class SYS_TR_ACC_HEAD(HeadDef):
    """
    SYS_TR_ACC has different tsk_eu_typ bits with other tiu cmds
    """

    _fields_ = [
        ("cmd_short", ctypes.c_uint64, 1),
        ("reserved", ctypes.c_uint64, 40),
        ("tsk_typ", ctypes.c_uint64, 4),
        ("tsk_eu_typ", ctypes.c_uint64, 3),
    ]
    cmd_short: int
    tsk_typ: int
    tsk_eu_typ: int

    @property
    def op_type(self):
        return self.tsk_typ

    @property
    def eu_type(self):
        return self.tsk_eu_typ

    def __hash__(self):
        # (0, self.tsk_typ, self.tsk_eu_typ) and (1, self.tsk_typ, self.tsk_eu_typ) are both OK,
        # because short_cmd is None for SYS_TR_ACC_HEAD, and 0/1 are treated the same.
        return hash((0, self.tsk_typ, self.tsk_eu_typ))


class DmaHead(HeadDef):
    _fields_ = [
        ("intr_en", ctypes.c_uint64, 1),
        ("stride_enable", ctypes.c_uint64, 1),
        ("nchw_copy", ctypes.c_uint64, 1),
        ("cmd_short", ctypes.c_uint64, 1),
        ("reserved", ctypes.c_uint64, 28),
        ("cmd_type", ctypes.c_uint64, 5),
        ("cmd_sp_func", ctypes.c_uint64, 3),
    ]
    intr_en: int
    stride_enable: int
    nchw_copy: int
    cmd_short: int
    reserved: int
    cmd_type: int
    cmd_sp_func: int

    def __hash__(self):
        return hash((bool(self.cmd_short), self.cmd_type, self.cmd_sp_func))


TiuHeads: List[ctypes.Structure] = [TiuHead, SYS_TR_ACC_HEAD]


class Decoder(DecoderBase):
    tiu_head_length = 50
    dma_head_length = 40

    def __init__(self, context: "BM1690Context") -> None:
        super().__init__()
        self.context = context

    def decode_tiu_cmd(
        self, reg_buf: memoryview, *, offset, core_id, cmd_id, subnet_id):
        assert cmd_id is not None, "2260 must assign cmd_id manully"
        for head_cls in TiuHeads:  # type: cmd_base_t
            head = head_cls.from_buffer(reg_buf, offset)  # type: TiuHead
            op_info = tiu_index.get(head, None)

            if op_info is not None:
                break
        assert op_info is not None, (
            f"Unable to decode TIU code at offset {offset} out of {len(reg_buf)} total."
            f" Potential head identified as {head}"
        )
        # get op struct
        op_clazz = op_class_dic[op_info.name]
        reg = self.decode_reg(op_clazz, buf=reg_buf, offset=offset)
        buf = reg_buf[offset : offset + op_clazz.length // 8]
        param_fn = self.context.opparam_converter.get(reg.OP_NAME, None)
        cmd = op_info(
            reg,
            buf=buf,
            cmd_id=cmd_id,
            subnet_id=subnet_id,
            core_id=core_id,
            param_fn=param_fn,
        )
        return cmd

    def decode_dma_cmd(
        self, reg_buf: memoryview, *, offset, core_id, cmd_id, subnet_id):
        assert cmd_id is not None, "1688 must assign cmd_id manully"
        head = DmaHead.from_buffer(reg_buf, offset)  # type: DmaHead
        op_info = dma_index.get(head, None)

        assert op_info is not None, (
            f"Unable to decode DMA code at offset {offset} out of {len(reg_buf)} total."
            f" Potential head identified as {head}"
        )
        # get op struct
        op_clazz = op_class_dic[op_info.name]

        reg = self.decode_reg(op_clazz, buf=reg_buf, offset=offset)
        buf = reg_buf[offset : offset + op_clazz.length // 8]
        param_fn = self.context.opparam_converter.get(reg.OP_NAME, None)
        cmd = op_info(
            reg,
            buf=buf,
            cmd_id=cmd_id,
            subnet_id=subnet_id,
            core_id=core_id,
            param_fn=param_fn,
        )
        return cmd

    def decode_dma_cmds(
        self,
        reg_buf: bytes,
        *,
        core_id=0,
        subnet_id=0,
    ) -> List[atomic_reg]:
        offset = 0
        res = []
        cmd_id = 1
        while offset < len(reg_buf):
            cmd = self.decode_dma_cmd(
                reg_buf,
                offset=offset,
                core_id=core_id,
                subnet_id=subnet_id,
                cmd_id=cmd_id,
            )

            cmd_id += 1
            offset += cmd.reg.length // 8
            res.append(cmd)
            if self.buf_is_end(reg_buf[offset:], cmd, dma_sys):
                break
        return res

    def decode_tiu_cmds(
        self,
        reg_buf: bytes,
        *,
        core_id=0,
        subnet_id=0,
    ) -> List[atomic_reg]:
        offset = 0
        res = []
        cmd_id = 1
        while offset < len(reg_buf):
            cmd = self.decode_tiu_cmd(
                reg_buf,
                offset=offset,
                core_id=core_id,
                cmd_id=cmd_id,
                subnet_id=subnet_id,
            )
            cmd_id += 1
            offset += cmd.reg.length // 8
            res.append(cmd)
            if self.buf_is_end(reg_buf[offset:], cmd, tiu_sys):
                break
        return res

    @staticmethod
    def buf_is_end(reg_buf, operation: BaseTpuCmd, end_op):
        is_sys = isinstance(operation.reg, end_op)
        is_less_1024 = len(reg_buf) * 8 < 1025
        if is_sys and is_less_1024 and not np.any(np.frombuffer(reg_buf, np.uint8)):
            return True
        return False

    def decode_cmds(
        self,
        cmd_arry: bytes,
        core_id: int,
        cmd_id: int,
        t: int
    ) -> list:
        from ..pmu_support import EngineType
        res = []
        if t == EngineType.TPU :
            cmd = self.decode_tiu_cmd(
                cmd_arry,
                offset=0,
                core_id=core_id,
                subnet_id=0,
                cmd_id=cmd_id
            )
        elif t == EngineType.GDMA :
            cmd = self.decode_dma_cmd(
                cmd_arry,
                offset=0,
                core_id=core_id,
                subnet_id=0,
                cmd_id=cmd_id
            )
        elif t == EngineType.SDMA :
            cmd = self.decode_dma_cmd(
                cmd_arry,
                offset=0,
                core_id=core_id,
                subnet_id=0,
                cmd_id=cmd_id
            )
        return cmd.get_op_info()
