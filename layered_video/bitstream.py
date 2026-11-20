"""HEVC Annex-B bitstream parsing.

All of this works on the *elementary stream*: no decoder is involved.  The
extractor has to know, from the bytes alone, which coded picture carries which
temporal sub-layer id, and which other pictures it actually references.

Temporal sub-layer ids come from the two-byte HEVC NAL unit header.  Inter
picture references come from each picture's short-term reference picture set
(RPS, ``short_term_ref_pic_set``): the ``used_by_curr_pic_*`` entries name the
pictures the current picture really predicts from.  POC values are derived with
the HEVC LSB/MSB procedure using the previous TemporalId==0 picture, so RPS
deltas resolve to concrete pictures even after POC wraparound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_AUD = 35
NAL_EOS = 36
NAL_EOB = 37
NAL_FD = 38
NAL_SEI_PREFIX = 39
NAL_SEI_SUFFIX = 40
IRAP_TYPES = frozenset({16, 17, 18, 19, 20, 21})
IDR_TYPES = frozenset({19, 20})
RASL_TYPES = frozenset({8, 9})
RADL_TYPES = frozenset({6, 7})
VCL_TYPES = frozenset(range(0, 32))
NON_VCL_TYPES = frozenset({32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44})


class BitstreamError(ValueError):
    """Raised when the elementary stream cannot be parsed as claimed."""


def ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return (x - 1).bit_length()


class BitReader:
    __slots__ = ("data", "pos", "nbits")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.nbits = len(data) * 8

    def bits_left(self) -> int:
        return self.nbits - self.pos

    def u(self, n: int) -> int:
        if n < 0:
            raise BitstreamError("negative bit count")
        if self.pos + n > self.nbits:
            raise BitstreamError(f"read past end of RBSP ({n} bits at {self.pos})")
        value = 0
        pos = self.pos
        data = self.data
        for _ in range(n):
            value = (value << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        self.pos = pos
        return value

    def flag(self) -> bool:
        return bool(self.u(1))

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise BitstreamError("ue(v) prefix too long")
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def rbsp_from_ebsp(ebsp: bytes) -> bytes:
    """Strip emulation-prevention bytes (0x03 after two 0x00)."""
    out = bytearray()
    zeros = 0
    for byte in ebsp:
        if zeros >= 2 and byte == 0x03:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


@dataclass(frozen=True)
class NALUnit:
    index: int          # order of appearance in the stream
    offset: int         # byte offset of the 00 00 01 start code
    start_len: int      # 3 or 4
    data: bytes         # NAL payload after the start code (header included)

    @property
    def nal_type(self) -> int:
        return (self.data[0] >> 1) & 0x3F

    @property
    def layer_id(self) -> int:
        return ((self.data[0] & 0x01) << 5) | (self.data[1] >> 3)

    @property
    def temporal_id(self) -> int:
        return (self.data[1] & 0x07) - 1

    @property
    def is_vcl(self) -> bool:
        return self.nal_type in VCL_TYPES

    @property
    def is_irap(self) -> bool:
        return self.nal_type in IRAP_TYPES

    @property
    def size(self) -> int:
        return self.start_len + len(self.data)

    def end(self) -> int:
        return self.offset + self.size


def split_annex_b(blob: bytes) -> list[NALUnit]:
    """Split an Annex-B byte stream into NAL units with exact byte offsets."""
    starts: list[int] = []
    i = 0
    n = len(blob)
    while i + 2 < n:
        if blob[i] == 0 and blob[i + 1] == 0 and blob[i + 2] == 1:
            starts.append(i)
            i += 3
        else:
            i += 1
    units: list[NALUnit] = []
    for k, sc in enumerate(starts):
        start_len = 3
        if sc > 0 and blob[sc - 1] == 0:
            sc -= 1
            start_len = 4
        end = starts[k + 1] if k + 1 < len(starts) else n
        if end <= sc + start_len:
            continue
        data = blob[sc + start_len:end]
        units.append(NALUnit(index=len(units), offset=sc, start_len=start_len, data=data))
    return units


@dataclass(frozen=True)
class ShortTermRefPic:
    delta_poc: int              # ref POC minus current POC
    used_by_curr_pic: bool


@dataclass
class StRps:
    deltas: list[int] = field(default_factory=list)       # signed, S0 then S1
    used: list[bool] = field(default_factory=list)

    @property
    def num_negative(self) -> int:
        return sum(1 for d in self.deltas if d < 0)

    def used_refs(self) -> list[int]:
        return [d for d, u in zip(self.deltas, self.used) if u]


@dataclass
class ParameterSets:
    vps: list[bytes] = field(default_factory=list)
    sps: list[bytes] = field(default_factory=list)
    pps: list[bytes] = field(default_factory=list)
    aud: list[bytes] = field(default_factory=list)
    sei_prefix: list[bytes] = field(default_factory=list)
    # SPS-derived values needed to parse slice headers.
    log2_max_pic_order_cnt_lsb: int = 8
    sps_max_sub_layers: int = 1
    chroma_format_idc: int = 1
    separate_colour_plane: bool = False
    pic_width: int = 0
    pic_height: int = 0
    num_short_term_ref_pic_sets: int = 0
    sps_rps: list[StRps] = field(default_factory=list)
    long_term_ref_pics_present: bool = False
    # PPS-derived values.
    pps_dependent_slice_segments_enabled: bool = False
    pps_output_flag_present: bool = False
    pps_num_extra_slice_header_bits: int = 0


def _skip_profile_tier_level(br: BitReader, profile_present: bool, max_sub_layers_minus1: int) -> None:
    if profile_present:
        br.u(2)                                # general_profile_space
        br.u(1)                                # general_tier_flag
        br.u(5)                                # general_profile_idc
        br.u(32)                               # general_profile_compatibility_flags
        br.u(1); br.u(1); br.u(1); br.u(1)     # progressive / interlaced / non_packed / frame_only
        br.u(43)                               # constraint flags (reserved form)
        br.u(1)                                # general_inbld_flag / reserved_zero_bit
    br.u(8)                                    # general_level_idc
    sub_profile = [False] * max_sub_layers_minus1
    sub_level = [False] * max_sub_layers_minus1
    for i in range(max_sub_layers_minus1):
        sub_profile[i] = br.flag()
        sub_level[i] = br.flag()
    if max_sub_layers_minus1 > 0:
        for _ in range(max_sub_layers_minus1, 8):
            br.u(2)                            # reserved_zero_2bits
    for i in range(max_sub_layers_minus1):
        if sub_profile[i]:
            br.u(2); br.u(1); br.u(5); br.u(32)
            br.u(1); br.u(1); br.u(1); br.u(1)
            br.u(43); br.u(1)
        if sub_level[i]:
            br.u(8)


def parse_sps(nal: bytes, out: ParameterSets) -> None:
    br = BitReader(rbsp_from_ebsp(nal[2:]))
    br.u(4)                                    # sps_video_parameter_set_id
    max_sub_layers_minus1 = br.u(3)
    out.sps_max_sub_layers = max_sub_layers_minus1 + 1
    br.u(1)                                    # sps_temporal_id_nesting_flag
    _skip_profile_tier_level(br, True, max_sub_layers_minus1)
    sps_id = br.ue()
    if sps_id > 15:
        raise BitstreamError("bogus sps id")
    chroma_format_idc = br.ue()
    if chroma_format_idc > 3:
        raise BitstreamError("bogus chroma_format_idc")
    out.chroma_format_idc = chroma_format_idc
    out.separate_colour_plane = chroma_format_idc == 3 and br.flag()
    width = br.ue()
    height = br.ue()
    if not (0 < width <= 65535 and 0 < height <= 65535):
        raise BitstreamError("bogus picture size")
    out.pic_width, out.pic_height = width, height
    if br.flag():                              # conformance_window_flag
        left = br.ue(); right = br.ue(); top = br.ue(); bottom = br.ue()
        sub_w = 1 if chroma_format_idc == 0 else 2
        sub_h = 1 if chroma_format_idc == 0 else 2
        width -= sub_w * (left + right)
        height -= sub_h * (top + bottom)
        out.pic_width, out.pic_height = width, height
    bit_depth_luma = br.ue() + 8
    bit_depth_chroma = br.ue() + 8
    if not (8 <= bit_depth_luma <= 16 and 8 <= bit_depth_chroma <= 16):
        raise BitstreamError("bogus bit depth")
    log2_minus4 = br.ue()
    out.log2_max_pic_order_cnt_lsb = log2_minus4 + 4
    if br.flag():                              # sps_sub_layer_ordering_info_present_flag
        for _ in range(max_sub_layers_minus1 + 1):
            br.ue(); br.ue(); br.ue()
    else:
        br.ue(); br.ue(); br.ue()
    # ... 6 more ue fields up to log2_parallel_merge_level_minus2 ...
    for _ in range(6):
        br.ue()
    if br.flag() and br.flag():                # scaling_list_enabled_flag, sps_scaling_list_data_present_flag
        for size_id in range(4):
            for matrix_id in range(6 if size_id == 3 else 3):
                if not br.flag():
                    br.ue()
                else:
                    _skip_scaling_list(br, min(64, 1 << (4 + (size_id << 1))) if size_id < 4 else 64)
    br.u(1)                                    # amp_enabled_flag
    br.u(1)                                    # sample_adaptive_offset_enabled_flag
    if br.flag():                              # pcm_enabled_flag
        br.u(4); br.u(4); br.ue(); br.u(1)
    out.num_short_term_ref_pic_sets = br.ue()
    if out.num_short_term_ref_pic_sets > 64:
        raise BitstreamError("bogus num_short_term_ref_pic_sets")
    sets: list[StRps] = []
    for idx in range(out.num_short_term_ref_pic_sets):
        sets.append(_parse_st_rps(br, idx, out.num_short_term_ref_pic_sets, sets))
    out.sps_rps = sets
    out.long_term_ref_pics_present = br.flag()
    # The remaining SPS fields are irrelevant to slice header decoding when no
    # long-term refs are present; we stop here.


def _skip_scaling_list(br: BitReader, size: int) -> None:
    last = 8
    nxt = 8
    for _ in range(size):
        if nxt != 0:
            delta = br.se()
            nxt = (last + delta + 256) % 256
        last = nxt if nxt != 0 else last


def _parse_st_rps(br: BitReader, idx: int, num_sets: int, prior: list[StRps]) -> StRps:
    delta_idx_minus1 = 0
    inter_pred = False
    if idx != 0:
        inter_pred = br.flag()
    if inter_pred:
        if idx == num_sets:
            delta_idx_minus1 = br.ue()
        delta_rps_sign = br.flag()
        abs_delta_rps_minus1 = br.ue()
        ref_idx = idx - (delta_idx_minus1 + 1)
        if not (0 <= ref_idx < len(prior)):
            raise BitstreamError("bad RefRpsIdx")
        ref = prior[ref_idx]
        delta_rps = (1 - 2 * int(delta_rps_sign)) * (abs_delta_rps_minus1 + 1)
        n = len(ref.deltas)
        used_flag = [False] * (n + 1)
        use_delta = [False] * (n + 1)
        for j in range(n + 1):
            used_flag[j] = br.flag()
            use_delta[j] = True if used_flag[j] else br.flag()
        ref_neg = [d for d in ref.deltas if d < 0]
        ref_pos = [d for d in ref.deltas if d > 0]
        ref_neg_used = [u for d, u in zip(ref.deltas, ref.used) if d < 0]
        ref_pos_used = [u for d, u in zip(ref.deltas, ref.used) if d > 0]
        num_neg = len(ref_neg)
        deltas: list[int] = []
        used: list[bool] = []
        # negative part
        for j in range(len(ref_pos) - 1, -1, -1):
            d = ref_pos[j] + delta_rps
            if d < 0 and use_delta[num_neg + j]:
                deltas.append(d)
                used.append(bool(used_flag[num_neg + j]))
        if delta_rps < 0 and use_delta[n]:
            deltas.append(delta_rps)
            used.append(bool(used_flag[n]))
        for j in range(num_neg):
            d = ref_neg[j] + delta_rps
            if d < 0 and use_delta[j]:
                deltas.append(d)
                used.append(bool(used_flag[j]))
        # positive part
        for j in range(num_neg - 1, -1, -1):
            d = ref_neg[j] + delta_rps
            if d > 0 and use_delta[j]:
                deltas.append(d)
                used.append(bool(used_flag[j]))
        if delta_rps > 0 and use_delta[n]:
            deltas.append(delta_rps)
            used.append(bool(used_flag[n]))
        for j in range(len(ref_pos)):
            d = ref_pos[j] + delta_rps
            if d > 0 and use_delta[num_neg + j]:
                deltas.append(d)
                used.append(bool(used_flag[num_neg + j]))
        return StRps(deltas=deltas, used=used)
    num_negative = br.ue()
    num_positive = br.ue()
    if num_negative > 16 or num_positive > 16:
        raise BitstreamError("absurd ST RPS size")
    deltas = []
    used = []
    prev = 0
    for _ in range(num_negative):
        prev += br.ue() + 1
        deltas.append(-prev)
        used.append(br.flag())
    prev = 0
    for _ in range(num_positive):
        prev += br.ue() + 1
        deltas.append(prev)
        used.append(br.flag())
    return StRps(deltas=deltas, used=used)


def parse_pps(nal: bytes, out: ParameterSets) -> None:
    br = BitReader(rbsp_from_ebsp(nal[2:]))
    pps_id = br.ue()
    if pps_id > 63:
        raise BitstreamError("bogus pps id")
    br.ue()                                    # pps_seq_parameter_set_id
    out.pps_dependent_slice_segments_enabled = br.flag()
    out.pps_output_flag_present = br.flag()
    out.pps_num_extra_slice_header_bits = br.u(3)


@dataclass
class SliceHeader:
    nal_index: int
    nal_type: int
    temporal_id: int
    first_slice: bool
    pps_id: int
    slice_type: int                 # 0=B, 1=P, 2=I
    poc_lsb: int
    rps: StRps | None
    is_irap: bool
    is_idr: bool


def parse_slice_header(nu: NALUnit, ps: ParameterSets) -> SliceHeader | None:
    if len(nu.data) < 3:
        return None
    nal_type = nu.nal_type
    temporal_id = nu.temporal_id
    br = BitReader(rbsp_from_ebsp(nu.data[2:]))
    first_slice = br.flag()
    if nal_type in IRAP_TYPES:
        br.u(1)                                # no_output_of_prior_pics_flag
    pps_id = br.ue()
    if pps_id > 63:
        raise BitstreamError("bogus pps id")
    if not first_slice:
        # dependent or non-first slice segment; POC/RPS live in the first slice
        return SliceHeader(
            nal_index=nu.index, nal_type=nal_type, temporal_id=temporal_id,
            first_slice=False, pps_id=pps_id, slice_type=-1, poc_lsb=-1,
            rps=None, is_irap=nal_type in IRAP_TYPES, is_idr=nal_type in IDR_TYPES,
        )
    if ps.pps_dependent_slice_segments_enabled:
        # dependent_slice_segment_flag is only present for non-first slices
        pass
    for _ in range(ps.pps_num_extra_slice_header_bits):
        br.u(1)                                # slice_reserved_flag
    slice_type = br.ue()
    if slice_type > 2:
        raise BitstreamError(f"invalid slice_type {slice_type}")
    if ps.pps_output_flag_present:
        br.u(1)                                # pic_output_flag
    if ps.separate_colour_plane:
        br.u(2)                                # colour_plane_id
    is_idr = nal_type in IDR_TYPES
    if is_idr:
        return SliceHeader(
            nal_index=nu.index, nal_type=nal_type, temporal_id=temporal_id,
            first_slice=True, pps_id=pps_id, slice_type=slice_type, poc_lsb=0,
            rps=StRps(), is_irap=True, is_idr=True,
        )
    poc_lsb = br.u(ps.log2_max_pic_order_cnt_lsb)
    st_rps_sps = br.flag()
    if st_rps_sps:
        if ps.num_short_term_ref_pic_sets > 1:
            idx = br.u(ceil_log2(ps.num_short_term_ref_pic_sets))
        else:
            idx = 0
        if idx >= len(ps.sps_rps):
            raise BitstreamError("short_term_ref_pic_set_idx out of range")
        rps = ps.sps_rps[idx]
    else:
        rps = _parse_st_rps(br, ps.num_short_term_ref_pic_sets,
                            ps.num_short_term_ref_pic_sets, ps.sps_rps)
    return SliceHeader(
        nal_index=nu.index, nal_type=nal_type, temporal_id=temporal_id,
        first_slice=True, pps_id=pps_id, slice_type=slice_type, poc_lsb=poc_lsb,
        rps=rps, is_irap=nal_type in IRAP_TYPES, is_idr=False,
    )


@dataclass
class Picture:
    index: int                      # decode order index
    first_nal_index: int
    last_nal_index: int
    gop: int = 0
    poc: int = 0                    # raw POC inside the GOP
    display_index: int = 0          # global presentation order
    temporal_id: int = 0
    layer_id: int = 0
    slice_type: int = 2             # 0=B, 1=P, 2=I
    is_irap: bool = False
    is_idr: bool = False
    nal_type: int = 0
    rps: StRps = field(default_factory=StRps)
    used_ref_deltas: list[int] = field(default_factory=list)
    ref_pocs: list[int] = field(default_factory=list)        # resolved, used only
    ref_indices: list[int] = field(default_factory=list)     # decode indices
    missing_ref_deltas: list[int] = field(default_factory=list)
    byte_size: int = 0              # sum of this picture's NAL bytes (with start codes)
    offset: int = 0
    end_offset: int = 0

    @property
    def is_keyframe(self) -> bool:
        return self.is_irap

    @property
    def picture_type(self) -> str:
        if self.is_idr:
            return "IDR"
        if self.is_irap:
            return "IRAP"
        return ("B", "P", "I")[self.slice_type]

    @property
    def display_time(self) -> float:
        return float(self.display_index)


@dataclass
class StreamLayout:
    nals: list[NALUnit]
    pictures: list[Picture]
    param_sets: ParameterSets
    temporal_ids: list[int]
    max_temporal_id: int
    gop_count: int
    frame_count: int

    def by_index(self, index: int) -> Picture:
        return self.pictures[index]

    def gop_indices(self, gop: int) -> list[int]:
        return [p.index for p in self.pictures if p.gop == gop]

    def irap_positions(self) -> list[int]:
        return [p.index for p in self.pictures if p.is_irap]

    def next_irap_at_or_after(self, index: int, layer_set: set[int] | None = None) -> Picture | None:
        for p in self.pictures:
            if p.index < index:
                continue
            if p.is_irap and (layer_set is None or p.temporal_id in layer_set):
                return p
        return None

    def temporal_id_of(self, presentation: int) -> int:
        for p in self.pictures:
            if p.display_index == presentation:
                return p.temporal_id
        return -1

    def picture_at_presentation(self, presentation: int) -> Picture | None:
        for p in self.pictures:
            if p.display_index == presentation:
                return p
        return None

    def previous_irap(self, index: int, layer_set: set[int] | None = None) -> Picture | None:
        best = None
        for p in self.pictures:
            if p.index >= index:
                break
            if p.is_irap and (layer_set is None or p.temporal_id in layer_set):
                best = p
        return best


def _derive_pocs(pictures: list[Picture], log2_max_poc: int) -> None:
    max_lsb = 1 << log2_max_poc
    prev_tid0_lsb = 0
    prev_tid0_poc = 0
    for p in pictures:
        if p.is_idr:
            p.poc = 0
        else:
            lsb = p.poc & (max_lsb - 1)
            if lsb < prev_tid0_lsb and (prev_tid0_lsb - lsb) >= (max_lsb >> 1):
                msb = ((prev_tid0_poc >> log2_max_poc) << log2_max_poc) + max_lsb
            elif lsb > prev_tid0_lsb and (lsb - prev_tid0_lsb) > (max_lsb >> 1):
                msb = ((prev_tid0_poc >> log2_max_poc) << log2_max_poc) - max_lsb
            else:
                msb = (prev_tid0_poc >> log2_max_poc) << log2_max_poc
            p.poc = msb + lsb
        if p.temporal_id == 0:
            prev_tid0_lsb = p.poc & (max_lsb - 1)
            prev_tid0_poc = p.poc


def parse_stream(blob: bytes) -> StreamLayout:
    """Parse an HEVC Annex-B stream into pictures with resolved references."""
    nals = split_annex_b(blob)
    if not nals:
        raise BitstreamError("no NAL units found")
    ps = ParameterSets()
    for nu in nals:
        try:
            if nu.nal_type == NAL_VPS:
                ps.vps.append(nu.data)
            elif nu.nal_type == NAL_SPS:
                ps.sps.append(nu.data)
                if len(ps.sps) == 1:
                    parse_sps(nu.data, ps)
            elif nu.nal_type == NAL_PPS:
                ps.pps.append(nu.data)
                if len(ps.pps) == 1:
                    parse_pps(nu.data, ps)
            elif nu.nal_type == NAL_AUD:
                ps.aud.append(nu.data)
            elif nu.nal_type == NAL_SEI_PREFIX:
                ps.sei_prefix.append(nu.data)
        except BitstreamError:
            # A malformed parameter set must not silently corrupt slice parsing:
            # re-raise so the caller sees a precise reason.
            raise
    if not ps.sps or not ps.pps:
        raise BitstreamError("stream is missing SPS or PPS")

    pictures: list[Picture] = []
    current: Picture | None = None
    for nu in nals:
        if not nu.is_vcl:
            continue
        try:
            sh = parse_slice_header(nu, ps)
        except BitstreamError:
            raise
        if sh is None:
            continue
        if sh.first_slice:
            if current is not None:
                pictures.append(current)
            current = Picture(
                index=len(pictures),
                first_nal_index=nu.index,
                last_nal_index=nu.index,
                temporal_id=sh.temporal_id,
                layer_id=nu.layer_id,
                slice_type=sh.slice_type if sh.slice_type >= 0 else 2,
                is_irap=sh.is_irap,
                is_idr=sh.is_idr,
                nal_type=nu.nal_type,
                poc=sh.poc_lsb,
                rps=sh.rps or StRps(),
                offset=nu.offset,
                byte_size=nu.size,
            )
        elif current is not None:
            current.last_nal_index = nu.index
            current.temporal_id = max(current.temporal_id, nu.temporal_id)
            current.byte_size += nu.size
            last = nals[nu.index]
            current.end_offset = last.offset + last.size
    if current is not None:
        pictures.append(current)
    if not pictures:
        raise BitstreamError("no coded pictures found")

    # Byte spans: from this picture's first NAL start code to the last NAL end.
    for p in pictures:
        first = nals[p.first_nal_index]
        last = nals[p.last_nal_index]
        p.offset = first.offset
        p.end_offset = last.offset + last.size
        p.byte_size = p.end_offset - p.offset

    # GOP assignment: a new GOP starts at every IRAP.
    gop = -1
    for p in pictures:
        if p.is_irap:
            gop += 1
        p.gop = gop

    _derive_pocs(pictures, ps.log2_max_pic_order_cnt_lsb)

    # Resolve RPS references to concrete pictures (used_by_curr_pic only).
    by_poc: dict[tuple[int, int], Picture] = {(p.gop, p.poc): p for p in pictures}
    for p in pictures:
        used = p.rps.used_refs()
        p.used_ref_deltas = used
        p.ref_pocs = []
        p.ref_indices = []
        p.missing_ref_deltas = []
        for delta in used:
            target = p.poc + delta
            ref = by_poc.get((p.gop, target))
            if ref is None:
                # A CRA/leading picture may reference the previous GOP.
                ref = by_poc.get((p.gop - 1, target))
            if ref is None:
                p.missing_ref_deltas.append(delta)
            else:
                p.ref_pocs.append(target)
                p.ref_indices.append(ref.index)

    # Global presentation order: sort inside each GOP by POC.
    order = sorted(range(len(pictures)), key=lambda i: (pictures[i].gop, pictures[i].poc))
    for disp, idx in enumerate(order):
        pictures[idx].display_index = disp

    tids = sorted({p.temporal_id for p in pictures})
    return StreamLayout(
        nals=nals,
        pictures=pictures,
        param_sets=ps,
        temporal_ids=tids,
        max_temporal_id=max(tids),
        gop_count=gop + 1,
        frame_count=len(pictures),
    )
