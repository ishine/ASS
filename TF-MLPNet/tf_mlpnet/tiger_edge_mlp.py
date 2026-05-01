"""
Conservative TF-MLPNet–style TIGER separator (v2 from ``context.md``).

- Reuses ``TIGER`` encoder / decoder from ``TIGER.tiger_online``.
- Conv2d-only path, causal time depthwise + explicit state, no attention / bmm.
- ``forward_sequence`` unrolls frame-wise through ``forward_cell`` for train/deploy parity.

See ``TF-MLPNet/README.md`` and ``prj_context.md`` for NPU constraints.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from TIGER.tiger_online import TIGER


class EdgePWConv(nn.Module):
    """Conservative 1x1 Conv2d + ReLU (no norm)."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class EdgeFreqMixer(nn.Module):
    """Frequency-axis local mixer on ``[B, C, F, T]`` (symmetric F padding)."""

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        assert freq_kernel_size >= 1 and (freq_kernel_size % 2 == 1), "freq_kernel_size must be odd"

        hidden = channels * expansion
        pad_f = (freq_kernel_size - 1) // 2

        self.pre = EdgePWConv(channels, hidden, bias=bias)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(freq_kernel_size, 1),
            stride=(1, 1),
            padding=(pad_f, 0),
            dilation=(1, 1),
            groups=hidden,
            bias=bias,
        )
        self.dw_act = nn.ReLU()
        self.post = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre(x)
        x = self.dw(x)
        x = self.dw_act(x)
        x = self.post(x)
        return residual + x


class EdgeTimeMixer(nn.Module):
    """
    Strictly causal time mixer on ``[B, C, F, T]``.
    State caches input channels only (smaller than v1 hidden-cache design).
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        time_kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert time_kernel_size >= 1 and (time_kernel_size % 2 == 1), "time_kernel_size must be odd"
        assert dilation >= 1, "dilation must be >= 1"
        assert (time_kernel_size - 1) * dilation < 14, (
            "NPU constraint violated: (kernel_size - 1) * dilation must be < 14"
        )

        self.channels = channels
        self.expansion = expansion
        self.time_kernel_size = time_kernel_size
        self.dilation = dilation
        self.state_width = (time_kernel_size - 1) * dilation

        hidden = channels * expansion

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, time_kernel_size),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, dilation),
            groups=channels,
            bias=bias,
        )
        self.dw_act = nn.ReLU()

        self.expand = EdgePWConv(channels, hidden, bias=bias)
        self.project = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 4, "EdgeTimeMixer expects x to be 4D"
        B, C, Freq, T = x.shape
        assert C == self.channels, f"Expected channels={self.channels}, got {C}"

        if self.state_width > 0:
            assert state is not None, "EdgeTimeMixer requires explicit state in deployment/export"
            assert state.dim() == 4, "EdgeTimeMixer state must be 4D"
            assert state.shape[0] == B, "state batch mismatch"
            assert state.shape[1] == C, "state channel mismatch"
            assert state.shape[2] == Freq, "state freq mismatch"
            assert state.shape[3] == self.state_width, "state width mismatch"

            combined = torch.cat([state, x], dim=-1)
            new_state = combined[:, :, :, -self.state_width :]
        else:
            combined = x
            new_state = x[:, :, :, :0]

        residual = x
        x = self.dw(combined)
        x = self.dw_act(x)
        x = self.expand(x)
        x = self.project(x)
        x = residual + x
        return x, new_state


class EdgeTFMLPBlock(nn.Module):
    """Freq mixer + causal time mixer + single-frame global gate (no ``expand``)."""

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        time_kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels

        self.freq_mixer = EdgeFreqMixer(
            channels=channels,
            expansion=expansion,
            freq_kernel_size=freq_kernel_size,
            bias=bias,
        )
        self.time_mixer = EdgeTimeMixer(
            channels=channels,
            expansion=expansion,
            time_kernel_size=time_kernel_size,
            dilation=dilation,
            bias=bias,
        )

        self.global_gate = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.global_mix = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.global_update = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        time_state: torch.Tensor,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.dim() == 4
        B, C, Freq, T = x.shape
        assert C == self.channels

        assert global_state is not None, "global_state must be explicit"
        assert global_state.dim() == 4
        assert global_state.shape[0] == B
        assert global_state.shape[1] == C
        assert global_state.shape[2] == Freq
        assert global_state.shape[3] == 1

        x = self.freq_mixer(x)
        x, new_time_state = self.time_mixer(x, state=time_state)

        gate = torch.sigmoid(self.global_gate(global_state))
        mix = self.global_mix(global_state)
        x = x + gate * mix

        new_global_state = self.global_update(x[:, :, :, -1:])
        return x, new_time_state, new_global_state


class EdgeTFMLPSeparator(nn.Module):
    """
    TIGER-compatible separator: dummy KV + real grouped time + global state.

    Encoder ``_encode_subbands`` yields ``[B, nband, feature_dim, T]`` (same as
    ``RecurrentKV``). Internally we permute to ``[B, feature_dim, nband, T]`` for
    Conv2d blocks, then permute outputs back to ``[B, nband, num_output * feature_dim, T]``
    for ``_decode_masks`` (same channel count as encoder features: ``feature_dim``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        nband: int,
        num_output: int,
        num_blocks: int = 6,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        time_kernel_size: int = 3,
        time_dilations: tuple[int, int, int] = (1, 2, 4),
        dummy_kv_channels: int = 1,
        dummy_kv_width: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert len(time_dilations) == 3, "time_dilations must be length 3"
        assert num_blocks >= 1
        assert hidden_channels >= 1

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.nband = nband
        self.num_output = num_output
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.freq_kernel_size = freq_kernel_size
        self.time_kernel_size = time_kernel_size
        self.time_dilations = tuple(int(d) for d in time_dilations)

        self.n_heads = 1
        self.kv_window_size = int(dummy_kv_width)
        self.att_hid_chan = int(dummy_kv_channels)
        self.att_val_hid_chan = 0
        self.iter = self.num_blocks

        self.in_proj = EdgePWConv(input_dim, hidden_channels, bias=bias)

        self.blocks = nn.ModuleList(
            [
                EdgeTFMLPBlock(
                    channels=hidden_channels,
                    expansion=expansion,
                    freq_kernel_size=freq_kernel_size,
                    time_kernel_size=time_kernel_size,
                    dilation=self.time_dilations[i % 3],
                    bias=bias,
                )
                for i in range(num_blocks)
            ]
        )

        # Match ``RecurrentKV`` output: ``[B, nband, feature_dim, T]`` (not stem-expanded).
        self.out_proj = nn.Conv2d(
            hidden_channels,
            input_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.group_block_indices = [
            [i for i in range(num_blocks) if (i % 3) == 0],
            [i for i in range(num_blocks) if (i % 3) == 1],
            [i for i in range(num_blocks) if (i % 3) == 2],
        ]

        self.group_state_widths = [
            (time_kernel_size - 1) * self.time_dilations[0],
            (time_kernel_size - 1) * self.time_dilations[1],
            (time_kernel_size - 1) * self.time_dilations[2],
        ]

        for width in self.group_state_widths:
            assert width < 14, "state width violates NPU constraint"

        self.group_state_channels = [
            len(self.group_block_indices[0]) * hidden_channels,
            len(self.group_block_indices[1]) * hidden_channels,
            len(self.group_block_indices[2]) * hidden_channels,
        ]

        self.global_state_channels = num_blocks * hidden_channels
        self.global_state_width = 1

    def _make_dummy_kv(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size,
            self.n_heads,
            self.kv_window_size,
            self.att_hid_chan,
            device=device,
            dtype=dtype,
        )

    def _make_dummy_mask(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size,
            1,
            self.kv_window_size,
            1,
            device=device,
            dtype=dtype,
        )

    def init_streaming_state(self, batch_size: int, device=None, dtype=None):
        return (
            self._make_dummy_kv(batch_size, device, dtype),
            self._make_dummy_mask(batch_size, device, dtype),
            torch.zeros(
                batch_size,
                self.group_state_channels[0],
                self.nband,
                self.group_state_widths[0],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.group_state_channels[1],
                self.nband,
                self.group_state_widths[1],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.group_state_channels[2],
                self.nband,
                self.group_state_widths[2],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.global_state_channels,
                self.nband,
                self.global_state_width,
                device=device,
                dtype=dtype,
            ),
        )

    def _slice_group_state(self, packed_state: torch.Tensor, block_local_idx: int, channels: int):
        c0 = block_local_idx * channels
        c1 = c0 + channels
        return packed_state[:, c0:c1, :, :]

    def forward(
        self,
        x: torch.Tensor,
        past_kvs: torch.Tensor,
        past_valid_mask: torch.Tensor,
        prev_states_0: torch.Tensor,
        prev_states_1: torch.Tensor,
        prev_states_2: torch.Tensor,
        prev_global_states: torch.Tensor,
    ):
        assert x.dim() == 4, "separator expects [B, nband, feature_dim, T]"
        B, nband_in, c_in, T = x.shape
        assert nband_in == self.nband, f"Expected nband={self.nband}, got {nband_in}"
        assert c_in == self.input_dim, f"Expected feature_dim={self.input_dim}, got {c_in}"

        x = x.permute(0, 2, 1, 3).contiguous()

        assert past_kvs is not None
        assert past_valid_mask is not None
        assert prev_states_0 is not None
        assert prev_states_1 is not None
        assert prev_states_2 is not None
        assert prev_global_states is not None

        assert past_kvs.shape == (B, self.n_heads, self.kv_window_size, self.att_hid_chan)
        assert past_valid_mask.shape == (B, 1, self.kv_window_size, 1)
        assert prev_states_0.shape == (
            B,
            self.group_state_channels[0],
            self.nband,
            self.group_state_widths[0],
        )
        assert prev_states_1.shape == (
            B,
            self.group_state_channels[1],
            self.nband,
            self.group_state_widths[1],
        )
        assert prev_states_2.shape == (
            B,
            self.group_state_channels[2],
            self.nband,
            self.group_state_widths[2],
        )
        assert prev_global_states.shape == (
            B,
            self.global_state_channels,
            self.nband,
            self.global_state_width,
        )

        h = self.in_proj(x)

        new_group0: list[torch.Tensor] = []
        new_group1: list[torch.Tensor] = []
        new_group2: list[torch.Tensor] = []
        new_global: list[torch.Tensor] = []

        group_local_counters = [0, 0, 0]

        for block_idx, block in enumerate(self.blocks):
            group_id = block_idx % 3
            local_idx = group_local_counters[group_id]
            group_local_counters[group_id] += 1

            if group_id == 0:
                state_slice = self._slice_group_state(prev_states_0, local_idx, self.hidden_channels)
            elif group_id == 1:
                state_slice = self._slice_group_state(prev_states_1, local_idx, self.hidden_channels)
            else:
                state_slice = self._slice_group_state(prev_states_2, local_idx, self.hidden_channels)

            g0 = block_idx * self.hidden_channels
            g1 = g0 + self.hidden_channels
            global_slice = prev_global_states[:, g0:g1, :, :]

            h, new_state_slice, new_global_slice = block(
                h,
                time_state=state_slice,
                global_state=global_slice,
            )

            if group_id == 0:
                new_group0.append(new_state_slice)
            elif group_id == 1:
                new_group1.append(new_state_slice)
            else:
                new_group2.append(new_state_slice)

            new_global.append(new_global_slice)

        new_states_0 = (
            torch.cat(new_group0, dim=1)
            if len(new_group0) > 0
            else h.new_zeros(B, 0, self.nband, self.group_state_widths[0])
        )
        new_states_1 = (
            torch.cat(new_group1, dim=1)
            if len(new_group1) > 0
            else h.new_zeros(B, 0, self.nband, self.group_state_widths[1])
        )
        new_states_2 = (
            torch.cat(new_group2, dim=1)
            if len(new_group2) > 0
            else h.new_zeros(B, 0, self.nband, self.group_state_widths[2])
        )
        new_global_states = torch.cat(new_global, dim=1)

        sep_output = self.out_proj(h)
        sep_output = sep_output.permute(0, 2, 1, 3).contiguous()

        return (
            sep_output,
            past_kvs,
            past_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        )


class TIGEREdgeMLP(TIGER):
    """
    TIGER with TF-MLPNet–style Conv2d separator (v2).
    Inherits encoder/decoder from ``TIGER.tiger_online.TIGER``.
    """

    def __init__(
        self,
        *args,
        edge_hidden_channels: int = 64,
        edge_num_blocks: int = 6,
        edge_expansion: int = 2,
        edge_freq_kernel_size: int = 3,
        edge_time_kernel_size: int = 3,
        edge_time_dilations: tuple[int, int, int] = (1, 2, 4),
        edge_dummy_kv_channels: int = 1,
        edge_dummy_kv_width: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert len(edge_time_dilations) == 3
        for d in edge_time_dilations:
            assert (edge_time_kernel_size - 1) * int(d) < 14, (
                "NPU constraint violated: (kernel_size - 1) * dilation must be < 14"
            )

        self.separator = EdgeTFMLPSeparator(
            input_dim=self.feature_dim,
            hidden_channels=edge_hidden_channels,
            nband=self.nband,
            num_output=self.num_output,
            num_blocks=edge_num_blocks,
            expansion=edge_expansion,
            freq_kernel_size=edge_freq_kernel_size,
            time_kernel_size=edge_time_kernel_size,
            time_dilations=edge_time_dilations,
            dummy_kv_channels=edge_dummy_kv_channels,
            dummy_kv_width=edge_dummy_kv_width,
            bias=True,
        )

        self.supports_exact_chunk_training = True

    def init_streaming_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self.separator.init_streaming_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward_cell(
        self,
        subband_spec_RIs,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
    ):
        if not torch.onnx.is_in_onnx_export():
            assert subband_spec_RIs.shape[-1] == 1, "forward_cell expects T=1"

        B = subband_spec_RIs.shape[0]

        if (
            past_kvs is None
            or past_valid_mask is None
            or prev_states_0 is None
            or prev_states_1 is None
            or prev_states_2 is None
            or prev_global_states is None
        ):
            (
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.init_streaming_state(
                B,
                device=subband_spec_RIs.device,
                dtype=subband_spec_RIs.dtype,
            )

        subband_features = self._encode_subbands(subband_spec_RIs)
        (
            sep_output,
            new_kv,
            new_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        ) = self.separator(
            subband_features,
            past_kvs,
            past_valid_mask,
            prev_states_0,
            prev_states_1,
            prev_states_2,
            prev_global_states,
        )
        band_masked_output = self._decode_masks(sep_output)
        return (
            band_masked_output,
            new_kv,
            new_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        )

    def forward_sequence(
        self,
        subband_spec_RIs=None,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
        detach_state=False,
        chunk_size: int = 1,
    ):
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        B, _, _, total_frames = subband_spec_RIs.shape

        if (
            past_kvs is None
            or past_valid_mask is None
            or prev_states_0 is None
            or prev_states_1 is None
            or prev_states_2 is None
            or prev_global_states is None
        ):
            (
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.init_streaming_state(
                B,
                device=subband_spec_RIs.device,
                dtype=subband_spec_RIs.dtype,
            )

        frame_outputs: list[torch.Tensor] = []
        for t in range(total_frames):
            frame = subband_spec_RIs[:, :, :, t : t + 1]
            (
                frame_out,
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.forward_cell(
                frame,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
            frame_outputs.append(frame_out)

            if detach_state:
                past_kvs = past_kvs.detach()
                past_valid_mask = past_valid_mask.detach()
                prev_states_0 = prev_states_0.detach()
                prev_states_1 = prev_states_1.detach()
                prev_states_2 = prev_states_2.detach()
                prev_global_states = prev_global_states.detach()

        return (
            torch.cat(frame_outputs, dim=-1),
            past_kvs,
            past_valid_mask,
            prev_states_0,
            prev_states_1,
            prev_states_2,
            prev_global_states,
        )

    def forward(
        self,
        subband_spec_RIs=None,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
        detach_state=False,
        chunk_size: int = 1,
    ):
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        if subband_spec_RIs.shape[-1] == 1:
            return self.forward_cell(
                subband_spec_RIs,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
        return self.forward_sequence(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
            detach_state=detach_state,
            chunk_size=chunk_size,
        )
