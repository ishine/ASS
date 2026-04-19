"""
Archived **v1** TF-MLPNet–style TIGER separator from ``context.md`` (chat export).

Compared to ``tiger_edge_mlp.py`` (v2):

- Uses ``V1EdgeChannelNorm`` + ``V1EdgePWConvBlock`` (mean/var normalization on channel).
- Time mixer caches **expanded** hidden channels (larger streaming state).
- Global path uses ``expand`` to broadcast ``[B, C, F, 1]`` to ``T``.
- ``V1TIGEREdgeMLP.forward_sequence`` feeds **chunks** to the separator (not
  frame-by-frame ``forward_cell`` like v2).

**Integration fixes** (same as v2 vs raw chat text):

- Encoder yields ``[B, nband, feature_dim, T]``; separator permutes to internal
  ``[B, feature_dim, nband, T]`` and permutes output back to
  ``[B, nband, feature_dim, T]`` for ``_decode_masks``.
- ``out_proj`` outputs **``feature_dim``** channels (not ``num_output * feature_dim``).

All public symbols are prefixed with ``V1`` to avoid clashing with v2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from TIGER.tiger_online import TIGER


class V1EdgeChannelNorm(nn.Module):
    """Channel-only norm on ``[B, C, F, T]`` (4D only)."""

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        return centered * inv_std * self.weight + self.bias


class V1EdgePWConvBlock(nn.Module):
    """1x1 Conv2d + ReLU + ``V1EdgeChannelNorm``."""

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
        self.norm = V1EdgeChannelNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class V1EdgeFreqMixer(nn.Module):
    """Frequency-axis mixer: 1x1 → DW (kf,1) → 1x1 + residual."""

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

        self.pre = V1EdgePWConvBlock(channels, hidden, bias=bias)
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
        self.dw_norm = V1EdgeChannelNorm(hidden)
        self.post = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.post_norm = V1EdgeChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre(x)
        x = self.dw(x)
        x = self.dw_act(x)
        x = self.dw_norm(x)
        x = self.post(x)
        x = self.post_norm(x)
        return residual + x


class V1EdgeTimeMixer(nn.Module):
    """
    Causal time mixer; state is ``[B, hidden_channels * expansion, F, W]``.
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

        hidden = channels * expansion
        self.channels = channels
        self.hidden = hidden
        self.time_kernel_size = time_kernel_size
        self.dilation = dilation
        self.state_width = (time_kernel_size - 1) * dilation

        self.pre = V1EdgePWConvBlock(channels, hidden, bias=bias)

        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, time_kernel_size),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, dilation),
            groups=hidden,
            bias=bias,
        )
        self.dw_act = nn.ReLU()
        self.dw_norm = V1EdgeChannelNorm(hidden)

        self.post = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.post_norm = V1EdgeChannelNorm(channels)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.pre(x)

        B, H, Freq, T = x.shape

        if self.state_width > 0:
            if state is None:
                state = x.new_zeros(B, H, Freq, self.state_width)
            combined = torch.cat([state, x], dim=-1)
            new_state = combined[:, :, :, -self.state_width :]
        else:
            combined = x
            new_state = x[:, :, :, :0]

        x = self.dw(combined)
        x = self.dw_act(x)
        x = self.dw_norm(x)
        x = self.post(x)
        x = self.post_norm(x)
        return residual + x, new_state


class V1EdgeTFMLPBlock(nn.Module):
    """Freq + time mixers + global ``expand`` gate (v1 chat design)."""

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        time_kernel_size: int = 3,
        dilation: int = 1,
        global_context_width: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.global_context_width = global_context_width

        self.freq_mixer = V1EdgeFreqMixer(
            channels=channels,
            expansion=expansion,
            freq_kernel_size=freq_kernel_size,
            bias=bias,
        )
        self.time_mixer = V1EdgeTimeMixer(
            channels=channels,
            expansion=expansion,
            time_kernel_size=time_kernel_size,
            dilation=dilation,
            bias=bias,
        )

        self.gate_conv = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.mix_conv = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.out_norm = V1EdgeChannelNorm(channels)

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
        time_state: torch.Tensor | None = None,
        global_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, Freq, T = x.shape

        x = self.freq_mixer(x)
        x, new_time_state = self.time_mixer(x, state=time_state)

        if global_state is None:
            global_state = x.new_zeros(B, C, Freq, self.global_context_width)

        if global_state.shape[-1] != 1:
            global_state = global_state[:, :, :, -1:]
        global_rep = global_state.expand(B, C, Freq, T)

        fusion_in = torch.cat([x, global_rep], dim=1)
        gate = torch.sigmoid(self.gate_conv(fusion_in))
        mix = self.mix_conv(fusion_in)
        x = self.out_norm(x + gate * mix)

        new_global_state = self.global_update(x[:, :, :, -1:])
        return x, new_time_state, new_global_state


class V1EdgeTFMLPSeparator(nn.Module):
    """
    v1 separator: optional ``None`` states, KV passthrough outputs,
    encoder layout ``[B, nband, feature_dim, T]`` in / out.
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
        assert len(time_dilations) == 3, "time_dilations must have length 3, e.g. (1,2,4)"
        assert num_blocks >= 1, "num_blocks must be >= 1"
        assert hidden_channels >= 1, "hidden_channels must be >= 1"

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.nband = nband
        self.num_output = num_output
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.freq_kernel_size = freq_kernel_size
        self.time_kernel_size = time_kernel_size
        self.time_dilations = tuple(int(d) for d in time_dilations)

        self.dummy_kv_channels = int(dummy_kv_channels)
        self.dummy_kv_width = int(dummy_kv_width)

        self.n_heads = 1
        self.kv_window_size = self.dummy_kv_width
        self.att_hid_chan = self.dummy_kv_channels
        self.att_val_hid_chan = 0
        self.iter = self.num_blocks

        self.in_proj = V1EdgePWConvBlock(input_dim, hidden_channels, bias=bias)

        self.blocks = nn.ModuleList(
            [
                V1EdgeTFMLPBlock(
                    channels=hidden_channels,
                    expansion=expansion,
                    freq_kernel_size=freq_kernel_size,
                    time_kernel_size=time_kernel_size,
                    dilation=self.time_dilations[i % 3],
                    global_context_width=1,
                    bias=bias,
                )
                for i in range(num_blocks)
            ]
        )

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
            assert width < 14, "Per-group state width must remain small for deployment"

        hd = hidden_channels * expansion
        self.group_state_channels = [
            len(self.group_block_indices[0]) * hd,
            len(self.group_block_indices[1]) * hd,
            len(self.group_block_indices[2]) * hd,
        ]

        self.global_state_channels = num_blocks * hidden_channels
        self.global_state_width = 1

    def _make_dummy_kv(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size,
            self.n_heads,
            self.kv_window_size,
            self.dummy_kv_channels,
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

    def _slice_group_state(self, packed_state: torch.Tensor, block_local_idx: int, hidden_dim: int):
        c0 = block_local_idx * hidden_dim
        c1 = c0 + hidden_dim
        return packed_state[:, c0:c1, :, :]

    def forward(
        self,
        x: torch.Tensor,
        past_kvs: torch.Tensor | None = None,
        past_valid_mask: torch.Tensor | None = None,
        prev_states_0: torch.Tensor | None = None,
        prev_states_1: torch.Tensor | None = None,
        prev_states_2: torch.Tensor | None = None,
        prev_global_states: torch.Tensor | None = None,
    ):
        assert x.dim() == 4, "V1EdgeTFMLPSeparator expects 4D tensor"
        B, nband_in, c_in, T = x.shape
        assert nband_in == self.nband, f"Expected nband={self.nband}, got {nband_in}"
        assert c_in == self.input_dim, f"Expected feature_dim={self.input_dim}, got {c_in}"

        x = x.permute(0, 2, 1, 3).contiguous()

        if past_kvs is None:
            past_kvs = self._make_dummy_kv(B, x.device, x.dtype)
        if past_valid_mask is None:
            past_valid_mask = self._make_dummy_mask(B, x.device, x.dtype)

        if prev_states_0 is None:
            prev_states_0 = x.new_zeros(B, self.group_state_channels[0], self.nband, self.group_state_widths[0])
        if prev_states_1 is None:
            prev_states_1 = x.new_zeros(B, self.group_state_channels[1], self.nband, self.group_state_widths[1])
        if prev_states_2 is None:
            prev_states_2 = x.new_zeros(B, self.group_state_channels[2], self.nband, self.group_state_widths[2])
        if prev_global_states is None:
            prev_global_states = x.new_zeros(B, self.global_state_channels, self.nband, self.global_state_width)

        h = self.in_proj(x)

        new_group0: list[torch.Tensor] = []
        new_group1: list[torch.Tensor] = []
        new_group2: list[torch.Tensor] = []
        new_global: list[torch.Tensor] = []

        group_local_counters = [0, 0, 0]
        hd = self.hidden_channels * self.expansion

        for block_idx, block in enumerate(self.blocks):
            group_id = block_idx % 3
            local_idx = group_local_counters[group_id]
            group_local_counters[group_id] += 1

            if group_id == 0:
                state_slice = self._slice_group_state(prev_states_0, local_idx, hd)
            elif group_id == 1:
                state_slice = self._slice_group_state(prev_states_1, local_idx, hd)
            else:
                state_slice = self._slice_group_state(prev_states_2, local_idx, hd)

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


class V1TIGEREdgeMLP(TIGER):
    """v1 TIGER + TF-MLP separator (chunk ``forward_sequence``)."""

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

        assert len(edge_time_dilations) == 3, "edge_time_dilations must have length 3"
        for d in edge_time_dilations:
            assert (edge_time_kernel_size - 1) * int(d) < 14, (
                "NPU constraint violated: (kernel_size - 1) * dilation must be < 14"
            )

        self.separator = V1EdgeTFMLPSeparator(
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
            assert subband_spec_RIs.shape[-1] == 1, "forward_cell expects a single frame (T=1)"

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
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
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
        chunk_size: int = 8,
    ):
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        batch_size, _, _, total_frames = subband_spec_RIs.shape

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
                batch_size,
                device=subband_spec_RIs.device,
                dtype=subband_spec_RIs.dtype,
            )

        if not self.supports_exact_chunk_training:
            chunk_size = 1
        elif chunk_size is None or chunk_size <= 0:
            chunk_size = total_frames

        frame_outputs: list[torch.Tensor] = []
        for t in range(0, total_frames, chunk_size):
            chunk = subband_spec_RIs[:, :, :, t : t + chunk_size]
            subband_features = self._encode_subbands(chunk)
            (
                sep_output,
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.separator(
                subband_features,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
            frame_outputs.append(self._decode_masks(sep_output))

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
        chunk_size: int = 8,
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
