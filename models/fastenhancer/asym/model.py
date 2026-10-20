"""FastEnhancer + 비대칭 분석/합성 창 (Wang 외 2021, arXiv 2106.11794 식 1-5). lowlat L3용.

분석창 A는 n_fft(32 ms) 길이로 길게 듣고, 합성창 S는 마지막 2*hop만 0이 아니라서
출력이 기다리는 시간(알고리즘 지연)이 2*hop(hop 64면 8 ms)이 된다.
모델 가중치는 default와 같다. 창은 persistent=False 버퍼라 state_dict 키가 그대로라 공개 가중치 106개가 다 들어간다.
"""
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from functional import ONNXSTFT
from models.fastenhancer.default import model as base


def asym_windows(n_fft: int, hop: int, d: int = 0):
    """분석창 A(길이 K=n_fft), 합성창 S(길이 K, 마지막 2M만 0이 아님, M=hop). 겹치는 끝 2M에서 A*S = 길이 2M Hann (식 4)."""
    K, M = n_fft, hop
    n = torch.arange(K, dtype=torch.float64)
    h2m = lambda m: 0.5 * (1 - torch.cos(math.pi * m / M))                   # 식 1: 길이 2M 주기 Hann
    A, S = torch.zeros(K, dtype=torch.float64), torch.zeros(K, dtype=torch.float64)
    r = (n >= d) & (n < K - M)
    A[r] = torch.sqrt(0.5 * (1 - torch.cos(math.pi * (n[r] - d) / (K - M - d))))  # 식 3
    t = n >= K - M
    A[t] = S[t] = torch.sqrt(h2m(n[t] - K + 2 * M))                          # 식 2
    s = (n >= K - 2 * M) & (n < K - M)
    S[s] = h2m(n[s] - K + 2 * M) / A[s]                                      # 식 5
    return A.float(), S.float()


class AsymCompressedSTFT(torch.nn.Module):
    """학습용(오프라인). 프레임 k = x[kH-(N-H) : kH+H] 라서 ONNX 스트리밍과 프레이밍이 같다.
    fp16 학습 중에도 FFT와 창은 fp32로 계산한다(창 나눗셈 S = h/A가 반정밀도에서 흔들리지 않게)."""
    def __init__(self, n_fft: int, hop_size: int, compression: float, eps: float = 1e-5):
        super().__init__()
        self.n_fft, self.hop_size, self.compression, self.eps = n_fft, hop_size, compression, eps
        A, S = asym_windows(n_fft, hop_size)
        self.register_buffer("wa", A, persistent=False)
        self.register_buffer("ws", S, persistent=False)

    def forward(self, x: Tensor) -> Tensor:                 # [B,T] -> [B, N//2, K, 2] (압축)
        if x.dim() == 3:
            x = x.squeeze(1)
        N, H = self.n_fft, self.hop_size
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.pad(x.float(), (N - H, H)).unfold(-1, N, H) * self.wa       # [B, K, N]
            x = torch.view_as_real(torch.fft.rfft(x, dim=-1)[..., :-1]).transpose(1, 2).contiguous()
            mag = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
            return x * mag.pow(self.compression - 1.0)

    def inverse(self, x: Tensor) -> Tensor:                 # [B, N//2, K] 복소(압축) -> [B, T]
        N, H = self.n_fft, self.hop_size
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.complex64)
            x = x * x.abs().pow(1.0 / self.compression - 1.0)
            x = F.pad(x, (0, 0, 0, 1))                                       # 버린 마지막 bin(0) 복원
            x = torch.fft.irfft(x.transpose(1, 2), n=N, dim=-1) * self.ws    # [B, K, N]
            K = x.size(1)
            y = F.fold(x.transpose(1, 2), (1, (K - 1) * H + N), (1, N), stride=(1, H)).flatten(1)
            return y[:, N - H: N - H + (K - 1) * H]


class AsymONNXSTFT(ONNXSTFT):
    """스트리밍(ONNX)용. 입력 캐시 N-H, 출력 캐시 2M-H. 출력은 오프라인보다 syn-H 샘플 늦게 나온다(지연 2M)."""
    def __init__(self, n_fft: int, hop_size: int):
        super().__init__(n_fft, hop_size)
        A, S = asym_windows(n_fft, hop_size)
        self.syn = 2 * hop_size
        self.window = A                        # 기존 버퍼 교체(persistent=False 유지)
        self.window_istft = S[-self.syn:]

    def initialize_cache(self, x: Tensor):
        z = lambda n: torch.zeros(x.size(0), n, dtype=x.dtype, device=x.device)
        return [z(self.cache_len), z(self.syn - self.hop_size)]

    def inverse(self, x: Tensor, cache: Tensor):
        # irFFT 부분은 ONNXSTFT.inverse(Method 2)와 같음
        x_0, x_last = x[:, 0:1, 0, 0], x[:, -1:, 0, 0]
        x = F.pad(x.squeeze(2), (0, 0, 0, self.n_fft // 2 - 1))
        x = torch.fft.ifft(torch.view_as_complex(x), dim=1).real
        x = x.reshape(-1, self.n_fft // 2, 2)
        x = 2 * x - torch.stack([x_0 + x_last, x_0 - x_last], dim=2) / self.n_fft
        x = x.view(-1, self.n_fft)[:, -self.syn:] * self.window_istft        # 짧은 합성창 구간만
        x[:, :cache.size(1)] += cache
        return x[:, :self.hop_size], x[:, self.hop_size:]


class ONNXModel(base.ONNXModel):
    def get_stft(self, n_fft, hop_size, win_size, window, normalized):
        return AsymONNXSTFT(n_fft, hop_size)


class Model(base.Model):
    def get_stft(self, n_fft, hop_size, win_size, window, normalized):
        return AsymCompressedSTFT(n_fft, hop_size, self.input_compression)


if __name__ == "__main__":   # python -m models.fastenhancer.asym.model  (CPU 검사, 몇 초)
    st = AsymCompressedSTFT(512, 64, 0.3)
    t = torch.arange(16000) / 16000
    x = (torch.sin(2 * math.pi * 440 * t) + 0.5 * torch.sin(2 * math.pi * 3000 * t)).unsqueeze(0)
    y = st.inverse(torch.view_as_complex(st(x).contiguous()))
    err = (y - x)[:, :-64].abs().max().item()
    assert y.shape == x.shape and err < 2e-3, (y.shape, err)
    print("asym STFT 복원 OK, max err", err)
