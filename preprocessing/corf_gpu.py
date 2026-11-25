#!/usr/bin/env python3
"""Estrazione di mappe di contorno Push-Pull CORF on GPU CUDA.

Example:
    python corf_gpu.py --geometry corf_geometry.mat \
        --input-dir /path/clean --output-dir /path/contours \
        --inhibition-factor 4

"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

# Stessi parametri del benchmark validato contro MATLAB:
# dtype=float64, shift_mode=zero, gray_mode=matlab-uint8.
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
MAX_CACHED_SIZES = 8  # Evita crescita illimitata della memoria GPU su dataset grandi.


def matlab_round(x: float) -> int:
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def matlab_trunc(x: float) -> int:
    return math.trunc(x)  # MATLAB fix(x)


def symmetric_indices(length: int, padding: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(-padding, length + padding, device=device)
    positions = positions.remainder(2 * length)
    return torch.where(positions < length, positions, 2 * length - 1 - positions).long()


@dataclass(frozen=True)
class TupleRF:
    delta: int
    rho_key: float
    shift_rows: int
    shift_cols: int
    weight: float


class CORFGPU:
    def __init__(self, geometry_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError('GPU CUDA non disponibile.')
        self.device = torch.device('cuda:0')
        self.dtype = torch.float64

        g = scipy.io.loadmat(str(geometry_path), squeeze_me=True)
        for field in ('excitation', 'inhibition', 'radius', 'sigma', 'sigmaRatio',
                      'sigma0', 'alpha', 'beta'):
            if field not in g:
                raise ValueError(f'Manca {field} in {geometry_path}: rigenera il MAT con exporter incluso')
        self.sigma = float(g['sigma'])
        self.beta = float(g['beta'])
        self.sigma_ratio = float(g['sigmaRatio'])
        self.sigma0 = float(g['sigma0'])
        self.alpha = float(g['alpha'])
        self.radius = int(g['radius'])
        self.excitation = np.asarray(g['excitation'], dtype=np.float64).reshape(4, -1)
        self.inhibition = np.asarray(g['inhibition'], dtype=np.float64).reshape(4, -1)
        if self.excitation.size == 0 or self.inhibition.size == 0:
            raise RuntimeError('Receptive fields MATLAB vuoti')

        self.orientations = np.deg2rad(np.arange(0, 360, 30)).astype(np.float64)
        keys = set()
        for a in (self.excitation, self.inhibition):
            for k in range(a.shape[1]):
                keys.add((int(round(a[0, k])), round(float(a[2, k]), 3)))
        self.keys = sorted(keys)
        self.weight_sigma = max(rho for _, rho in self.keys) / 3.0
        if self.weight_sigma <= 0:
            raise RuntimeError('weight sigma non valido')
        self.rf = []
        for angle in self.orientations:
            ex = self._build_tuples(self.excitation, angle)
            inh = self._build_tuples(self.inhibition, angle)
            self.rf.append((ex, inh))

        self._dog_cache = {}
        self._smooth_cache = {}
        self._idx_cache = {}

    def _build_tuples(self, raw: np.ndarray, angle: float) -> list[TupleRF]:
        result = []
        for index in range(raw.shape[1]):
            delta = int(round(raw[0, index]))
            rho = float(raw[2, index])
            rho_key = round(rho, 3)
            phi = (float(raw[3, index]) + angle) % (2 * np.pi)
            col = rho * np.cos(phi)
            row = rho * np.sin(phi)
            # Original: imshift(blurredResponse,fix(row),-fix(col))
            dr = matlab_trunc(row)
            dc = matlab_trunc(-col)
            weight = math.exp(-(rho_key ** 2) / (2 * self.weight_sigma ** 2))
            result.append(TupleRF(delta, rho_key, dr, dc, weight))
        return result

    def _gray(self, rgb: torch.Tensor) -> torch.Tensor:
        """Input RGB uint8 HWC, output scala di grigi min-max su GPU."""
        if rgb.ndim == 2:
            gray = rgb.to(self.dtype)
        elif rgb.ndim == 3 and rgb.shape[-1] == 3:
            values = rgb.to(self.dtype)
            gray = (values[..., 0] * 0.298936021293775
                    + values[..., 1] * 0.587043074451121
                    + values[..., 2] * 0.114020904255103)
            gray = gray.round().clamp(0, 255)  # matlab-uint8
        else:
            raise ValueError('Immagine attesa come HxW o HxWx3 RGB')
        minimum = gray.amin()
        maximum = gray.amax()
        return (gray - minimum) / torch.clamp(maximum - minimum, min=1e-20)

    def _dog_fft(self, hp: int, wp: int) -> torch.Tensor:
        key = hp, wp
        if key in self._dog_cache:
            return self._dog_cache[key]

        def gaussian(sigma):
            y = torch.arange(hp, device=self.device, dtype=self.dtype) - (hp - 1) / 2
            x = torch.arange(wp, device=self.device, dtype=self.dtype) - (wp - 1) / 2
            gy = torch.exp(-0.5 * (y / sigma).square())
            gx = torch.exp(-0.5 * (x / sigma).square())
            kernel = gy[:, None] * gx[None, :]
            return kernel / kernel.sum()

        kernel = gaussian(self.sigma) - gaussian(self.sigma * self.sigma_ratio)
        if len(self._dog_cache) >= MAX_CACHED_SIZES:
            self._dog_cache.clear()
        self._dog_cache[key] = torch.fft.fft2(kernel)
        return self._dog_cache[key]

    def _smooth(self, image: torch.Tensor, rho: float) -> torch.Tensor:
        r = matlab_round((self.sigma0 + self.alpha * rho) / 2)
        if r <= 0:
            return image
        if r not in self._smooth_cache:
            x = torch.arange(-r, r + 1, device=self.device, dtype=self.dtype)
            kern = torch.exp(-0.5 * (x / (r / 3.0)).square())
            kern = kern / kern.sum()
            self._smooth_cache[r] = (kern.view(1, 1, -1, 1), kern.view(1, 1, 1, -1))
        ky, kx = self._smooth_cache[r]
        x = image.unsqueeze(0).unsqueeze(0)
        x = F.conv2d(x, ky, padding=(r, 0))
        x = F.conv2d(x, kx, padding=(0, r))
        return x[0, 0]

    def _bank(self, gray: torch.Tensor) -> dict[tuple[int, float], torch.Tensor]:
        h, w = gray.shape
        pad = self.radius
        idx_key = h, w
        if idx_key not in self._idx_cache:
            if len(self._idx_cache) >= MAX_CACHED_SIZES:
                self._idx_cache.clear()
            self._idx_cache[idx_key] = (
                symmetric_indices(h, pad, self.device),
                symmetric_indices(w, pad, self.device)
            )
        ys, xs = self._idx_cache[idx_key]
        padded = gray.index_select(0, ys).index_select(1, xs)
        response = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.fft2(padded) * self._dog_fft(*padded.shape)).real
        )
        lgn = {0: response.clamp_min(0), 1: (-response).clamp_min(0)}
        bank = {}
        for delta, rho in self.keys:
            x = self._smooth(lgn[delta], rho)
            weight = math.exp(-rho ** 2 / (2 * self.weight_sigma ** 2))
            bank[delta, rho] = x.clamp_min(0).pow(weight)
        return bank

    def _rf_response(self, bank, terms, h, w) -> torch.Tensor:
        result = torch.ones((h, w), device=self.device, dtype=self.dtype)
        sum_weights = 0.0
        pad = self.radius
        for term in terms:
            x = bank[term.delta, term.rho_key]
            y0 = pad - term.shift_rows
            x0 = pad - term.shift_cols
            if y0 < 0 or x0 < 0 or y0 + h > x.shape[0] or x0 + w > x.shape[1]:
                raise RuntimeError('Shift eccede il padding. Controllare geometria MATLAB')
            shifted = x[y0:y0+h, x0:x0+w]
            result = result * shifted
            sum_weights += term.weight
        return result.pow(1.0 / sum_weights)

    @torch.inference_mode()
    def response(self, rgb: torch.Tensor, inhibition_factor: float) -> torch.Tensor:
        if rgb.device != self.device:
            raise ValueError(f'Immagine non sul device corretto: {rgb.device} vs {self.device}')
        gray = self._gray(rgb)
        h, w = gray.shape
        bank = self._bank(gray)
        response = torch.zeros((h, w), device=self.device, dtype=self.dtype)
        for ex, inh in self.rf:
            positive = self._rf_response(bank, ex, h, w)
            negative = self._rf_response(bank, inh, h, w)
            oriented = (positive - inhibition_factor * negative).clamp_min(0)
            response = torch.maximum(response, oriented)
        return response

    @torch.inference_mode()
    def contour(self, rgb: torch.Tensor, inhibition_factor: float) -> torch.Tensor:
        response = self.response(rgb, inhibition_factor)
        minimum = response.amin()
        maximum = response.amax()
        normalized = (response - minimum) / torch.clamp(maximum - minimum, min=1e-20)
        return (normalized * 255.0).round().clamp(0, 255).to(torch.uint8)


def extract(geometry: Path, input_dir: Path, output_dir: Path, inhibition_factor: float) -> None:
    if not math.isfinite(inhibition_factor) or inhibition_factor < 0:
        raise ValueError('Il fattore di inibizione deve essere un numero finito >= 0.')
    if not geometry.is_file():
        raise FileNotFoundError(f'File .mat non trovato: {geometry}')
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Cartella input non trovata: {input_dir}')

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError('Le cartelle input e output devono essere diverse.')

    # Include file nelle sottocartelle senza rileggere quelli gia' prodotti.
    paths = sorted(p for p in input_dir.rglob('*')
                   if p.is_file() and p.suffix.lower() in EXTENSIONS
                   and not p.is_relative_to(output_dir))
    if not paths:
        raise RuntimeError(f'Nessuna immagine trovata in {input_dir}')

    corf = CORFGPU(str(geometry))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'GPU: {torch.cuda.get_device_name(corf.device)}', flush=True)
    print(f'CORF: sigma={corf.sigma}, beta={corf.beta}, inhibition={inhibition_factor}', flush=True)
    print(f'Immagini trovate: {len(paths)}', flush=True)

    processed = skipped = failed = 0
    for k, path in enumerate(paths, 1):
        target = output_dir / path.relative_to(input_dir).with_suffix('.png')
        if target.is_file():
            skipped += 1
        else:
            try:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError('Immagine non leggibile')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                x = torch.from_numpy(rgb.copy()).to(corf.device)
                contour = corf.contour(x, inhibition_factor).cpu().numpy()
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), contour):
                    raise RuntimeError('Salvataggio PNG non riuscito')
                processed += 1
            except Exception as exc:
                failed += 1
                print(f'[ERRORE] {path}: {exc}', file=sys.stderr, flush=True)

        if k % 50 == 0 or k == len(paths):
            print(f'[{k}/{len(paths)}] Generate: {processed} | '
                  f'Gia presenti: {skipped} | Errori: {failed}', flush=True)

    if failed:
        raise RuntimeError(f'{failed} immagini non elaborate; controlla gli errori sopra.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Estrai contorni CORF su GPU (PNG uint8).')
    parser.add_argument('--geometry', required=True, type=Path, help='File corf_geometry.mat')
    parser.add_argument('--input-dir', required=True, type=Path, help='Cartella immagini originali')
    parser.add_argument('--output-dir', required=True, type=Path, help='Cartella contorni PNG')
    parser.add_argument('--inhibition-factor', type=float, required=True,
                        help='Fattore di inibizione Push-Pull (es. 4 o 5.2)')
    args = parser.parse_args()
    extract(args.geometry, args.input_dir, args.output_dir, args.inhibition_factor)


if __name__ == '__main__':
    main()
