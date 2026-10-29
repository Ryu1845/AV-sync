import ctypes, math, shlex, subprocess, sys, wave
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SR = 16000
FPS = 25
WINDOW_SIZE = 512
OVERLAP = 256
HOP = WINDOW_SIZE - OVERLAP
WINDOWS_PER_SECOND = SR/(WINDOW_SIZE-OVERLAP)
WINDOWS_PER_FRAME = WINDOWS_PER_SECOND/FPS

batch_size = 32
smooth_div = 20
relax_end = 10
MAX_WARP = 200

data_dir = 'data/'

def tensor_bytes(t):
    t = t.contiguous().cpu()
    return ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())

def read_aud(audio_path, verbose=True):
    if not audio_path.endswith('.wav'):
        video_path = audio_path
        audio_path = video_path[:-4] + '.wav'
        command = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-vn', '-acodec', 'pcm_s16le', '-ar', str(SR), audio_path]
        run_command(command, verbose)

    with wave.open(audio_path, 'rb') as f:
        assert f.getframerate() == SR and f.getnchannels() == 1 and f.getsampwidth() == 2
        y = torch.frombuffer(bytearray(f.readframes(f.getnframes())), dtype=torch.int16)
    # keep the int16 scale: the MFCC features SyncNet was trained on depend on it
    return y.double()

def write_aud(audio_path, aud):
    aud = (aud / aud.abs().max() * 32767).round().to(torch.int16)
    with wave.open(audio_path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(tensor_bytes(aud))

def probe_vid(video_path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate',
         '-of', 'default=noprint_wrappers=1', video_path], capture_output=True, text=True, check=True).stdout
    meta = dict(line.split('=', 1) for line in out.split())
    num, den = meta['r_frame_rate'].split('/')
    return int(meta['width']), int(meta['height']), int(num) / int(den)

def iter_frames(video_path):
    w, h, _ = probe_vid(video_path)
    size = w * h * 3
    command = ['ffmpeg', '-v', 'error', '-i', video_path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        while proc.stdout.readinto(buf := bytearray(size)) == size:
            yield torch.frombuffer(buf, dtype=torch.uint8).view(h, w, 3)

def read_vid(video_path):
    assert probe_vid(video_path)[2] == FPS, 'when changing fps MAKE SURE no new frames are inserted'
    images = torch.stack(list(iter_frames(video_path)))
    print ('number of frames in vid:', len(images))
    return images

def write_vid(video_path, frames, fps=FPS, audio_path=None, shortest=False):
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[:2]
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (w, h),
               '-r', str(fps), '-i', '-']
    if audio_path is not None:
        command += ['-i', audio_path, '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'copy']
    if shortest:
        command += ['-shortest']
    # libx264 at crf 25 is imageio's default; yuv420p needs even frame sizes, avi needs no b-frames
    command += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-crf', '25', '-bf', '0', '-pix_fmt', 'yuv420p',
                video_path]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        proc.stdin.write(tensor_bytes(first))
        for frame in frames:
            proc.stdin.write(tensor_bytes(frame))
    if proc.returncode:
        sys.exit('ffmpeg failed writing %s' % video_path)

def calc_mfcc(aud, numcep=13, nfilt=26, nfft=512, preemph=0.97, ceplifter=22):
    # port of python_speech_features.mfcc with its defaults (25ms frames, 10ms step, no window)
    frame_len, frame_step = int(0.025 * SR), int(0.01 * SR)
    sig = torch.cat([aud[:1], aud[1:] - preemph * aud[:-1]])
    numframes = 1 + max(0, math.ceil((sig.numel() - frame_len) / frame_step))
    padlen = (numframes - 1) * frame_step + frame_len
    frames = F.pad(sig, (0, padlen - sig.numel())).unfold(0, frame_len, frame_step)
    pspec = torch.fft.rfft(frames, nfft).abs() ** 2 / nfft

    eps = torch.finfo(torch.float64).eps
    energy = pspec.sum(1)
    energy = torch.where(energy == 0, eps, energy)
    feat = pspec @ mel_filterbank(nfilt, nfft).T
    feat = torch.where(feat == 0, eps, feat)

    feat = feat.log() @ dct_matrix(nfilt)[:numcep].T
    feat = feat * (1 + ceplifter / 2 * torch.sin(math.pi * torch.arange(numcep, dtype=torch.float64) / ceplifter))
    feat[:, 0] = energy.log()
    return feat.T

def mel_filterbank(nfilt, nfft):
    # computed in python floats to reproduce numpy.linspace and the floored fft bins exactly
    lowmel, highmel = 0.0, 2595 * math.log10(1 + SR / 2 / 700.)
    step = (highmel - lowmel) / (nfilt + 1)
    melpoints = [i * step + lowmel for i in range(nfilt + 1)] + [highmel]
    bins = [math.floor((nfft + 1) * (700 * (10 ** (mel / 2595.0) - 1)) / SR) for mel in melpoints]
    fbank = torch.zeros(nfilt, nfft // 2 + 1, dtype=torch.float64)
    for j in range(nfilt):
        for i in range(bins[j], bins[j + 1]):
            fbank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(bins[j + 1], bins[j + 2]):
            fbank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    return fbank

def dct_matrix(n):
    # orthonormal DCT-II, rows are basis vectors
    k = torch.arange(n, dtype=torch.float64)
    mat = torch.cos(math.pi * k[:, None] * (2 * k[None] + 1) / (2 * n)) * math.sqrt(2 / n)
    mat[0] /= math.sqrt(2)
    return mat

def interp(x, xp, fp):
    """numpy.interp along dim 0 of fp (xp increasing); x outside xp takes the edge values."""
    x = x.clamp(xp[0], xp[-1])
    i = torch.searchsorted(xp, x, right=True).clamp(1, len(xp) - 1)
    w = ((x - xp[i - 1]) / (xp[i] - xp[i - 1])).to(fp.dtype)
    return torch.lerp(fp[i - 1], fp[i], w.view(-1, *[1] * (fp.dim() - 1)))

def medfilt(x, kernel_size):
    # scipy.signal.medfilt: zero padded running median
    r = kernel_size // 2
    return F.pad(x, (r, r)).unfold(0, kernel_size, 1).median(-1).values

def gaussian_filter(x, sigma, truncate=4.0):
    # 1d scipy.ndimage.gaussian_filter with mode='reflect' (half-sample symmetric)
    radius = int(truncate * sigma + 0.5)
    k = torch.arange(-radius, radius + 1, dtype=torch.float64)
    weights = torch.exp(-0.5 / sigma ** 2 * k ** 2)
    weights /= weights.sum()
    n = x.numel()
    idx = torch.arange(-radius, n + radius) % (2 * n)
    idx = torch.where(idx < n, idx, 2 * n - 1 - idx)
    return x[idx].unfold(0, 2 * radius + 1, 1) @ weights

def stabilizeSequence(samples, regularization_lambda=30.0):
    length = samples.numel()
    if length < 6:
        return samples

    # find x that solves laplacian_matrix * x = b, laplacian_matrix = tridiag(-lambda, 1 + 2 lambda, -lambda).
    # It is strictly diagonally dominant, so the Thomas algorithm is stable and is what LU without pivoting does.
    s = samples.tolist()
    b = s[1:-1]
    b[0] += regularization_lambda * s[0]
    b[-1] += regularization_lambda * s[-1]
    diag, off = 1.0 + 2 * regularization_lambda, -1.0 * regularization_lambda

    cp, dp = [off / diag], [b[0] / diag]
    for k in range(1, len(b)):
        denom = diag - off * cp[-1]
        cp.append(off / denom)
        dp.append((b[k] - off * dp[-1]) / denom)
    for k in range(len(b) - 2, -1, -1):
        dp[k] -= cp[k] * dp[k + 1]

    x = samples.clone()
    x[1:-1] = torch.tensor(dp, dtype=samples.dtype)
    return x

def aud2tensor(aud):
    return calc_mfcc(aud).float()[None, None]

def aud2emb(aud, model, every_frame=False):
    if type(aud) == str:
        aud = read_aud(aud)
    lastframe = round(aud.numel()/SR*FPS) - 6
    step = 1 if every_frame else 4
    if every_frame:
        lastframe *= 4
    windows = aud2tensor(aud)[0, 0].unfold(1, 20, step)  # 13 x N x 20 view, window k starts at mfcc frame k*step
    dev = next(model.__S__.parameters()).device
    cc_feat = []
    with torch.inference_mode():
        for i in range(0, lastframe, batch_size):
            cc_in = windows[:, i:min(lastframe, i+batch_size)].permute(1, 0, 2)[:, None].contiguous()
            cc_feat.append(model.__S__.forward_aud(cc_in.to(dev)).cpu())
    return aud, torch.cat(cc_feat)

def _lip_batch(model, frames, dev):
    # T x H x W x 3 uint8 frames -> embeddings of the T-4 windows of 5 consecutive frames
    clip = torch.stack(frames).to(dev).permute(3, 0, 1, 2)  # 3 x T x H x W
    im_in = clip.unfold(1, 5, 1).permute(1, 0, 4, 2, 3).float().contiguous()  # (T-4) x 3 x 5 x H x W
    return model.__S__.forward_lip(im_in).cpu()

def vid2emb(images, model, every_frame=False):
    """images: video path or a sequence of 224 x 224 x 3 uint8 frames; frames are streamed, not stored."""
    if type(images) == str:
        assert probe_vid(images)[2] == FPS, 'when changing fps MAKE SURE no new frames are inserted'
        images = iter_frames(images)
    dev = next(model.__S__.parameters()).device
    # windows start at frames 0 .. T-7, batch_size at a time; a full batch needs batch_size + 6 buffered frames
    im_feat, buf = [], []
    with torch.inference_mode():
        for frame in images:
            assert frame.shape == (224, 224, 3)
            buf.append(frame)
            if len(buf) == batch_size + 6:
                im_feat.append(_lip_batch(model, buf[:batch_size + 4], dev))
                buf = buf[batch_size:]
        if len(buf) > 6:
            im_feat.append(_lip_batch(model, buf[:-2], dev))
    im_feat = torch.cat(im_feat)

    if every_frame:
        frames = torch.arange(len(im_feat), dtype=torch.float64)
        im_feat = interp(torch.arange(len(im_feat) * 4, dtype=torch.float64) / 4, frames, im_feat)

    return im_feat

def run_command(command, verbose=True):
    print('runing: ', shlex.join(command) if verbose else '...' , end=' ')
    ret = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ret :
        print('\nFailed!!!')
        sys.exit(1)
    else:
        print('.... OK!')

def _window():
    return torch.hann_window(WINDOW_SIZE, dtype=torch.float64)

def aud2spec(aud):
    # scipy.signal.stft framing (zero boundary, padded to whole frames), without its 1/sum(window) scale
    aud = F.pad(aud, (0, -aud.numel() % HOP))
    return torch.stft(aud, WINDOW_SIZE, HOP, window=_window(), center=True, pad_mode='constant', return_complex=True)

def spec2aud(spec):
    return torch.istft(spec, WINDOW_SIZE, HOP, window=_window(), center=True)

def resize_vec(vec, new_length):
    new_length = int(new_length)
    vec = vec * new_length / vec.numel()
    coordinates = torch.linspace(0, vec.numel()-1, new_length, dtype=torch.float64)
    return interp(coordinates, torch.arange(vec.numel(), dtype=torch.float64), vec)

def band_limits(n, m, max_warp=MAX_WARP):
    # row i of the dp only visits columns [lo[i], hi[i]); both are non-decreasing in i
    lo = [max(0, i*m//n - max_warp) for i in range(n)]
    hi = [min(i*m//n + max_warp, m) for i in range(n)]
    return lo, hi

def data_cost(emb1, emb2, lo, hi, prefer_delay=False, block=64):
    """Minimal mean squared distance over all pairs of feature streams, for the band columns only.

    Row i holds cost[i, lo[i]:hi[i]] at [i, :hi[i]-lo[i]], the rest of the row is inf."""
    n = len(lo)
    width = max(h - l for l, h in zip(lo, hi))
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    out = torch.full((n, width), math.inf, dtype=torch.float64)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        # with prefer_delay each row also needs the previous row on its own columns
        r0 = max(0, i0 - 1) if prefer_delay else i0
        c0, c1 = lo[i0], hi[i1 - 1]
        cost = torch.stack([
            torch.cdist(f1[r0:i1].double(), f2[c0:c1].double(), compute_mode='donot_use_mm_for_euclid_dist') ** 2
            / f1.shape[1] for f1 in emb1 for f2 in emb2]).amin(0)
        if prefer_delay:
            prev = cost[:-1] if r0 < i0 else torch.cat([cost[:1], cost[:-1]])
            cost = torch.minimum(cost[i0 - r0:], prev)
        cols = (lo_t[i0:i1, None] - c0) + torch.arange(width)
        valid = cols < (hi_t[i0:i1, None] - c0)
        out[i0:i1] = torch.where(valid, cost.gather(1, cols.clamp(max=c1 - c0 - 1)), math.inf)
    return out

def is_init(i, j):
    # cells the dp starts from (zero accumulated cost)
    return (i == 0 and j < relax_end) or (j == 0 and i < relax_end)

class BandedDP:
    """Accumulated dp cost and back pointers, stored only around the band each row visits.

    Row i keeps columns [start[i], end[i]) of the accumulated cost and one direction code per band cell
    (0: left, 1: diagonal, 2: up). Every other cell holds its initial value (0 or inf)."""

    def __init__(self, n, m, lo, hi, start, end, vals, dirs):
        self.shape = (n, m)
        self.lo, self.hi, self.start, self.end, self.vals, self.dirs = lo, hi, start, end, vals, dirs

    def value(self, i, j):
        if self.start[i] <= j < self.end[i]:
            return self.vals[i, j - self.start[i]].item()
        return 0.0 if is_init(i, j) else math.inf

    def origin(self, i, j):
        k = j - self.lo[i]
        if not 0 <= k < self.hi[i] - self.lo[i]:
            return 0, 0
        d = self.dirs[i][k]
        pi, pj = max(0, i-1), max(0, j-1)
        return ((i, pj), (pi, pj), (pi, j))[d]

    def dense(self):
        n, m = self.shape
        mat = torch.full((n, m), math.inf, dtype=torch.float64)
        mat[0, :relax_end] = 0
        mat[:relax_end, 0] = 0
        for i, (s, e) in enumerate(zip(self.start, self.end)):
            mat[i, s:e] = self.vals[i, :e - s]
        return mat

def dynamic_programming(emb1, emb2, max_warp=MAX_WARP, prefer_delay=False):
    if not type(emb1) == list:
        emb1 = [emb1]
        emb2 = [emb2]
    n, m = emb1[0].shape[0], emb2[0].shape[0]
    lo, hi = band_limits(n, m, max_warp)
    cost = data_cost(emb1, emb2, lo, hi, prefer_delay)
    smooth_cost = 1 / smooth_div

    # row i is read by rows i and i+1 at columns [lo[i]-1, hi[i+1]); keep exactly those
    start = [max(0, l - 1) for l in lo]
    end = hi[1:] + hi[-1:]
    vals = torch.full((n, max(e - s for s, e in zip(start, end))), math.inf, dtype=torch.float64)
    dirs = []

    # plain python floats: fast enough for the banded loop and keeps np.argmin's tie breaking
    prev_row = None
    for i in range(n):
        s, e = start[i], end[i]
        row = [math.inf] * (e - s)
        if i < relax_end and s == 0:
            row[0] = 0.0
        if i == 0:
            row[:relax_end] = [0.0] * min(relax_end, e)
            prev_row = row
        pi = max(0, i-1)
        ps = start[pi]
        l, h = lo[i], hi[i]
        cost_row = cost[i, :h - l].tolist()
        d_row = bytearray(h - l)
        for j in range(l, h):
            pj = j - 1 if j else 0
            c = cost_row[j - l]
            left, diag, up = c + smooth_cost + row[pj - s], c + prev_row[pj - ps], c + smooth_cost + prev_row[j - ps]
            if left <= diag and left <= up:
                row[j - s] = left
            elif diag <= up:
                row[j - s], d_row[j - l] = diag, 1
            else:
                row[j - s], d_row[j - l] = up, 2
        vals[i, :e - s] = torch.tensor(row, dtype=torch.float64)
        dirs.append(d_row)
        prev_row = row
    return BandedDP(n, m, lo, hi, start, end, vals, dirs)

def done(i,j):
    return is_init(i, j)

def find_path(dp):
    n, m = dp.shape
    last_cols = range(m)[-relax_end:]
    last_rows = range(n)[-relax_end:]
    amin1 = torch.tensor([dp.value(n-1, j) for j in last_cols], dtype=torch.float64).argmin().item() + m-relax_end
    amin2 = torch.tensor([dp.value(i, m-1) for i in last_rows], dtype=torch.float64).argmin().item() + n-relax_end
    amin1, amin2 = range(m)[amin1], range(n)[amin2]
    path = []
    if dp.value(n-1, amin1) < dp.value(amin2, m-1):
        i,j = n-1, amin1
    else:
        i,j = amin2, m-1
    cost = dp.value(i, j)
    while not done(i,j):
        path.append([i,j])
        i,j = dp.origin(i, j)
    path.append([i,j])
    path = path[::-1]
    return torch.tensor(path), cost

def griffin_lim(spec, iterations=100):
    mag = spec.abs()
    iterations = max(iterations, 1)
    for i in range(iterations):
        wav = spec2aud(spec)
        spec = torch.polar(mag, aud2spec(wav).angle())
    return wav

def phase_vocoder(spec, time_steps):
    # magnitude: linear interpolation in time, clamped at the edges
    frames = torch.arange(spec.shape[1], dtype=torch.float64)
    warped_mag = interp(time_steps, frames, spec.abs().T).T

    # Expected phase advance in each bin
    phi_advance = torch.linspace(0, math.pi * HOP, spec.shape[0], dtype=torch.float64)[:, None]

    spec_angle = F.pad(spec.angle(), (0, 1))
    steps = time_steps.floor().long()

    # Phase advance between the source frames, wrapped to -pi:pi range
    dphase = spec_angle[:, steps+1] - spec_angle[:, steps] - phi_advance
    dphase = torch.remainder(dphase-math.pi, 2*math.pi) - math.pi

    # Accumulate phase, starting from the phase of the first frame
    phase_acc = torch.cumsum(phi_advance + dphase, 1)
    phase_acc = spec_angle[:, :1] + torch.cat([torch.zeros_like(phase_acc[:, :1]), phase_acc[:, :-1]], 1)

    return torch.polar(warped_mag, phase_acc)
