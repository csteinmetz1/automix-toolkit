import warnings
import pyloudnorm as pyln
import sklearn
import scipy
import librosa
import numpy as np




# Utils - general-purpose functions

def amp_to_db(x):
    return 20*np.log10(x + 1e-30)

def db_to_amp(x):
    return 10**(x/20)

def running_mean_std(x, N):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        cumsum2 = np.cumsum(np.insert(x**2, 0, 0)) 
        mean = (cumsum[N:] - cumsum[:-N]) / float(N)

        std = np.sqrt(((cumsum2[N:] - cumsum2[:-N]) / N) - (mean * mean)) 

    return mean, std

def get_running_stats(x, features, N=20):
    """
    Returns running mean and standard deviation from array x. This, based on the previous N frames

    Args:
        x: multi-dimensional array containing features
        features: index of features to be taken into account
    Returns:
        mean and std arrays
    """
    
    mean = []
    std = []
    x = x.copy()
    for i in range(len(features)):
        mean_, std_ = running_mean_std(x[:,i], N)
        mean.append(mean_)
        std.append(std_)
    mean = np.asarray(mean)
    std = np.asarray(std)
    
    return mean, std

def compute_stft(samples, hop_length, fft_size, stft_window):
    """
    Compute the STFT of `samples` applying a Hann window of size `FFT_SIZE`, shifted for each frame by `hop_length`.

    Args:
        samples: num samples x channels
        hop_length: window shift in samples
        fft_size: FFT size which is also the window size
        stft_window: STFT analysis window

    Returns:
        stft: frames x channels x freqbins
    """
    n_channels = samples.shape[1]
    n_frames = 1+int((samples.shape[0] - fft_size)/hop_length)
    stft = np.empty((n_frames, n_channels, fft_size//2+1), dtype=np.complex64)

    # convert into f_contiguous (such that [:,n] slicing is c_contiguous)
    samples = np.asfortranarray(samples)

    for n in range(n_channels):
        # compute STFT (output has size `n_frames x N_BINS`)
        stft[:, n, :] = librosa.stft(samples[:, n],
                                     n_fft=fft_size,
                                     hop_length=hop_length,
                                     window=stft_window,
                                     center=False).transpose()
    return stft


# Functions to compute Fx-related low-level features


# Loudness

def get_lufs_peak_frames(x, sr, frame_size, hop_size):
    """
    Computes lufs and peak loudness in a frame-wise manner

    Args:
        x: audio
        sr: sampling rate
        frame_size: frame size, ideally larger than 400ms (LUFS)
        hop_size: frame hop

    Returns:
        loudness_ and peak_ arrays
    """ 
    
    x_frames = librosa.util.frame(x.T, frame_length=frame_size, hop_length=hop_size)

    peak_ = []
    loudness_ = []

    for i in range(x_frames.shape[-1]):

        x_ = x_frames[:,:,i]
        peak = np.max(np.abs(x_.T))
        peak_db = 20.0 * np.log10(peak)
        peak_.append(peak_db)

        meter = pyln.Meter(sr, block_size=0.4) # create BS.1770 meter
        loudness = meter.integrated_loudness(x_.T)
        loudness_.append(loudness)
    peak_ = np.asarray(peak_)
    loudness_ = np.asarray(loudness_)
    
    peak_ = np.expand_dims(peak_, -1)
    loudness_ = np.expand_dims(loudness_, -1)
    
    return loudness_, peak_
    
def compute_loudness_features(audio_out, audio_tar, sr, frame_size, hop_size):
    """
    Computes lufs and peak loudness mape error using a running mean

    Args:
        audio_out: automix audio (output of models)
        audio_tar: target audio
        sr: sampling rate
        frame_size: frame size, ideally larger than 400ms (LUFS)
        hop_size: frame hop

    Returns:
        loudness_ dictionary
    """ 
    
   
    loudness_ = {key:[] for key in ['lufs_loudness', 'peak_loudness', ]}
    
    loudness_tar, peak_tar = get_lufs_peak_frames(audio_tar, sr, frame_size, hop_size)
    loudness_out, peak_out = get_lufs_peak_frames(audio_out, sr, frame_size, hop_size)
    
    eps = 1e-10
    N = 40 # Considers previous 40 frames
    
    if peak_tar.shape[0] > N:
        
        mean_peak_tar, std_peak_tar = get_running_stats(peak_tar+eps, [0], N=N)
        mean_peak_out, std_peak_out = get_running_stats(peak_out+eps, [0], N=N)

        mean_lufs_tar, std_lufs_tar = get_running_stats(loudness_tar+eps, [0], N=N)
        mean_lufs_out, std_lufs_out = get_running_stats(loudness_out+eps, [0], N=N)
        
    else:
        
        mean_peak_tar = np.expand_dims(np.asarray([np.mean(peak_tar+eps)]), 0)
        mean_peak_out = np.expand_dims(np.asarray([np.mean(peak_out+eps)]), 0)
        mean_lufs_tar = np.expand_dims(np.asarray([np.mean(loudness_tar+eps)]), 0)
        mean_lufs_out = np.expand_dims(np.asarray([np.mean(loudness_out+eps)]), 0)
        
    mape_mean_peak = sklearn.metrics.mean_absolute_percentage_error(mean_peak_tar[0], mean_peak_out[0])
    mape_mean_lufs = sklearn.metrics.mean_absolute_percentage_error(mean_lufs_tar[0], mean_lufs_out[0])
    
    loudness_['peak_loudness'] = mape_mean_peak
    loudness_['lufs_loudness'] = mape_mean_lufs
    
    loudness_['mean_mape_loudness'] = np.mean([loudness_['peak_loudness'],
                                      loudness_['lufs_loudness'],
                                      ])

    return loudness_


# Spectral

def compute_spectral_features(audio_out, audio_tar, sr, fft_size=4096, hop_length=1024, channels=2):
    """
    Computes spectral features' mape error using a running mean

    Args:
        audio_out: automix audio (output of models)
        audio_tar: target audio
        sr: sampling rate
        fft_size: fft_size size
        hop_length: fft hop size
        channels: channels to compute

    Returns:
        spectral_ dictionary
    """ 
    
    audio_out_ = pyln.normalize.peak(audio_out, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar, -1.0)
    
    spec_out_ = compute_stft(audio_out_,
                         hop_length,
                         fft_size,
                         np.sqrt(np.hanning(fft_size+1)[:-1]))
    spec_out_ = np.transpose(spec_out_, axes=[1, -1, 0])
    spec_out_ = np.abs(spec_out_)
    
    spec_tar_ = compute_stft(audio_tar_,
                             hop_length,
                             fft_size,
                             np.sqrt(np.hanning(fft_size+1)[:-1]))
    spec_tar_ = np.transpose(spec_tar_, axes=[1, -1, 0])
    spec_tar_ = np.abs(spec_tar_)
   
    spectral_ = {key:[] for key in ['centroid',
                                    'bandwidth',
                                    'contrast_lows',
                                    'contrast_mids',
                                    'contrast_highs',
                                    'rolloff',
                                    'flatness',
                                    'mean_mape_spectral',
                                   ]}
        
    centroid_mean_ = []
    centroid_std_ = []
    bandwidth_mean_ = []
    bandwidth_std_ = []
    contrast_l_mean_ = []
    contrast_l_std_ = []
    contrast_m_mean_ = []
    contrast_m_std_ = []
    contrast_h_mean_ = []
    contrast_h_std_ = []
    rolloff_mean_ = []
    rolloff_std_ = []
    flatness_mean_ = []

    for ch in range(channels):
        tar = spec_tar_[ch]
        out = spec_out_[ch]

        tar_sc = librosa.feature.spectral_centroid(y=None, sr=sr, S=tar,
                             n_fft=fft_size, hop_length=hop_length)

        out_sc = librosa.feature.spectral_centroid(y=None, sr=sr, S=out,
                             n_fft=fft_size, hop_length=hop_length)

        tar_bw = librosa.feature.spectral_bandwidth(y=None, sr=sr, S=tar,
                                                    n_fft=fft_size, hop_length=hop_length, 
                                                    centroid=tar_sc, norm=True, p=2)

        out_bw = librosa.feature.spectral_bandwidth(y=None, sr=sr, S=out,
                                                    n_fft=fft_size, hop_length=hop_length, 
                                                    centroid=out_sc, norm=True, p=2)
        
        # l = 0-250, m = 1-2-3 = 250 - 2000, h = 2000 - SR/2
        tar_ct = librosa.feature.spectral_contrast(y=None, sr=sr, S=tar,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   fmin=250.0, n_bands=4, quantile=0.02, linear=False)

        out_ct = librosa.feature.spectral_contrast(y=None, sr=sr, S=out,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   fmin=250.0, n_bands=4, quantile=0.02, linear=False)

        tar_ro = librosa.feature.spectral_rolloff(y=None, sr=sr, S=tar,
                                                  n_fft=fft_size, hop_length=hop_length, 
                                                  roll_percent=0.85)

        out_ro = librosa.feature.spectral_rolloff(y=None, sr=sr, S=out,
                                                  n_fft=fft_size, hop_length=hop_length, 
                                                  roll_percent=0.85)
        
        tar_ft = librosa.feature.spectral_flatness(y=None, S=tar,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   amin=1e-10, power=2.0)

        out_ft = librosa.feature.spectral_flatness(y=None, S=out,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   amin=1e-10, power=2.0)
        
        eps = 1e-0
        N = 40
        mean_sc_tar, std_sc_tar = get_running_stats(tar_sc.T+eps, [0], N=N)
        mean_sc_out, std_sc_out = get_running_stats(out_sc.T+eps, [0], N=N)
        
        assert np.isnan(mean_sc_tar).any() == False, f'NAN values mean_sc_tar'
        assert np.isnan(mean_sc_out).any() == False, f'NAN values mean_sc_out'
        
        mean_bw_tar, std_bw_tar = get_running_stats(tar_bw.T+eps, [0], N=N)
        mean_bw_out, std_bw_out = get_running_stats(out_bw.T+eps, [0], N=N)
        
        assert np.isnan(mean_bw_tar).any() == False, f'NAN values tar mean bw'
        assert np.isnan(mean_bw_out).any() == False, f'NAN values out mean bw'
        
        mean_ct_tar, std_ct_tar = get_running_stats(tar_ct.T, list(range(tar_ct.shape[0])), N=N)
        mean_ct_out, std_ct_out = get_running_stats(out_ct.T, list(range(out_ct.shape[0])), N=N)
        
        assert np.isnan(mean_ct_tar).any() == False, f'NAN values tar mean ct'
        assert np.isnan(mean_ct_out).any() == False, f'NAN values out mean ct'
        
        mean_ro_tar, std_ro_tar = get_running_stats(tar_ro.T+eps, [0], N=N)
        mean_ro_out, std_ro_out = get_running_stats(out_ro.T+eps, [0], N=N)
        
        assert np.isnan(mean_ro_tar).any() == False, f'NAN values tar mean ro'
        assert np.isnan(mean_ro_out).any() == False, f'NAN values out mean ro'
        
        mean_ft_tar, std_ft_tar = get_running_stats(tar_ft.T, [0], N=40) # If flatness mean error is too large, increase N. e.g. 100, 1000
        mean_ft_out, std_ft_out = get_running_stats(out_ft.T, [0], N=40)
        
        mape_mean_sc = sklearn.metrics.mean_absolute_percentage_error(mean_sc_tar[0], mean_sc_out[0])
        
        mape_mean_bw = sklearn.metrics.mean_absolute_percentage_error(mean_bw_tar[0], mean_bw_out[0])
        
        mape_mean_ct_l = sklearn.metrics.mean_absolute_percentage_error(mean_ct_tar[0], mean_ct_out[0])
        
        mape_mean_ct_m = sklearn.metrics.mean_absolute_percentage_error(np.mean(mean_ct_tar[1:4], axis=0),
                                                                        np.mean(mean_ct_out[1:4], axis=0))

        mape_mean_ct_h = sklearn.metrics.mean_absolute_percentage_error(mean_ct_tar[-1], mean_ct_out[-1])
   
        mape_mean_ro = sklearn.metrics.mean_absolute_percentage_error(mean_ro_tar[0], mean_ro_out[0])
        
        mape_mean_ft = sklearn.metrics.mean_absolute_percentage_error(mean_ft_tar[0], mean_ft_out[0])
        
        centroid_mean_.append(mape_mean_sc)
        bandwidth_mean_.append(mape_mean_bw)
        contrast_l_mean_.append(mape_mean_ct_l)
        contrast_m_mean_.append(mape_mean_ct_m)
        contrast_h_mean_.append(mape_mean_ct_h)
        rolloff_mean_.append(mape_mean_ro)
        flatness_mean_.append(mape_mean_ft)

    spectral_['centroid'] = np.mean(centroid_mean_)
    spectral_['bandwidth'] = np.mean(bandwidth_mean_)
    spectral_['contrast_lows'] = np.mean(contrast_l_mean_)
    spectral_['contrast_mids'] = np.mean(contrast_m_mean_)
    spectral_['contrast_highs'] = np.mean(contrast_h_mean_)
    spectral_['rolloff'] = np.mean(rolloff_mean_)
    spectral_['flatness'] = np.mean(flatness_mean_)
    spectral_['mean_mape_spectral'] = np.mean([np.mean(centroid_mean_),
                                      np.mean(bandwidth_mean_),
                                      np.mean(contrast_l_mean_),
                                      np.mean(contrast_m_mean_),
                                      np.mean(contrast_h_mean_),
                                      np.mean(rolloff_mean_),
                                      np.mean(flatness_mean_),
                                     ])

    return spectral_


# PANNING 

def get_SPS(x, n_fft=2048, hop_length=1024, smooth=False, frames=False):
    
    """
    Computes Stereo Panning Spectrum (SPS) and similarity measure (phi)
    
    See: 
    Tzanetakis, George, Randy Jones, and Kirk McNally. "Stereo Panning Features for Classifying Recording Production Style." ISMIR. 2007.
    
    Args:
        x: input audio array
        n_fft: fft size
        hop_length: fft hop
        smooth: Applies smoothing filter to SPS
        frames: SPS is calculated in a frame-wise manner

    Returns:
        SPS_mean, phi_mean mean arrays
        SPS, phi arrays
    """ 
    
    
    x = np.copy(x)
    eps = 1e-20
        
    audio_D = compute_stft(x,
                 hop_length,
                 n_fft,
                 np.sqrt(np.hanning(n_fft+1)[:-1]))
    
    audio_D_l = np.abs(audio_D[:, 0, :] + eps)
    audio_D_r = np.abs(audio_D[:, 1, :] + eps)
    
    phi = 2 * (np.abs(audio_D_l*np.conj(audio_D_r)))/(np.abs(audio_D_l)**2+np.abs(audio_D_r)**2)
    
    phi_l = np.abs(audio_D_l*np.conj(audio_D_r))/(np.abs(audio_D_l)**2)
    phi_r = np.abs(audio_D_r*np.conj(audio_D_l))/(np.abs(audio_D_r)**2)
    delta = phi_l - phi_r
    delta_ = np.sign(delta)
    SPS = (1-phi)*delta_
    
    phi_mean = np.mean(phi, axis=0)
    if smooth:
        phi_mean = scipy.signal.savgol_filter(phi_mean, 501, 1, mode='mirror')
    
    SPS_mean = np.mean(SPS, axis=0)
    if smooth:
        SPS_mean = scipy.signal.savgol_filter(SPS_mean, 501, 1, mode='mirror')
        

    return SPS_mean, phi_mean, SPS, phi

def get_panning_rms_frame(sps_frame, freqs=[0,22050], sr=44100, n_fft=2048):
    """
    Computes Stereo Panning Spectrum RMS energy within a specifc frequency band.
    
    Args:
        sps_frame: sps frame
        freqs: frequency band
        sr: sampling rate
        n_fft: fft size

    Returns:
        p_rms SPS rms energy
    """ 
    
    
    idx1 = freqs[0]
    idx2 = freqs[1]

    f1 = int(np.floor(idx1*n_fft/sr))
    f2 = int(np.floor(idx2*n_fft/sr))
    
    p_rms = np.sqrt((1/(f2-f1)) * np.sum(sps_frame[f1:f2]**2))
    
    return p_rms

def get_panning_rms(sps, freqs=[[0, 22050]], sr=44100, n_fft=2048):
    """
    Computes Stereo Panning Spectrum RMS energy within a specifc frequency band.
    
    Args:
        sps: sps
        freqs: frequency band
        sr: sampling rate
        n_fft: fft size

    Returns:
        p_rms SPS rms energy array
    """ 
    
    p_rms = []
    for frame in sps:
        p_rms_ = []
        for f in freqs:
            rms = get_panning_rms_frame(frame, freqs=f, sr=sr, n_fft=n_fft)
            p_rms_.append(rms)
        p_rms.append(p_rms_)
    
    return np.asarray(p_rms)


def compute_panning_features(audio_out, audio_tar, sr, fft_size=4096, hop_length=1024):
    """
    Computes panning features' mape error using a running mean

    Args:
        audio_out: automix audio (output of models)
        audio_tar: target audio
        sr: sampling rate
        fft_size: fft_size size
        hop_length: fft hop size

    Returns:
        panning_ dictionary
    """ 
     
    audio_out_ = pyln.normalize.peak(audio_out, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar, -1.0)
    
    panning_ = {}
                               
    freqs=[[0, sr//2], [0, 250], [250, 2500], [2500, sr//2]]  
    
    _, _, sps_frames_tar, _ = get_SPS(audio_tar_, n_fft=fft_size,
                                  hop_length=hop_length,
                                  smooth=True, frames=True)
    
    _, _, sps_frames_out, _ = get_SPS(audio_out_, n_fft=fft_size,
                                      hop_length=hop_length,
                                      smooth=True, frames=True)


    p_rms_tar = get_panning_rms(sps_frames_tar,
                    freqs=freqs,
                    sr=sr,
                    n_fft=fft_size)
    
    p_rms_out = get_panning_rms(sps_frames_out,
                    freqs=freqs,
                    sr=sr,
                    n_fft=fft_size)
    
    # to avoid num instability, deletes frames with zero rms from target
    if np.min(p_rms_tar) == 0.0:
        id_zeros = np.where(p_rms_tar.T[0] == 0)
        p_rms_tar_ = []
        p_rms_out_ = []
        for i in range(len(freqs)):
            temp_tar = np.delete(p_rms_tar.T[i], id_zeros)
            temp_out = np.delete(p_rms_out.T[i], id_zeros)
            p_rms_tar_.append(temp_tar)
            p_rms_out_.append(temp_out)
        p_rms_tar_ = np.asarray(p_rms_tar_)
        p_rms_tar = p_rms_tar_.T
        p_rms_out_ = np.asarray(p_rms_out_)
        p_rms_out = p_rms_out_.T
    
    N = 40 
    
    mean_tar, std_tar = get_running_stats(p_rms_tar, freqs, N=N)
    mean_out, std_out = get_running_stats(p_rms_out, freqs, N=N)
    
    panning_['panning_rms_total'] = sklearn.metrics.mean_absolute_percentage_error(mean_tar[0], mean_out[0])
    panning_['panning_rms_lows'] = sklearn.metrics.mean_absolute_percentage_error(mean_tar[1], mean_out[1])
    panning_['panning_rms_mids'] = sklearn.metrics.mean_absolute_percentage_error(mean_tar[2], mean_out[2])
    panning_['panning_rms_highs'] = sklearn.metrics.mean_absolute_percentage_error(mean_tar[3], mean_out[3])

    panning_['mean_mape_panning'] = np.mean([panning_['panning_rms_total'],
                                      panning_['panning_rms_lows'],
                                      panning_['panning_rms_mids'],
                                      panning_['panning_rms_highs'],
                                     ])
    
    return panning_


# DYNAMIC

def get_rms_dynamic_crest(x, frame_length, hop_length):
    """
    computes rms level, dynamic spread and crest factor
    
    See: Ma, Zheng, et al. "Intelligent multitrack dynamic range compression." Journal of the Audio Engineering Society 
    
    Args:
        x: input audio array
        frame_length: frame size
        hop_length: frame hop

    Returns:
        rms, dynamic_spread, crest arrays
    """ 
    
    rms = []
    dynamic_spread = []
    crest = []
    for ch in range(x.shape[-1]):
        frames = librosa.util.frame(x[:, ch], frame_length=frame_length, hop_length=hop_length)
        rms_ = []
        dynamic_spread_ = []
        crest_ = []
        for i in frames.T:
            x_rms = amp_to_db(np.sqrt(np.sum(i**2)/frame_length))   
            x_d = np.sum(amp_to_db(np.abs(i)) - x_rms)/frame_length
            x_c = amp_to_db(np.max(np.abs(i)))/x_rms
            
            rms_.append(x_rms)
            dynamic_spread_.append(x_d)
            crest_.append(x_c)
        rms.append(rms_)
        dynamic_spread.append(dynamic_spread_)
        crest.append(crest_)
        
    rms = np.asarray(rms)
    dynamic_spread = np.asarray(dynamic_spread)
    crest = np.asarray(crest)  
    
    rms = np.mean(rms, axis=0)
    dynamic_spread = np.mean(dynamic_spread, axis=0)
    crest = np.mean(crest, axis=0)
    
    rms = np.expand_dims(rms, axis=0)
    dynamic_spread = np.expand_dims(dynamic_spread, axis=0)
    crest = np.expand_dims(crest, axis=0)
    
    return rms, dynamic_spread, crest

def lowpassFiltering(x, f0, sr):
    """
    low pass filters

    Args:
        x: input audio array
        f0:cut-off frequency
        sr: sampling rate

    Returns:
        filtered audio array
    """ 
    

    b1, a1 = scipy.signal.butter(4, f0/(sr/2),'lowpass')
    x_f = []
    for ch in range(x.shape[-1]):
        x_f_ = scipy.signal.filtfilt(b1, a1, x[:, ch]).copy(order='F')
        x_f.append(x_f_)
    return np.asarray(x_f).T  


def compute_dynamic_features(audio_out, audio_tar, sr, fft_size=4096, hop_length=1024):
    """
    Computes dynamic features' mape error using a running mean

    Args:
        audio_out: automix audio (output of models)
        audio_tar: target audio
        sr: sampling rate
        fft_size: fft_size size
        hop_length: fft hop size

    Returns:
        spectral_ dictionary
    """ 
    

    audio_out_ = pyln.normalize.peak(audio_out, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar, -1.0)
    
    dynamic_ = {}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
    
        rms_tar, dyn_tar, crest_tar = get_rms_dynamic_crest(audio_tar_, fft_size, hop_length)
        rms_out, dyn_out, crest_out = get_rms_dynamic_crest(audio_out_, fft_size, hop_length)
        
    N = 40
    
    eps = 1e-10
    
    rms_tar = (-1*rms_tar) + 1.0
    rms_out = (-1*rms_out) + 1.0
    dyn_tar = (-1*dyn_tar) + 1.0
    dyn_out = (-1*dyn_out) + 1.0
   
    mean_rms_tar, std_rms_tar = get_running_stats(rms_tar.T, [0], N=N)
    mean_rms_out, std_rms_out = get_running_stats(rms_out.T, [0], N=N)
    
    mean_dyn_tar, std_dyn_tar = get_running_stats(dyn_tar.T, [0], N=N)
    mean_dyn_out, std_dyn_out = get_running_stats(dyn_out.T, [0], N=N)
    
    mean_crest_tar, std_crest_tar = get_running_stats(crest_tar.T, [0], N=N)
    mean_crest_out, std_crest_out = get_running_stats(crest_out.T, [0], N=N)
        
    dynamic_['rms_level'] = sklearn.metrics.mean_absolute_percentage_error(mean_rms_tar, mean_rms_out)
    dynamic_['dynamic_spread'] = sklearn.metrics.mean_absolute_percentage_error(mean_dyn_tar, mean_dyn_out)
    dynamic_['crest_factor'] = sklearn.metrics.mean_absolute_percentage_error(mean_crest_tar, mean_crest_out)

    dynamic_['mean_mape_dynamic'] = np.mean([dynamic_['rms_level'],
                                      dynamic_['dynamic_spread'],
                                      dynamic_['crest_factor'],
                                     ])
    
    return dynamic_


def get_features(target, output, sr=44100,
                 frame_size=17640, frame_hop=8820,
                 fft_size=4096, fft_hop=1024):
    
    """
    Computes all features' mape error using a running mean

    Args:
        output: automix audio (output of models)
        target: target audio
        sr: sampling rate
        frame_size: frame size for loudness computations, ideally larger than 400ms (LUFS) 
        hop_size: frame hop
        fft_size: fft_size size
        hop_length: fft hop size

    Returns:
        features dictionary
    """ 
    
    # Finds the starting and ending silences in target_mix and trims both mixes

    features = {}
    x = target.T
    y = output.T
    
    x, idx = librosa.effects.trim(x.T, top_db=45, frame_length=4096, hop_length=1024)
    x = x.T
    y = y[idx[0]:idx[1],:]
    assert x.shape == y.shape
    
    # Compute Loudness features

    loudness_features = compute_loudness_features(y, x, sr, frame_size, frame_hop)
    for k, i in loudness_features.items():
        features[k] = i
        
    # Compute spectral features

    n_channels = x.shape[1]
    spectral_features = compute_spectral_features(y, x, sr, fft_size, fft_hop, n_channels)
    for k, i in spectral_features.items():
        features[k] = i
        
    # Computes panning features

    panning_features = compute_panning_features(y, x, sr, fft_size, fft_hop)  

    for k, i in panning_features.items():
        features[k] = i
        
    # Computes dynamic features

    dynamic_features = compute_dynamic_features(y, x, sr, fft_size, fft_hop)
    for k, i in dynamic_features.items():
        features[k] = i
        
    return features
