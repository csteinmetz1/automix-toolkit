<div align="center">

# automix-toolkit
Models and datasets for training deep learning automatic mixing models

</div>

# Setup

```
python -m venv env 
source env/bin/activate
pip install --upgrade pip
```

```
pip install -e . 
```

Eventually we want `pip install automix`

Download the demo dataset

Right now we could build a demo dataset using some creative commons mixes
http://c4dm.eecs.qmul.ac.uk/multitrack/MixEvaluation/
Or we could build our own dataset using loops from Logic

# Usage


Patch stereo audio tensorboard logging.
When you first run the training script you will hit an error. 
Go to the line indicated and replace with the following function.

```Python
def audio(tag, tensor, sample_rate=44100):
    chs, samp = tensor.size()
    tensor = make_np(tensor)
    tensor = tensor.squeeze()
    if abs(tensor).max() > 1:
        print("warning: audio amplitude out of range, auto clipped.")
        tensor = tensor.clip(-1, 1)
    # assert(tensor.ndim == 1), 'input tensor should be 1 dimensional.'
    tensor = (tensor * np.iinfo(np.int16).max).astype("<i2")

    import io
    import wave

    fio = io.BytesIO()
    wave_write = wave.open(fio, "wb")
    wave_write.setnchannels(chs)
    wave_write.setsampwidth(2)
    wave_write.setframerate(sample_rate)

    # create an interleaved tensor
    if chs == 2:
        for n in range(samp):
            wave_write.writeframes(tensor[0, n].data)
            wave_write.writeframes(tensor[1, n].data)
    else:
        wave_write.writeframes(tensor.data)

    wave_write.close()
    audio_string = fio.getvalue()
    fio.close()
    audio = Summary.Audio(
        sample_rate=sample_rate,
        num_channels=chs,
        length_frames=tensor.shape[-1],
        encoded_audio_string=audio_string,
        content_type="audio/wav",
    )
    return Summary(value=[Summary.Value(tag=tag, audio=audio)])
```