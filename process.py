from pathlib import Path
import os
from tqdm import tqdm
import time
from urllib.parse import quote_plus
import pysubs2
import torch 
import whisper
from faster_whisper import WhisperModel
from srt2ass import srt2ass

def auto_sub_jp(type_, model, split, method, beam_size, file_name):
    file_type = type_
    model_size = model
    language = 'ja'
    is_split = split
    split_method = method
    sub_style = "default"
    is_vad_filter = "False"
    set_beam_size = beam_size
    file_basename = Path(file_name).stem
    output_dir = Path(file_name).parent.resolve()
    print('Loading model...')

    if model_size == 'large-v3':
        model = whisper.load_model("large-v3")
        is_whisperv3 = True
        torch.cuda.empty_cache()

    else:
        model = WhisperModel(model_size)
        is_whisperv3 = False
        torch.cuda.empty_cache()

    #Transcribe
    if file_type == "video":
        print('Extracting audio from video file...')
        os.system(f'ffmpeg -i {file_name} -f mp3 -ab 192000 -vn {file_basename}.mp3')
        print('Done.')
    tic = time.time()
    print('Transcribe in progress...')

    if is_whisperv3:
        results = model.transcribe(audio = f'{file_name}', language= language, verbose=False)

    else:
        segments, info = model.transcribe(audio = f'{file_name}',
                                            beam_size=set_beam_size,
                                            language=language,
                                            vad_filter=is_vad_filter,
                                            vad_parameters=dict(min_silence_duration_ms=1000))

        # segments is a generator so the transcription only starts when you iterate over it
        # to use pysubs2, the argument must be a segment list-of-dicts
        total_duration = round(info.duration, 2)  # Same precision as the Whisper timestamps.
        results= []
        with tqdm(total=total_duration, unit=" seconds") as pbar:
            for s in segments:
                segment_dict = {'start':s.start,'end':s.end,'text':s.text}
                results.append(segment_dict)
                segment_duration = s.end - s.start
                pbar.update(segment_duration)


    #Time comsumed
    toc = time.time()
    print('Done')
    print(f'Time consumpution {toc-tic}s')
    time_consumtion = toc-tic
    subs = pysubs2.load_from_whisper(results)
    subs.save(f'{output_dir}/{file_basename}.srt')

    ass_sub = srt2ass(f"{output_dir}/{file_basename}.srt", sub_style, is_split,split_method)
    print('ASS subtitle saved as: ' + ass_sub)

    torch.cuda.empty_cache()
    return time_consumtion