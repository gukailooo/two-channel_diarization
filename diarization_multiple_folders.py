import subprocess
import wave
import numpy as np
from pathlib import Path
import os
import re
import difflib
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# === Конвертация аудио в PCM ===
def convert_to_pcm(input_file, output_file):
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === Сохранение отдельного канала ===
def save_wav_channel(folder, wav, channel):
    nch = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())
    typ = {1: np.uint8, 2: np.int16, 4: np.int32}.get(depth)
    if channel >= nch:
        raise ValueError(f"Канал {channel+1} не существует.")
    data = np.frombuffer(sdata, dtype=typ)
    ch_data = data[channel::nch]
    out_path = Path(folder) / f"speaker{channel+1}.wav"
    with wave.open(str(out_path), 'wb') as outwav:
        outwav.setparams((1, depth, wav.getframerate(), 0, 'NONE', 'not compressed'))
        outwav.writeframes(ch_data.tobytes())

# === Проверка на моно ===
def is_mono_audio(file1, file2, tolerance=1e-3):
    y1, sr1 = torchaudio.load(file1)
    y2, sr2 = torchaudio.load(file2)
    if sr1 != sr2:
        return False
    min_len = min(y1.shape[-1], y2.shape[-1])
    y1, y2 = y1[..., :min_len], y2[..., :min_len]
    y1 = y1.mean(dim=0)
    y2 = y2.mean(dim=0)
    corr = torch.corrcoef(torch.stack([y1, y2]))[0,1].item()
    diff = torch.mean(torch.abs(y1 - y2)).item()
    return corr > 0.995 and diff < tolerance

# === Очистка повторяющихся фраз по смыслу ===
def clean_repeated_phrases(segments, similarity_threshold=0.85):
    cleaned = []
    for start, end, speaker, text in segments:
        text = text.strip()
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        new_sentences = []
        last_sentence = ""
        for s in sentences:
            sim = difflib.SequenceMatcher(None, last_sentence, s).ratio()
            if sim < similarity_threshold:
                new_sentences.append(s)
                last_sentence = s
        cleaned_text = '. '.join(new_sentences)

        # Убираем повторяющиеся слова подряд
        cleaned_text = re.sub(r'\b(\w+)(?:\s+\1\b){1,}', r'\1', cleaned_text, flags=re.IGNORECASE)

        # Список паттернов для удаления/замены
        patterns = [
            r'\b(да|ага|угу|поняла|понял|ну|вот|это|сейчас|так|здравствуйте)\b(?:\s+\1\b){1,}',
            r'редактор\s*субтитр\w*\s*[аa]\.?[\s\w]*семкин',
            r'корректор\s*[аa]\.?[\s\w]*егоров',
            r'продолжение следует',
            r'звонок в дверь',
            r'телефонный звонок'
        ]
        for pat in patterns:
            cleaned_text = re.sub(pat, '', cleaned_text, flags=re.IGNORECASE)

        # Убираем лишние пробелы
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()

        # Убираем повторяющийся хвост
        words = cleaned_text.split()
        if len(words) > 10:
            tail = " ".join(words[-6:])
            if cleaned_text.count(tail) > 1:
                cleaned_text = cleaned_text.replace(tail, "", cleaned_text.count(tail) - 1).strip()

        if cleaned_text:
            cleaned.append((start, end, speaker, cleaned_text))
    return cleaned


# === Распознавание с таймкодами ===
def transcribe_with_timestamps(audio_path, speaker_name, asr):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    temp_path = f"temp_{speaker_name}.wav"
    torchaudio.save(temp_path, waveform, sr)
    result = asr(temp_path, chunk_length_s=25, return_timestamps=True)
    segments = []
    for chunk in result["chunks"]:
        start, end = chunk["timestamp"]
        start = start if start is not None else 0.0
        end = end if end is not None else start + 0.5
        text = chunk["text"].strip()
        if text:
            segments.append((start, end, speaker_name, text))
    os.remove(temp_path)
    segments = clean_repeated_phrases(segments)
    return segments

# === Слияние сегментов двух говорящих ===
def merge_dialog_segments(segments1, segments2, overlap_tolerance=1.0):
    merged = sorted(segments1 + segments2, key=lambda x: x[0])
    cleaned = []

    for seg in merged:
        start, end, speaker, text = seg
        if cleaned and start - cleaned[-1][1] < overlap_tolerance and speaker == cleaned[-1][2]:
            combined_text = cleaned[-1][3] + " " + text
            cleaned[-1] = (cleaned[-1][0], max(end, cleaned[-1][1]), speaker, combined_text)
        else:
            cleaned.append(seg)

    cleaned = clean_repeated_phrases(cleaned)
    return cleaned

# === Извлечение метаданных из txt ===
def extract_metadata(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    date = re.search(r"(\d{2}_\d{2}_\d{4})", text)
    time = re.search(r"(\d{2}:\d{2}:\d{2})", text)
    abonent = re.search(r"Объект\s*[-–]?\s*(\d+)", text)
    contact = re.search(r"Абонент\s*[-–]?\s*(\d+)", text)
    return {
        "date": date.group(1) if date else "Неизвестно",
        "time": time.group(1) if time else "Неизвестно",
        "abonent": abonent.group(1) if abonent else "Неизвестно",
        "contact": contact.group(1) if contact else "Неизвестно"
    }

# === Основной блок ===
if __name__ == "__main__":
    target_numbers = []

    input_root = Path(input("Введите путь к папке с файлами: ").strip())
    output_root = input_root / "results"
    request_root = output_root / "request"
    other_root = output_root / "other"
    request_root.mkdir(parents=True, exist_ok=True)
    other_root.mkdir(parents=True, exist_ok=True)

    request_summary_path = request_root / "all_request.txt"
    other_summary_path = other_root / "all_other.txt"

    model_name = "C:/Users/trans/Desktop/diar/models/whisper-large-v3-turbo"
    device = 0 if torch.cuda.is_available() else -1
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda" if device==0 else "cpu")
    processor = WhisperProcessor.from_pretrained(model_name)
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        generate_kwargs={"temperature":0.0,"no_repeat_ngram_size":4}
    )

    for txt_file in input_root.rglob("*.txt"):
        base_name = txt_file.stem
        audio_path = None
        for ext in [".wav", ".mp3", ".flac"]:
            candidate = txt_file.with_suffix(ext)
            if candidate.exists():
                audio_path = candidate
                break
        if not audio_path:
            continue

        meta = extract_metadata(txt_file)
        abonent = meta["abonent"]
        contact = meta["contact"]
        time_clean = meta["time"].replace(":", "-") if meta["time"] != "Неизвестно" else "00-00-00"
        date = meta["date"]
        folder_name = f"{date}-({time_clean})-{contact}"

        category_root = request_root if abonent in target_numbers or contact in target_numbers else other_root
        category_summary_path = request_summary_path if category_root==request_root else other_summary_path

        contact_folder = category_root / contact / folder_name
        contact_folder.mkdir(parents=True, exist_ok=True)
        pcm_file = contact_folder / f"{base_name}_pcm.wav"
        convert_to_pcm(audio_path, str(pcm_file))

        wav = wave.open(str(pcm_file), 'rb')
        for ch in range(wav.getnchannels()):
            save_wav_channel(contact_folder, wav, ch)
        wav.close()

        file1 = str(contact_folder / "speaker1.wav")
        file2 = str(contact_folder / "speaker2.wav")

        if os.path.exists(file2) and not is_mono_audio(file1, file2):
            segments1 = transcribe_with_timestamps(file1, "Абонент_А", asr)
            segments2 = transcribe_with_timestamps(file2, "Абонент_Б", asr)
            all_segments = merge_dialog_segments(segments1, segments2)
        else:
            all_segments = transcribe_with_timestamps(file1, "Абонент", asr)

        dialogue_path = contact_folder / f"dialogue_{date}-({time_clean})-{contact}.txt"
        with open(dialogue_path, "w", encoding="utf-8") as f:
            f.write(f"Дата: {meta['date']}\nВремя: {meta['time']}\nАбонент: {meta['abonent']}\nКонтакт: {meta['contact']}\n\n")
            for _, _, speaker, text in all_segments:
                f.write(f"{speaker}: {text}\n")

        with open(category_summary_path, "a", encoding="utf-8") as f:
            f.write(f"\nДата: {meta['date']}\nВремя: {meta['time']}\nАбонент: {meta['abonent']}\nКонтакт: {meta['contact']}\n\n")
            for _, _, speaker, text in all_segments:
                f.write(f"{speaker}: {text}\n")
            f.write("="*80 + "\n")

        contact_summary_path = contact_folder.parent / f'{contact}_all.txt'
        with open(contact_summary_path, 'a', encoding='utf-8') as f:
            f.write(f"\nДата: {meta['date']}\nВремя: {meta['time']}\nАбонент: {meta['abonent']}\nКонтакт: {meta['contact']}\n\n")
            for _, _, speaker, text in all_segments:
                f.write(f"{speaker}: {text}\n")
            f.write("\n" + '-'*100 + "\n" + "="*80 + "\n")

    print("\n📘 Все транскрипции сохранены по номерам и категориям.")
