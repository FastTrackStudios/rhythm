#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "adtof-plus-drum-transcription @ git+https://github.com/xavriley/adtof_plus_drum_transcription.git@58c1eb431cddc2c10530f4f369f36825677b78ec",
#   "mdx23c-drum-separation @ git+https://github.com/xavriley/mdx23c-drum-separation.git@80ae44e3d2c091e7413512bfb2a2ec7c989a095c",
#   "adtof-pytorch @ git+https://github.com/xavriley/ADTOF-pytorch.git@85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9",
#   "torch==2.8.0", "pretty-midi", "soundfile",
# ]
# ///
"""Run MDX23C kit separation and ADTOF+ transcription on Crescendum stems."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import pretty_midi
import soundfile
import torch

PIECES = {
    35: "kick", 38: "snare", 42: "hh_closed", 46: "hh_open",
    47: "toms", 49: "crash", 51: "ride",
}


def main():
    cache = Path.home() / ".cache" / "fts-rhythm"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stems", type=Path, default=cache / "crescendum-stems")
    parser.add_argument("--input", choices=["drum_bus", "ohs"], default="drum_bus")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--song", help="Select one song")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the driver library path or pass --device cpu")
    model_cache = Path(os.environ.get("FTS_RHYTHM_MODEL_DIR") or cache / "models")
    os.environ["XDG_CACHE_HOME"] = str(model_cache.resolve())
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from adtof_plus_drum_transcription import transcribe_audio_file

    files = sorted(args.stems.glob("*/gt_onsets.json"))
    if args.song:
        files = [path for path in files if path.parent.name.strip() == args.song]
    if not files:
        parser.error(f"no matching exported songs under {args.stems}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    songs = {}
    total_audio = total_elapsed = 0.0
    with tempfile.TemporaryDirectory(dir=args.out.resolve().parent) as scratch:
        tempfile.tempdir = scratch
        audio = Path(scratch) / "input.wav"
        midi = Path(scratch) / "output.mid"
        for path in files:
            song = json.loads(path.read_text())["song"]
            start = perf_counter()
            waveform, rate = soundfile.read(path.parent / f"{args.input}.wav", dtype="float32")
            duration = len(waveform) / rate
            peak = abs(waveform).max()
            if peak > 0:
                waveform *= min(32.0, 0.95 / peak)
            soundfile.write(audio, waveform, rate, subtype="FLOAT")
            transcribe_audio_file(audio, midi, input_is_mix=False)
            notes = pretty_midi.PrettyMIDI(str(midi))
            songs[song] = [
                {"piece": PIECES[note.pitch], "at": note.start, "velocity": note.velocity}
                for instrument in notes.instruments
                if instrument.is_drum
                for note in instrument.notes
                if note.pitch in PIECES and 0 <= note.start < duration
            ]
            elapsed = perf_counter() - start
            total_audio += duration
            total_elapsed += elapsed
            print(f"{song} | {len(songs[song])} events | {elapsed / duration:.4f}x", flush=True)

    processor = "gpu" if args.device == "cuda" else "cpu"
    report = {
        "engine": f"adtof-plus-{args.input.replace('_', '-')}",
        f"{processor}_rtf": total_elapsed / total_audio,
        "songs": songs,
    }
    args.out.write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
