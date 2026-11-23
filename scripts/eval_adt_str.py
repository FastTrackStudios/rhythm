#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.8.0", "torchaudio==2.8.0", "transformers==4.57.3",
#   "huggingface-hub>=0.34,<1", "omegaconf", "pretty-midi", "soundfile", "safetensors",
# ]
# ///
"""Evaluate the published ADT_STR checkpoint on exported Crescendum stems."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import pretty_midi
import soundfile
import torch
from huggingface_hub import snapshot_download

MODEL = "Pierfrancesco/adt-str"
REVISION = "a33c5c6b191a4ca1e0f6dc22140947485eb36ce8"
VARIANT = "setting-tau-0.8"
# The upstream tokenizer emits its compact drum-note vocabulary.
PIECES = {
    35: "kick", 36: "kick",
    37: "snare", 38: "snare", 40: "snare",
    41: "toms", 45: "toms", 47: "toms",
    42: "hihat", 43: "hihat", 44: "hihat",
    46: "cymbals", 48: "cymbals", 49: "cymbals", 51: "cymbals",
}


def read_hits(path, duration):
    midi = pretty_midi.PrettyMIDI(str(path))
    return sorted(
        (
            {"piece": PIECES[note.pitch], "at": note.start, "velocity": note.velocity}
            for instrument in midi.instruments
            if instrument.is_drum
            for note in instrument.notes
            if note.pitch in PIECES and 0 <= note.start < duration
        ),
        key=lambda hit: hit["at"],
    )


def main():
    cache = Path.home() / ".cache" / "fts-rhythm"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stems", type=Path, default=cache / "crescendum-stems")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--song", help="Select a single song for a smoke test")
    args = parser.parse_args()

    files = sorted(args.stems.glob("*/gt_onsets.json"))
    if args.song:
        files = [path for path in files if path.parent.name.strip() == args.song]
    if not files:
        parser.error(f"no matching exported songs under {args.stems}")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the driver library path or pass --device cpu")

    model_cache = Path(os.environ.get("FTS_RHYTHM_MODEL_DIR") or cache / "models")
    snapshot = snapshot_download(
        MODEL,
        revision=REVISION,
        cache_dir=model_cache,
        allow_patterns=["*.py", f"{VARIANT}/*"],
    )
    sys.path.insert(0, snapshot)
    from adt_transcriber import ADTTranscriber

    torch.manual_seed(0)
    transcriber = ADTTranscriber.from_pretrained(snapshot, variant=VARIANT, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    songs = {}
    total_audio = total_elapsed = 0.0
    with TemporaryDirectory(dir=args.out.resolve().parent) as output_dir:
        for path in files:
            metadata = json.loads(path.read_text())
            song = metadata["song"]
            start = perf_counter()
            waveform, rate = soundfile.read(path.parent / "drum_bus.wav", dtype="float32")
            duration = len(waveform) / rate
            # Peak normalization matches the upstream evaluation dataset.
            peak = abs(waveform).max()
            if peak > 0:
                waveform /= peak
            midi = transcriber.transcribe(
                torch.from_numpy(waveform), sample_rate=rate, output_dir=output_dir
            )
            songs[song] = read_hits(midi, duration)
            elapsed = perf_counter() - start
            total_audio += duration
            total_elapsed += elapsed
            print(f"{song} | {len(songs[song])} events | {elapsed / duration:.4f}x", flush=True)

    report = {
        "engine": "adt-str",
        f"{'gpu' if args.device == 'cuda' else 'cpu'}_rtf": total_elapsed / total_audio,
        "songs": songs,
    }
    args.out.write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
