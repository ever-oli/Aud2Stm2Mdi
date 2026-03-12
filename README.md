---
title: Aud2Stm2Mdi
emoji: "🎵"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
python_version: 3.10.13
app_file: app.py
pinned: false
---

# Multi-Model Stem Separation and MIDI Transcription

## Project Overview
A production-ready web application for testing different stem-separation models against different audio-to-MIDI transcription models. Built with Gradio and deployed on LightningAI, the app now supports multiple separator families plus selectable AMT backends in the same UI.

## Supported Model Matrix

### Separator Models
- Demucs family: `htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `hdemucs_mmi`, `mdx`, `mdx_extra`, `mdx_q`, `mdx_extra_q`
- ZFTurbo-backed MSST family: `msst_bs_roformer`, `msst_scnet`, `msst_mdx23c`

### Transcription Models
- `basic_pitch`
- `mt3`

## Technical Requirements

### Dependencies
```bash
pip install -r requirements.txt
```

Optional backend runtimes for RoFormer, SCNet, MDX23C, and quantized MDX variants:
```bash
pip install -r requirements.txt -r requirements-separation-backends.txt
```

Optional AMT runtime for MT3:
```bash
pip install -r requirements.txt -r requirements-amt-backends.txt
```

Full install for all separator and AMT backends:
```bash
pip install -r requirements.txt -r requirements-separation-backends.txt -r requirements-amt-backends.txt
```

### File Structure
```
project/
├── app.py                 # Main Gradio interface and processing logic
├── amt_backends.py        # AMT backend adapters (Basic Pitch, MT3)
├── amt_registry.py        # AMT model registry
├── demucs_handler.py      # Demucs inference wrapper
├── demucs_models.py       # Official Demucs checkpoint registry
├── separator_backends.py  # Separator backend adapters (Demucs, MSST)
├── separator_registry.py  # Separator model registry
├── basic_pitch_handler.py # MIDI conversion handler
├── requirements-amt-backends.txt         # Optional MT3 runtime deps
├── requirements-separation-backends.txt  # Optional non-Demucs backend deps
├── validators.py          # Audio file validation utilities
└── requirements.txt
```

## Implementation Details

### demucs_handler.py
Handles Demucs-family separation using official Demucs checkpoints:
- Supports mono and stereo input
- Automatic stereo conversion for mono inputs
- Supports multiple Demucs model variants (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `hdemucs_mmi`, `mdx`, `mdx_extra`, `mdx_q`, `mdx_extra_q`)
- Efficient tensor processing with PyTorch
- Proper error handling and logging
- Progress tracking during processing

### separator_backends.py / separator_registry.py
Manages the full separator model matrix used by the app:
- Demucs-family models
- ZFTurbo-backed MSST models (`msst_bs_roformer`, `msst_scnet`, `msst_mdx23c`)
- Shared separator contract for app and sweep scripts
- Per-model stem lists so both 4-stem and 6-stem outputs work cleanly

### amt_backends.py
Manages MIDI conversion using pluggable transcription backends:
- Spotify Basic Pitch for lightweight, tunable transcription
- MT3 via `mt3-infer` for heavier multi-instrument transcription
- Shared output contract for the Gradio app

### basic_pitch_handler.py
Manages MIDI conversion using Spotify's Basic Pitch:
- Optimized parameters for music transcription
- Support for polyphonic audio
- Pitch bend detection
- Configurable note duration and frequency ranges
- Robust MIDI file generation

### validators.py
Provides comprehensive audio file validation:
- Format verification (WAV, MP3, FLAC)
- File size limits (30MB default)
- Sample rate validation (8kHz-48kHz)
- Audio integrity checking
- Detailed error reporting

### app.py
Main application interface featuring:
- Clean, intuitive Gradio UI
- Multi-file upload support
- Separator model selection
- Per-model stem selection
- Optional MIDI conversion with selectable AMT backend
- Persistent file handling
- Progress tracking
- Comprehensive error handling

## Key Features

### Audio Processing
- High-quality stem separation across multiple model families
- Demucs-family separators plus ZFTurbo-backed RoFormer / SCNet / MDX23C
- Per-model stem selection in the Gradio UI
- Support for multiple audio formats
- Automatic audio format conversion
- Efficient memory management
- Progress tracking during processing

### MIDI Conversion
- Multiple transcription backends
- Basic Pitch
- MT3
- Accurate note detection
- Polyphonic transcription
- Configurable Basic Pitch parameters for thresholding and note cleanup

### User Interface
- Simple, intuitive design
- Real-time processing feedback
- Preview capabilities
- File download options

## Deployment

### Local Development
```bash
# Clone repository
git clone https://github.com/eyov7/Aud2Stm2Mdi.git

# Install dependencies
pip install -r requirements.txt

# Optional: prepare experimental separator backend runtimes
pip install -r requirements-separation-backends.txt

# Optional: enable the MT3 MIDI backend
pip install -r requirements-amt-backends.txt

# Run application
python app.py
```

The optional separator backend requirements enable the ZFTurbo-backed RoFormer, SCNet, and MDX23C integrations used alongside the Demucs family in the app and sweep scripts. The optional AMT backend requirements enable MT3 so the transcription side can be compared against Basic Pitch.

### Compare Demucs Models
```bash
python scripts/run_demucs_model_sweep.py path/to/audio.wav --models all
```

This is the Demucs-family-only sweep. It writes model-specific stem folders plus a `summary.json` file under `demucs_sweeps/<audio-name>/`.

### Compare Separator Models
```bash
python scripts/run_separator_model_sweep.py path/to/audio.wav --models all
```

This runs the full separator registry across Demucs plus the ZFTurbo-backed RoFormer / SCNet / MDX23C entries.

### Compare AMT Models
```bash
python scripts/run_amt_model_sweep.py path/to/audio.wav --models all
```

This compares the registered AMT backends directly without first running stem separation.

### Compare Full Separation -> MIDI Pipeline
```bash
python scripts/run_full_pipeline_sweep.py path/to/audio.wav --separator-models all --amt-models all --stems all
```

This is the main end-to-end matrix run. It executes every registered separator model, transcribes every produced stem with every AMT backend, and writes a single `summary.json` plus model-specific outputs under `pipeline_sweeps/<audio-name>/`.

### Lightning.ai Deployment
1. Create new Lightning App
2. Upload project files
3. Configure compute instance (CPU or GPU)
4. Deploy

## Error Handling
Implemented comprehensive error handling for:
- Invalid file formats
- File size limits
- Processing failures
- Memory constraints
- File system operations
- Model inference errors


## Production Features
- Robust file validation
- Persistent storage management
- Proper error logging
- Progress tracking
- Clean user interface
- Download capabilities
- Multi-format support

## Limitations
- Maximum file size: 30MB
- Supported formats: WAV, MP3, FLAC
- Single file processing (no batch)
- CPU-only processing by default

## Notes
- Ensure proper audio codec support
- Monitor system resources
- Regular temporary file cleanup
- Consider implementing rate limiting
- Add user session management

## Closing Note
This implementation is currently running successfully on Lightning.ai, providing reliable audio stem separation and MIDI conversion capabilities through an intuitive web interface.
