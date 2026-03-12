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

# Audio Processing Pipeline: Stem Separation and MIDI Conversion

## Project Overview
A production-ready web application that separates audio stems and converts them to MIDI using state-of-the-art deep learning models. Built with Gradio and deployed on LightningAI, this pipeline now supports multiple separator backends plus selectable AMT backends (`Basic Pitch`, `MT3`) in the same UI.

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
├── demucs_handler.py      # Audio stem separation handler
├── demucs_models.py       # Official Demucs checkpoint registry
├── basic_pitch_handler.py # MIDI conversion handler
├── requirements-amt-backends.txt         # Optional MT3 runtime deps
├── requirements-separation-backends.txt  # Optional non-Demucs backend deps
├── validators.py          # Audio file validation utilities
└── requirements.txt
```

## Implementation Details

### demucs_handler.py
Handles audio stem separation using official Demucs checkpoints:
- Supports mono and stereo input
- Automatic stereo conversion for mono inputs
- Supports multiple Demucs model variants (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `hdemucs_mmi`, `mdx`, `mdx_extra`, `mdx_q`, `mdx_extra_q`)
- Efficient tensor processing with PyTorch
- Proper error handling and logging
- Progress tracking during processing

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
- Stem type selection (vocals, drums, bass, other)
- Optional MIDI conversion with selectable AMT backend
- Persistent file handling
- Progress tracking
- Comprehensive error handling

## Key Features

### Audio Processing
- High-quality stem separation using Demucs
- Per-model stem selection in the Gradio UI
- Support for multiple audio formats
- Automatic audio format conversion
- Efficient memory management
- Progress tracking during processing

### MIDI Conversion
- Multiple transcription backends:
  - Basic Pitch
  - MT3
- Accurate note detection
- Polyphonic transcription
- Configurable Basic Pitch parameters:
  - Note duration threshold
  - Frequency range
  - Onset detection sensitivity
  - Frame-level pitch activation

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

The optional separator backend requirements enable the RoFormer, SCNet, MDX23C, and quantized MDX integrations exposed in the app. The optional AMT backend requirement enables MT3 in the main MIDI transcription flow.

### Compare Demucs Models
```bash
python scripts/run_demucs_model_sweep.py path/to/audio.wav --models all
```

This writes model-specific stem folders plus a `summary.json` file under `demucs_sweeps/<audio-name>/`.

### Compare Separator Models
```bash
python scripts/run_separator_model_sweep.py path/to/audio.wav --models all
```

### Compare AMT Models
```bash
python scripts/run_amt_model_sweep.py path/to/audio.wav --models all
```

### Compare Full Separation -> MIDI Pipeline
```bash
python scripts/run_full_pipeline_sweep.py path/to/audio.wav --separator-models all --amt-models all --stems all
```

This runs every registered separator model, transcribes every produced stem with every AMT backend, and writes a single `summary.json` plus model-specific outputs under `pipeline_sweeps/<audio-name>/`.

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
