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
A production-ready web application that separates audio stems and converts them to MIDI using state-of-the-art deep learning models. Built with Gradio and deployed on LightningAI, this pipeline provides an intuitive interface for audio processing tasks.

## Technical Requirements

### Dependencies
```bash
pip install gradio>=4.0.0
pip install demucs>=4.0.0
pip install basic-pitch>=0.4.0
pip install torch>=2.0.0 torchaudio>=2.0.0
pip install soundfile>=0.12.1
pip install numpy>=1.26.4
pip install pretty_midi>=0.2.10
```

### File Structure
```
project/
├── app.py                 # Main Gradio interface and processing logic
├── demucs_handler.py      # Audio stem separation handler
├── basic_pitch_handler.py # MIDI conversion handler
├── validators.py          # Audio file validation utilities
└── requirements.txt
```

## Implementation Details

### demucs_handler.py
Handles audio stem separation using the Demucs model:
- Supports mono and stereo input
- Automatic stereo conversion for mono inputs
- Efficient tensor processing with PyTorch
- Proper error handling and logging
- Progress tracking during processing

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
- Optional MIDI conversion
- Persistent file handling
- Progress tracking
- Comprehensive error handling

## Key Features

### Audio Processing
- High-quality stem separation using Demucs
- Support for multiple audio formats
- Automatic audio format conversion
- Efficient memory management
- Progress tracking during processing

### MIDI Conversion
- Accurate note detection
- Polyphonic transcription
- Configurable parameters:
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

# Run application
python app.py
```

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




