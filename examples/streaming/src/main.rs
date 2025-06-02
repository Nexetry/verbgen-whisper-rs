use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const TARGET_SAMPLE_RATE: u32 = 16000;
const CHUNK_DURATION_SECONDS: f32 = 3.0;
const CHUNK_SIZE_SAMPLES: usize = (TARGET_SAMPLE_RATE as f32 * CHUNK_DURATION_SECONDS) as usize;

struct AudioResampler {
    input_rate: u32,
    output_rate: u32,
    buffer: Vec<f32>,
    phase: f32,
}

impl AudioResampler {
    fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            input_rate,
            output_rate,
            buffer: Vec::new(),
            phase: 0.0,
        }
    }

    fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        if self.input_rate == self.output_rate {
            return input.to_vec();
        }

        let ratio = self.input_rate as f32 / self.output_rate as f32;
        let mut output = Vec::new();

        for &sample in input {
            self.buffer.push(sample);

            while self.phase < self.buffer.len() as f32 - 1.0 {
                let index = self.phase as usize;
                let frac = self.phase - index as f32;

                // Linear interpolation
                let interpolated =
                    self.buffer[index] * (1.0 - frac) + self.buffer[index + 1] * frac;
                output.push(interpolated);

                self.phase += ratio;
            }
        }

        // Keep last sample for next iteration
        if !self.buffer.is_empty() {
            let last_sample = *self.buffer.last().unwrap();
            self.buffer.clear();
            self.buffer.push(last_sample);
            self.phase -= self.buffer.len() as f32 - 1.0;
        }

        output
    }
}

struct AudioChunker {
    buffer: Vec<f32>,
    chunk_size: usize,
}

impl AudioChunker {
    fn new(chunk_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            chunk_size,
        }
    }

    fn add_samples(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.buffer.extend_from_slice(samples);

        let mut chunks = Vec::new();
        while self.buffer.len() >= self.chunk_size {
            let chunk = self.buffer.drain(0..self.chunk_size).collect();
            chunks.push(chunk);
        }

        chunks
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 Starting real-time audio transcription...");
    println!("📊 Target sample rate: {}Hz", TARGET_SAMPLE_RATE);
    println!("⏱️  Chunk duration: {} seconds", CHUNK_DURATION_SECONDS);
    println!("📦 Chunk size: {} samples", CHUNK_SIZE_SAMPLES);

    // Initialize Whisper
    println!("🤖 Loading Whisper model...");
    let ctx = WhisperContext::new_with_params(
        "models/ggml-base.en.bin", // You'll need to download this model
        WhisperContextParameters::default(),
    )?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Set up audio system
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    println!("🔧 Using input device: {}", device.name()?);

    let config = device.default_input_config()?;
    println!("🔧 Input config: {:?}", config);

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    // Create audio processing pipeline
    let mut resampler = AudioResampler::new(sample_rate, TARGET_SAMPLE_RATE);
    let mut chunker = AudioChunker::new(CHUNK_SIZE_SAMPLES);

    // Channel for sending audio chunks to transcription
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<f32>>();

    // Shared state for audio capture
    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buffer_clone = audio_buffer.clone();

    // Start audio capture
    let stream = match config.sample_format() {
        SampleFormat::F32 => create_stream::<f32>(&device, &config.into(), buffer_clone, channels)?,
        SampleFormat::I16 => create_stream::<i16>(&device, &config.into(), buffer_clone, channels)?,
        SampleFormat::U16 => create_stream::<u16>(&device, &config.into(), buffer_clone, channels)?,
        _ => return Err("Unsupported sample format".into()),
    };

    stream.play()?;
    println!("🎵 Audio capture started!");

    // Audio processing thread
    let tx_clone = tx.clone();
    thread::spawn(move || {
        let mut last_process = Instant::now();
        let process_interval = Duration::from_millis(100); // Process every 100ms

        loop {
            if last_process.elapsed() >= process_interval {
                // Get accumulated audio samples
                let samples = {
                    let mut buffer = audio_buffer.lock().unwrap();
                    if buffer.is_empty() {
                        thread::sleep(Duration::from_millis(10));
                        continue;
                    }
                    let samples = buffer.clone();
                    buffer.clear();
                    samples
                };

                // Resample to 16kHz
                let resampled = resampler.resample(&samples);

                // Check for complete chunks
                let chunks = chunker.add_samples(&resampled);

                // Send chunks for transcription
                for chunk in chunks {
                    if let Err(e) = tx_clone.send(chunk) {
                        eprintln!("Error sending chunk: {}", e);
                        break;
                    }
                }

                last_process = Instant::now();
            }

            thread::sleep(Duration::from_millis(1));
        }
    });

    // Transcription loop
    let mut chunk_count = 0;
    println!("🎯 Ready for transcription! Speak into your microphone...");
    println!("📝 Press Ctrl+C to stop\n");

    // Set up Ctrl+C handler
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let running_clone = running.clone();
    ctrlc::set_handler(move || {
        println!("\n🛑 Stopping transcription...");
        running_clone.store(false, std::sync::atomic::Ordering::SeqCst);
    })?;

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        tokio::select! {
            chunk_opt = rx.recv() => {
                if let Some(chunk) = chunk_opt {
                    chunk_count += 1;
                    let start_time = Instant::now();

                    // Convert to the format whisper expects
                    let audio_data: Vec<f32> = chunk.iter().cloned().collect();

                    // Transcribe the chunk
                    match ctx.full(params.clone(), &audio_data) {
                        Ok(mut state) => {
                            let num_segments = state.full_n_segments()?;
                            if num_segments > 0 {
                                let mut transcription = String::new();
                                for i in 0..num_segments {
                                    let segment = state.full_get_segment_text(i)?;
                                    transcription.push_str(&segment);
                                }

                                let processing_time = start_time.elapsed();

                                if !transcription.trim().is_empty() {
                                    println!("🎤 Chunk #{}: \"{}\" (processed in {:?})",
                                           chunk_count, transcription.trim(), processing_time);
                                } else {
                                    println!("🔇 Chunk #{}: [silence detected] (processed in {:?})",
                                           chunk_count, processing_time);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ Transcription error for chunk #{}: {}", chunk_count, e);
                        }
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                // Keep the loop alive
            }
        }
    }

    println!(
        "👋 Transcription stopped. Processed {} chunks total.",
        chunk_count
    );
    Ok(())
}

fn create_stream<T>(
    device: &Device,
    config: &StreamConfig,
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    channels: usize,
) -> Result<Stream, Box<dyn std::error::Error>>
where
    T: cpal::Sample + Into<f32>,
{
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // Convert samples to f32 and handle multi-channel audio
            let samples: Vec<f32> = data
                .chunks(channels)
                .map(|frame| {
                    // Average channels to mono
                    let sum: f32 = frame.iter().map(|&s| s.into()).sum();
                    sum / channels as f32
                })
                .collect();

            // Add to buffer
            if let Ok(mut buffer) = audio_buffer.lock() {
                buffer.extend(samples);
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;

    Ok(stream)
}
