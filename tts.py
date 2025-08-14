from google import genai
from google.genai import types
from TTS.api import TTS
import concurrent.futures
import multiprocessing
import threading
import time
import os
import re
from typing import List, Tuple
import numpy as np
from scipy.io import wavfile
import tempfile
from pathlib import Path
import torch

class OptimizedGeminiCoquiPipeline:
    def __init__(self, 
                 tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC",
                 text: str = '',
                 use_gpu: bool = True,
                 max_workers: int = None,
                 chunk_size: int = 500):
        
        self.tts_model_name = tts_model
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.chunk_size = chunk_size
        self.text = text
        
        
        # Initialize TTS model (single instance for reuse)
        print(f"Loading Coqui TTS model: {tts_model}")
        self.tts = TTS(
            model_name=tts_model, 
            progress_bar=False, 
        ).to(device="cuda" if self.use_gpu else "cpu")
        print(f"✓ TTS model loaded (GPU: {self.use_gpu})")
        
        # Thread lock for TTS model (Coqui isn't fully thread-safe)
        self.tts_lock = threading.Lock()
        
        # Cache for processed chunks to avoid reprocessing
        self.chunk_cache = {}


    def intelligent_text_chunking(self, text: str) -> List[str]:
        """
        Chunk text intelligently to avoid stuttering and improve TTS quality
        """
        # Clean the text first
        text = self.clean_text_for_tts(text)

        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph alone is too long, split by sentences
            if len(paragraph) > self.chunk_size:
                sentences = self.split_into_sentences(paragraph)
                
                for sentence in sentences:
                    if len(current_chunk + " " + sentence) <= self.chunk_size:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
            else:
                # Check if adding this paragraph exceeds chunk size
                if len(current_chunk + "\n\n" + paragraph) <= self.chunk_size:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks (less than 10 characters)
        chunks = [chunk for chunk in chunks if len(chunk) > 10]
        
        print(f"✓ Created {len(chunks)} optimized chunks")
        return chunks

    def clean_text_for_tts(self, text: str) -> str:
        """Clean text to improve TTS quality and reduce stuttering"""
        
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Fix common formatting issues
            text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words split across lines
            text = re.sub(r'\n+', '\n\n', text)  # Normalize paragraph breaks
            
            # Remove problematic characters that cause TTS issues (fixed regex)
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-\(\)\n]', '', text)
            
            # Fix spacing around punctuation
            text = re.sub(r'\s+([\.!\?])', r'\1', text)
            text = re.sub(r'([\.!\?])\s*([A-Z])', r'\1 \2', text)
            
            # Normalize quotes (escape the regex properly)
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r"[''']", "'", text)
            
            return text.strip()
            
        except re.error as e:
            print(f"Regex error in text cleaning: {e}")

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences more intelligently"""
        # Simple sentence splitting that handles common abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def process_single_chunk_tts(self, chunk_data: Tuple[int, str]) -> Tuple[int, str, bool]:
        """Process a single text chunk with Coqui TTS"""
        chunk_id, text = chunk_data
        
        try:
            # Create temporary file for this chunk
            temp_dir = tempfile.gettempdir()
            chunk_audio_path = os.path.join(temp_dir, f"chunk_{chunk_id:04d}.wav")
            
            # Use thread lock to ensure thread safety with Coqui TTS
            with self.tts_lock:
                # Generate TTS for this chunk
                self.tts.tts_to_file(
                    text=text,
                    file_path=chunk_audio_path
                )
            
            print(f"✓ Processed chunk {chunk_id + 1}")
            return chunk_id, chunk_audio_path, True
            
        except Exception as e:
            print(f"✗ Error processing chunk {chunk_id}: {e}")
            return chunk_id, None, False

    def parallel_tts_processing(self, chunks: List[str]) -> List[Tuple[int, str]]:
        """Process multiple text chunks in parallel"""
        
        chunk_data = [(i, chunk) for i, chunk in enumerate(chunks)]
        successful_results = []
        
        # Use ThreadPoolExecutor with limited workers to prevent resource exhaustion
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self.process_single_chunk_tts, data): data[0] 
                for data in chunk_data
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    if result[2]:  # Successfully processed
                        successful_results.append((result[0], result[1]))
                    else:
                        print(f"Failed to process chunk {chunk_id}")
                        
                except Exception as e:
                    print(f"Exception in chunk {chunk_id}: {e}")
        
        # Sort results by chunk_id to maintain order
        successful_results.sort(key=lambda x: x[0])
        return successful_results

    def combine_audio_files(self, audio_file_paths: List[str], output_path: str) -> bool:
        """Combine multiple audio files into one seamless audiobook"""
        
        try:
            print("Combining audio files...")

            valid_audio_paths = []
            for audio_path in audio_file_paths:
                if audio_path and os.path.exists(audio_path):
                    valid_audio_paths.append(audio_path)
                else:
                    print(f"Warning: Audio file not found or None: {audio_path}")

            if not valid_audio_paths:
                print("No valid audio files to combine")
                return False
            
            combined_audio = []
            sample_rate = None

            print(f"Combining {len(valid_audio_paths)} audio files...")
            
            for i, audio_path in enumerate(valid_audio_paths):
                try:
                    # Read audio file
                    rate, audio_data = wavfile.read(audio_path)
                    
                    # Set sample rate from first file
                    if sample_rate is None:
                        sample_rate = rate
                    elif sample_rate != rate:
                        print(f"Warning: Sample rate mismatch in {audio_path}")
                    
                    # Add audio data
                    combined_audio.append(audio_data)
                    
                    # Add small pause between chunks (0.3 seconds), but not after the last chunk
                    if i < len(valid_audio_paths) - 1:
                        pause_samples = int(0.3 * sample_rate)
                        silence = np.zeros(pause_samples, dtype=audio_data.dtype)
                        combined_audio.append(silence)
                        
                    print(f"✓ Added audio file {i+1}/{len(valid_audio_paths)}")
                    
                except Exception as e:
                    print(f"Error reading audio file {audio_path}: {e}")
                    continue
            
            if not combined_audio:
                print("No audio data to combine")
                return False
            
            # Concatenate all audio
            final_audio = np.concatenate(combined_audio)
            
            # Ensure output directory exists (only if there's a directory component)
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if there's a directory component
                os.makedirs(output_dir, exist_ok=True)
            
            # Save combined audio
            wavfile.write(output_path, sample_rate, final_audio)
            
            # Clean up temporary files
            for audio_path in valid_audio_paths:
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {audio_path}: {e}")
            
            print(f"✓ Audio combined and saved to: {output_path}")
            print(f"✓ Total duration: {len(final_audio) / sample_rate:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"Error combining audio files: {e}")
            import traceback
            traceback.print_exc()
            return False

    def pdf_to_audiobook_pipeline(self, 
                                output_path: str,
                                chunk_size: int = None) -> bool:
        """
        Complete pipeline: PDF -> Text Extraction -> Chunking -> Parallel TTS -> Audio Combination
        """
        
        start_time = time.time()
        
        try:
            # Override chunk size if provided
            if chunk_size:
                self.chunk_size = chunk_size
            
            print("=" * 60)
            print(f"STARTING PDF TO AUDIOBOOK CONVERSION")
            print(f"Output: {output_path}")
            print(f"Chunk size: {self.chunk_size} characters")
            print(f"Max workers: {self.max_workers}")
            print("=" * 60)
            
            # Step 1: Extract story text from PDF using Gemini
            
            # Step 2: Chunk the text intelligently
            chunks = self.intelligent_text_chunking(self.text)
            if not chunks:
                print("Failed to create text chunks")
                return False
            
            print(f"Text length: {len(self.text)} characters")
            print(f"Number of chunks: {len(chunks)}")
            print(f"Average chunk size: {len(self.text) / len(chunks):.0f} characters")
            
            # Step 3: Process chunks in parallel with TTS
            print("\nStarting parallel TTS processing...")
            audio_results = self.parallel_tts_processing(chunks)
            
            if not audio_results:
                print("No audio chunks were generated successfully")
                return False
            
            print(f"Successfully generated {len(audio_results)}/{len(chunks)} audio chunks")
            
            # Step 4: Combine all audio files
            audio_file_paths = [result[1] for result in audio_results]
            success = self.combine_audio_files(audio_file_paths, output_path)
            
            if success:
                end_time = time.time()
                total_time = end_time - start_time
                
                print("\n" + "=" * 60)
                print("🎉 AUDIOBOOK CONVERSION COMPLETED SUCCESSFULLY!")
                print(f"⏱️  Total processing time: {total_time:.2f} seconds")
                print(f"📁 Output file: {output_path}")
                print(f"📊 Chunks processed: {len(audio_results)}/{len(chunks)}")
                print("=" * 60)
                
                return True
            else:
                print("Failed to combine audio files")
                return False
                
        except Exception as e:
            print(f"Pipeline error: {e}")
            return False
    
# Initialize the pipeline
# pipeline = OptimizedGeminiCoquiPipeline(
#     tts_model="tts_models/en/ljspeech/tacotron2-DDC",
#     text=text[:400],
#     use_gpu=True,
#     max_workers=4,
#     chunk_size=200  # Optimal size to prevent stuttering
# )

# # Convert a single PDF
# success = pipeline.pdf_to_audiobook_pipeline(
#     output_path="temp_audiobook.wav",
#     chunk_size=350  # Experiment with different sizes
# )

# if success:
#     print("🎉 Audiobook created successfully!")
# else:
#     print("❌ Failed to create audiobook")