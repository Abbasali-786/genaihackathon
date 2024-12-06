import streamlit as st
import yt_dlp
import os
import speech_recognition as sr
from transformers import pipeline

# Initialize Hugging Face models for summarization, QA, and classification
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to download audio from YouTube using yt-dlp
def download_audio(youtube_url, output_path="audio.mp4"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,  # Extract audio
        'audioquality': 1,     # High audio quality
        'outtmpl': output_path,  # Output path
        'quiet': False,        # Show progress
    }

    # Download the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    st.success(f"Downloaded audio to {output_path}")

# Function to convert audio (mp4) to text using SpeechRecognition
def audio_to_text(audio_path):
    # Use FFmpeg to extract audio from the MP4 file and convert it to PCM WAV format
    audio_wav_path = "audio.wav"
    
    # Convert MP4 to WAV format
    os.system(f"ffmpeg -i {audio_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_wav_path}")
    st.write(f"Audio extracted and converted to: {audio_wav_path}")
    
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    try:
        with sr.AudioFile(audio_wav_path) as source:
            audio_data = recognizer.record(source)
        
        # Use Google's speech recognition API to transcribe
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Function to summarize, generate Q&A, and classify content
def process_transcript(transcript):
    # Summarize Transcript
    summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)
    st.subheader("Summary:")
    st.write(summary[0]['summary_text'])

    # Question Answering
    questions = [
        "What is the main topic of the video?",
        "What does the speaker discuss in the video?"
    ]
    
    for question in questions:
        result = qa_pipeline({"context": transcript, "question": question})
        st.subheader(f"Question: {question}")
        st.write(f"Answer: {result['answer']}")

    # Content Classification
    labels = ["Entertainment", "Informative"]
    classification_result = classifier(transcript, candidate_labels=labels)
    st.subheader(f"Content Classification:")
    st.write(f"{classification_result['labels'][0]} with score: {classification_result['scores'][0]}")

# Streamlit UI
st.title("YouTube Audio Transcription and Analysis")

youtube_url = st.text_input("Enter YouTube URL:")
if youtube_url:
    with st.spinner('Downloading audio...'):
        download_audio(youtube_url, "audio.mp4")
    
    # Convert audio to text (transcript)
    with st.spinner('Transcribing audio...'):
        transcript = audio_to_text("audio.mp4")
    
    if transcript:
        st.subheader("Transcript:")
        st.write(transcript)
        
        # Process the transcript (summarization, Q&A, and classification)
        st.subheader("Processing Transcript...")
        process_transcript(transcript)
