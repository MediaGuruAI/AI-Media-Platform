from pydub import AudioSegment
# from pydub.silence import split_on_silence
import openai
import json
import dotenv
import os
import assemblyai as aai

config = dotenv.load_dotenv()
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
aai.settings.api_key=os.environ.get("ASSEMBLYAI_API_KEY")


def process_audio_whisperAPI(audio_content):
    # Export chunks as temporary MP3 files (smaller size)
    chunk_length_ms = 30 * 1000  # 30 seconds
    chunks = [
        audio_content[i:i + chunk_length_ms] 
        for i in range(0, len(audio_content), chunk_length_ms)
    ]

    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file = f"chunk_{i}.mp3"
        chunk.export(chunk_file, format="mp3", bitrate="128k")
        chunk_files.append(chunk_file)

    # --- Transcribe each chunk ---
    full_transcript = []
    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",  # Get timestamps
                temperature=0.2  # Reduce hallucinations
            )
            full_transcript.append(transcript.text)

    # Combine all chunks
    final_transcript = " ".join(full_transcript)
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    return final_transcript, 0    


def process_audio_assembleyAI(audio_content):
    transcriber = aai.Transcriber()

    config = aai.TranscriptionConfig(
    speaker_labels=True)
    transcript = transcriber.transcribe(audio_content, config)
    speaker_count = set()
    for utterance in transcript.utterances:
        # print(f"Speaker {utterance.speaker}: {utterance.text}")
        speaker_count.add(utterance.speaker)

    return transcript.text, len(speaker_count)

# Initialize OpenAI client (make sure to set your API key)
def get_audio_metadata(transcript, speaker_count):
    # Prepare the prompt for OpenAI
    prompt = f"""
    Analyze the following audio transcript and provide the following outputs in JSON format:
    1. Keyword detection: Identify and list the key phrases and topics discussed in the audio.
    2. Audio tagging: Tag the audio file based on contextual themes and content.

    transcript: {transcript}

    DO NOT HALLUCINATE ANYTHING. STICK TO THE LANGUAGE OF TRANSCRIPT AND THE CONTEXT AND INFORMATION IN IT

    Format your response as a JSON object with these keys:
    - "transcript"
    - "discussion_topic"
    - "keywords"
    - "audio_tags"
    - "speaker_count" {speaker_count}
    if speaker count 0 try estimating it from transcript.
    keep the text in same language as the transcript.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides audio transcript metadata."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        content = response.choices[0].message.content
        try:
            # Try to parse as JSON
            return json.loads(content)
        except json.JSONDecodeError:
            # If not valid JSON, return the raw content
            return {"enhanced_data": content}
    except Exception as e:
        return {"error": str(e)}


# def audio_metadata(transcript, speaker_count=0):
    # Craft a prompt to extract keywords and tags

# transcript, speaker_count = process_audio_assembleyAI("multi-speaker-audio.wav")
# Load the audio file
# audio = AudioSegment.from_file("V1.wav")

# transcript, speaker_count = process_audio_whisperAPI(audio)
# output = get_audio_metadata(transcript, speaker_count)


