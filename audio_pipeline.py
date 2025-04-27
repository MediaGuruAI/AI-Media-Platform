"""
- Get audio file
- extract small part and indentify language
- Check frequency and make it 16kh, mono channel
- Pass it to the Azure api and extract the response in a json
"""
#### Loading speech indentification model
import librosa
from speechbrain.inference.classifiers import EncoderClassifier
import torch
import os
import time
import azure.cognitiveservices.speech as speechsdk
import soundfile as sf
import dotenv
import json
import openai
from torch.nn.functional import softmax
import threading
import math
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

### Initialize model
language_id_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
tmp_audio_path = "temp_audio.wav"
tmp_transcript_file = "output.json"
frequent_langauges_list = ["en-US", "de-DE", "es-CL", "ar-AE", 
                           "ru-RU", "it-IT", "hi-IN", "yue-CN"]
 # Initialize the dictionary
label_id_to_code = {}


def parse_label_encoding_file():
    # Read the file (assuming 'languages.txt' contains lines like "'ab: Abkhazian' => 0")
    with open('tmp/label_encoder.ckpt', 'r') as file:
        for line in file:
            line = line.strip()  # Remove whitespace
            if not line:
                continue  # Skip empty lines
            # Split into parts: ["'ab: Abkhazian'", "0"]
            # Skip lines that don't follow the pattern: "'xx: Language Name' => ID"
            # Skip lines starting with '=' or containing 'starting_index'
            if line.startswith('=') or "'starting_index'" in line:
                continue
            parts = line.split('=>')
            if len(parts) != 2:
                continue  # Skip malformed lines
            # Extract the language code (e.g., 'ab')
            lang_part = parts[0].strip().strip("'")  # Remove quotes
            lang_code = lang_part.split(':', 1)[0].strip()  # Get 'ab'
            # Extract the ID (e.g., 0)
            lang_id = int(parts[1].strip())
            # Store in dictionary: {0: 'ab', 1: 'af', ...}
            label_id_to_code[lang_id] = lang_code


def find_language(lang_code):
    with open('azure_lang_codes.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(f"{lang_code}:"):
                parts = line.strip().split(':')
                return parts[2].strip().split(',')  # Return all locales
    return None


def get_resampled_audio(audio_segment, target_sr=16000):
    """
    Update sampling rate of audio if it is not 16khz
    """
    audio, orig_sr = librosa.load(audio_segment, sr=None)  # sr=None keeps original rate
    # audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    # if audio_segment.channels > 1:
    #     audio_array = audio_array.reshape((-1, audio_segment.channels))
    #     audio_array = librosa.to_mono(audio_array.T)
    # orig_sr = audio_segment.frame_rate
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio, target_sr

def run_speech_recognizer(audio_file_path, save_file_path, languages_in_audio, azure_key, azure_region):
    speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

    # Set the LanguageIdMode (Optional; Either Continuous or AtStart are accepted; Default AtStart)
    speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, value='Continuous')
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=languages_in_audio)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config)
    
    done = False
    # Store results in a structured format
    transcription_result = {
        # "speakers": {},       # Speaker ID -> List of utterances
        "transcript": "",     # Raw transcript (optional)
    }

    def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('*', end="")
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            # print(text)
            transcription_result["transcript"] += f"{text}\n" 
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))


    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True
        with open(save_file_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_result, f, indent=1, ensure_ascii=False)

    speech_recognizer.recognized.connect(conversation_transcriber_transcribed_cb)
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)
    speech_recognizer.stop_continuous_recognition()



def run_speech_diarization(audio_file_path, save_file_path, lang_code, azure_key, azure_region):
    """
    This methods runs for single language audio
    """
    speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
    speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, value='true')
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_config.speech_recognition_language=lang_code
    # Set the LanguageIdMode (Optional; Either Continuous or AtStart are accepted; Default AtStart)
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
                                speech_config=speech_config, 
                                audio_config=audio_config)    

    transcribing_stop = False
    # Store results in a structured format
    transcription_result = {
        # "speakers": {},       # Speaker ID -> List of utterances
        "transcript": "",     # Raw transcript (optional)
        "speakers": []    # Total unique speakers
    }

    def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
        print('Canceled event')

    def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
        print('SessionStopped event')

    def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
        print('SessionStarted event')

    def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('*', end="")
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            speaker_id = evt.result.speaker_id
            # print(text)
            transcription_result["transcript"] += f"{text}\n" 
            if speaker_id  not in transcription_result["speakers"] and not 'Unknown':
                transcription_result["speakers"].append(speaker_id)

        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dump(transcription_result, f, indent=1, ensure_ascii=False))
        print(f"\nTranscription saved to: {save_file_path}")

    conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
    # stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    conversation_transcriber.start_transcribing_async()

    while not transcribing_stop:
            time.sleep(.5)
    conversation_transcriber.stop_transcribing_async()


def split_audio_into_chunks(audio_array, sample_rate):
    chunk_duration, number_of_chunks = compute_chunk_duration(len(audio_array), sample_rate)
    chunk_size = int(chunk_duration * sample_rate)
    audio_chunks = []
    if number_of_chunks == 1:
        return [audio_array]
    else:
        start = 0
        end = chunk_size
        for _ in range(0, number_of_chunks-1):
            print(start, end)
            chunk = audio_array[start: end]
            audio_chunks.append(chunk)    
            start = end
            end = start + chunk_size
        last_chunk = audio_array[start:]
        audio_chunks.append(last_chunk)
        return audio_chunks

def compute_chunk_duration(audio_size, sample_rate):
    """
    We break each audio with some percent of chunk size if its greater 
    than 30 secs 
    """
    audio_time_length = math.floor(audio_size / sample_rate)
    print(audio_time_length)
    if audio_time_length < 30:
        per_of_chunk = 0.5
    elif audio_time_length > 30 and audio_time_length <= 120:
        per_of_chunk = 0.25
    else:
        per_of_chunk = 0.20
    chunk_duration = math.floor(audio_time_length * per_of_chunk)
    number_of_chunks = int(audio_time_length / chunk_duration)
    return chunk_duration, number_of_chunks


def create_audio_chunks(audio_array, sample_rate):
    # Split into 3 chunks 
    # chunk size should be 30% of total time of chunk
    chunks = split_audio_into_chunks(audio_array, sample_rate)
    # Save each chunk
    chunk_files = []
    for i, chunk in enumerate(chunks):
        sf.write(f"chunk_{i+1}.wav", chunk, sample_rate)
        chunk_files.append(f"chunk_{i+1}.wav")
    print(f"Saved {len(chunks)} chunks.")
    return chunk_files

def process_audio_chunks(audio_files, lang_code, func, azure_key, azure_region):
    threads = []
    saved_outputs = []

    # Start a thread for each chunk
    for i, audio_file in enumerate(audio_files, start=1):
        save_path = audio_file.split('.')[0] + ".json"
        print(save_path)
        thread = threading.Thread(
            target=func,
            args=(audio_file, save_path, lang_code, azure_key, azure_region)
        )
        thread.start()
        threads.append(thread)
        saved_outputs.append(save_path)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    
    print("Process completed")
    return saved_outputs


def process_json_files(files):
    combined_data = {
        "transcript": "",
        "speakers": set()  # Using a set to automatically handle uniqueness
    }
    # Process each file matching the pattern
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Append transcript with a space separator
            if combined_data["transcript"]:
                combined_data["transcript"] += " " + data["transcript"]
            else:
                combined_data["transcript"] = data["transcript"]
            # Add speakers to the set (automatically handles uniqueness)
            combined_data["speakers"].update(data["speakers"])
    # Convert the set back to a list for the final output
    combined_data["speakers"] = list(combined_data["speakers"])    
    return combined_data

def process_multi_audio_json(files):
    combined_data = {
        "transcript": "",
        "speakers": ""  # Using a set to automatically handle uniqueness
    }
    # Process each file matching the pattern
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Append transcript with a space separator
            if combined_data["transcript"]:
                combined_data["transcript"] += " " + data["transcript"]
            else:
                combined_data["transcript"] = data["transcript"]
    return combined_data

# Initialize OpenAI client (make sure to set your API key)
def get_speech_metadata(filepath, openai_api_key):
    # Prepare the prompt for OpenAI

    with open(filepath, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    transcript = data['transcript']
    speaker_count = data['speakers']
    prompt = f"""
    Think Step by Step:

    Analyze the following audio transcript and provide the following outputs in JSON format:
    1. Keyword detection: Identify and list the key phrases and topics discussed in the audio.
    2. Audio tagging: Tag the audio file based on contextual themes and content.

    transcript: {transcript}

    DO NOT HALLUCINATE ANYTHING. STICK TO THE LANGUAGE OF TRANSCRIPT AND THE CONTEXT AND INFORMATION IN IT

    Except original transcript give everything like discussion_topic, keywords and audio_tags in english

    Format your response as a JSON object with these keys:
    - "transcript" -- here add original transcript 
    - "discussion_topic" -- here add the precise topic of audio 
    - "keywords" -- here extract relevant search keywords related to it
    - "audio_tags" -- here add relevant tags
    - "speaker_count" {speaker_count}
    
    if speaker count 0 try estimating it from transcript.
    keep the text in same language as the transcript.
    """
    openai_client = openai.OpenAI(api_key=openai_api_key)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", 
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


def language_identification(audio, threshold=0.01):
    prediction = language_id_model.classify_batch(audio)
    ### we check for multilanguage 
    logits = prediction[0]
    probs = softmax(logits, dim=-1)
    probs = probs.sort(descending=True)
    first_5_values = probs.values[0, :5]  # [0] because values is 2D with shape (1, n)
    first_5_indices = probs.indices[0, :5]
    # Find which values are above the threshold
    above_threshold = first_5_values > threshold
    # Get the indices of values above threshold
    selected_indices = first_5_indices[above_threshold]
    print("Selected indices with probability > 0.01:", selected_indices.tolist())
    return selected_indices


def get_audio_data(audio_file, openai_api_key, azure_key, azure_region, save_path=None):
    ### resample audio file
    resampled_audio, audio_sr = get_resampled_audio(audio_file)
    parse_label_encoding_file()  
    audio_time_length = math.floor(len(resampled_audio) / audio_sr)

    ### extract audio sample for indentification
    start_sec = 0
    end_sec = 30    
    # start_sample = int(start_sec * audio_sr)
    # end_sample = int(end_sec * audio_sr)
    # audio_segment = resampled_audio[start_sample:end_sample]0

    ### perform language identification
    # fixed removal of 5 secs from start and end
    if audio_time_length < 20:
        trim_size = 0
    elif audio_time_length > 20 and audio_time_length < 40:
        # remove 10 secs
        trim_size = 160000
    elif audio_time_length > 40 and audio_time_length < 180:
        # remove 10 secs
        trim_size = 160000
    else:
        # remove 40 secs
        trim_size = 640000
    audio_segment = resampled_audio[0+trim_size: len(resampled_audio)-trim_size]
    # this can be modified to remove more minutes from longer audio
    audio_segment = torch.tensor(audio_segment) 
    # prediction = language_id_model.classify_batch(audio_segment)
    lid_result = language_identification(audio_segment)
    multi_lingual_audio = False
    print(lid_result)
    if len(lid_result) == 1:
        azure_language_code = find_language(label_id_to_code[lid_result.item()])[0]
        print(azure_language_code)   
    else:
        languages_used = []
        for id_value in lid_result:
            azure_language_code = find_language(label_id_to_code[id_value.item()])[0]
            languages_used.append(azure_language_code)
        multi_lingual_audio = True

    # print(f"The language of audio: {azure_language_code}")
    # sf.write(tmp_audio_path, resampled_audio, audio_sr, subtype='PCM_16')   
    chunk_names = create_audio_chunks(resampled_audio, audio_sr)
    
    # saved_outputs = process_audio_chunks(chunks_names)
    
    if multi_lingual_audio:
        saved_outputs = process_audio_chunks(chunk_names, 
                                             lang_code=languages_used, 
                                             func=run_speech_recognizer,
                                             azure_key=azure_key,
                                             azure_region=azure_region)
        output = process_multi_audio_json(saved_outputs)
    else:
    ### run speech extraction
        saved_outputs = process_audio_chunks(chunk_names, 
                                             lang_code=azure_language_code, 
                                             func=run_speech_diarization,
                                             azure_key=azure_key,
                                             azure_region=azure_region)

        output = process_json_files(saved_outputs) 
    # Save the combined output to a new file
    with open(tmp_transcript_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)     
    json_speech_metadata = get_speech_metadata(tmp_transcript_file, openai_api_key)
    json_speech_metadata = json.dumps(json_speech_metadata, indent=1, ensure_ascii=False)
    # with open(save_path, 'w', encoding='utf-8') as fh:
    #     fh.write(json.dumps(json_speech_metadata, ascii=False, indent=1))
    ## remove tmp files
    # for chunk in range(chunk_names):
    #     os.remove(chunk)
    # for tmpoutputfile in range(saved_outputs):
    #     os.remove(tmpoutputfile)
    # os.remove(tmp_transcript_file)
    return json_speech_metadata

if __name__=="__main__":
    # get_audio_data("health-german.mp3", "health-german.json")
    data = get_speech_metadata("output.json")
    with open("temp_file.json", 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(data, ensure_ascii=False, indent=1))
