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
import numpy as np
from torch.nn.functional import softmax

os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

dotenv.load_dotenv()
### Initialize model
language_id_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
tmp_audio_path = "temp_audio.wav"
tmp_transcript_file = "output.json"
frequent_langauges_list = ["en-US", "de-DE", "es-CL", "ar-AE", 
                           "ru-RU", "it-IT", "hi-IN", "yue-CN"]
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
 # Initialize the dictionary
label_id_to_code = {}


def parse_label_encoding_file():
    # Read the file (assuming 'languages.txt' contains lines like "'ab: Abkhazian' => 0")
    with open('tmp\\label_encoder.ckpt', 'r') as file:
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


def run_speech_recognizer_and_diarization(audio_file_path, save_file_path, languages_in_audio):
    speech_config1 = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_API_KEY'), region=os.environ.get('AZURE_API_REGION'))
    speech_config2 = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_API_KEY'), region=os.environ.get('AZURE_API_REGION'))

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

    # Set the LanguageIdMode (Optional; Either Continuous or AtStart are accepted; Default AtStart)
    speech_config1.set_property(property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, value='Continuous')
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=languages_in_audio)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config1, 
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config)
    
    speech_config2.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, value='true')
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
                                speech_config=speech_config2, 
                                audio_config=audio_config)    
                                # auto_detect_source_language_config=auto_detect_source_language_config)
    transcribing_stop = False
    done = False
    # Store results in a structured format
    transcription_result = {
        # "speakers": {},       # Speaker ID -> List of utterances
        "transcript": "",     # Raw transcript (optional)
        "speakers": []    # Total unique speakers
    }

    def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('*', end="")
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            # print(text)
            transcription_result["transcript"] += f"{text}\n" 
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))
    
    def conversation_transcriber_transcribed_cb2(evt: speechsdk.SpeechRecognitionEventArgs):
        print('*', end="")
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            speaker_id = evt.result.speaker_id
            if speaker_id  not in transcription_result["speakers"] and not 'Unknown':
                transcription_result["speakers"].append(speaker_id)
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))


    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dump(transcription_result, f, indent=1, ensure_ascii=False))


    def stop_cb2(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True
        # print(f"\nTranscription saved to: {save_file_path}")
            
    speech_recognizer.recognized.connect(conversation_transcriber_transcribed_cb)
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb2)
    conversation_transcriber.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    conversation_transcriber.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    conversation_transcriber.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb2)
    conversation_transcriber.canceled.connect(stop_cb2)
    conversation_transcriber.start_transcribing_async()

    speech_recognizer.start_continuous_recognition()
    while not done and not transcribing_stop:
        time.sleep(.5)
    conversation_transcriber.stop_transcribing_async()
    speech_recognizer.stop_continuous_recognition()



def run_speech_diarization(audio_file_path, lang_code, save_file_path):
    """
    This methods runs for single language audio
    """
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_API_KEY'), region=os.environ.get('AZURE_API_REGION'))
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


# Initialize OpenAI client (make sure to set your API key)
def get_speech_metadata(filepath):
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
    - "translated_text" -- here add transcript with english translation of any other language
    - "discussion_topic" -- here add the precise topic of audio 
    - "keywords" -- here extract relevant search keywords related to it
    - "audio_tags" -- here add relevant tags
    - "speaker_count" {speaker_count}
    
    if speaker count 0 try estimating it from transcript.
    keep the text in same language as the transcript.
    """
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


def get_audio_data(audio_file, save_path=None):
    ### resample audio file
    resampled_audio, audio_sr = get_resampled_audio(audio_file)
    parse_label_encoding_file()  

    ### extract audio sample for indentification
    start_sec = 0
    end_sec = 30    
    # start_sample = int(start_sec * audio_sr)
    # end_sample = int(end_sec * audio_sr)
    # audio_segment = resampled_audio[start_sample:end_sample]

    #TODO -> edit audio so we remove some start and end section
    ### perform language identification
    # fixed removal of 5 secs from start and end
    # this can be modified to remove more minutes from longer audio
    audio_segment = resampled_audio[0+80000:len(resampled_audio) - 80000]
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
    sf.write(tmp_audio_path, resampled_audio, audio_sr, subtype='PCM_16')   

    if multi_lingual_audio:
        run_speech_recognizer_and_diarization(tmp_audio_path, tmp_transcript_file, languages_used)
    else:
    ### run speech extraction
        run_speech_diarization(tmp_audio_path, azure_language_code, tmp_transcript_file)

    json_speech_metadata = get_speech_metadata(tmp_transcript_file)
    json_speech_metadata = json.dumps(json_speech_metadata, indent=1, ensure_ascii=False)
    # with open(save_path, 'w', encoding='utf-8') as fh:
    #     fh.write(json.dumps(json_speech_metadata, ascii=False, indent=1))
    return json_speech_metadata

if __name__=="__main__":
    # get_audio_data("health-german.mp3", "health-german.json")
    data = get_speech_metadata("output.json")
    with open("temp_file.json", 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(data, ensure_ascii=False, indent=1))