import boto3
import os
import time
import logging

# Configuration from environment variables (set by GitHub Actions)
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME') 
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'beta')
INPUT_FOLDER = 'audio_inputs/'
TARGET_LANGUAGE = 'es' # Example: 'es' for Spanish. Check supported language codes for Translate and Polly.
SOURCE_LANGUAGE = 'en' # Example: 'en' for English.

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize AWS Clients
s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe')
translate_client = boto3.client('translate')
polly_client = boto3.client('polly')

def upload_to_s3(file_path, object_name):
    """Upload a file to an S3 bucket."""
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, object_name)
        logging.info(f"Uploaded {file_path} to s3://{S3_BUCKET_NAME}/{object_name}")
    except ClientError as e:
        logging.error(e)
        return False
    return True

def start_transcription_job(audio_filename, job_name, s3_uri):
    """Start an Amazon Transcribe job."""
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='mp3', # Specify your media format
            LanguageCode=f"{SOURCE_LANGUAGE}-US" # Example: 'en-US'. Check supported language codes.
        )
        logging.info(f"Transcription job {job_name} started.")
    except Exception as e:
        logging.error(f"Error starting transcription job: {e}")

def get_transcription_result(job_name):
    """Wait for transcription job and return the transcript text."""
    while True:
        result = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = result['TranscriptionJob']['TranscriptionJobStatus']
        if status in ['COMPLETED', 'FAILED']:
            break
        logging.info(f"Transcription job status: {status}. Waiting...")
        time.sleep(30) # Wait 30 seconds before checking again

    if status == 'COMPLETED':
        transcript_uri = result['TranscriptionJob']['Transcript']['Uri']
        # Fetch the transcript file from S3
        import urllib.request
        with urllib.request.urlopen(transcript_uri) as response:
            transcript_data = response.read().decode('utf-8')
        # The transcript is a JSON, need to parse the actual text
        import json
        transcript_json = json.loads(transcript_data)
        transcript_text = transcript_json['results']['transcripts'][0]['transcript']
        return transcript_text
    else:
        logging.error(f"Transcription job failed: {result['TranscriptionJob'].get('FailureReason', 'No reason provided')}")
        return None

def translate_text(text, target_language):
    """Translate text using Amazon Translate."""
    result = translate_client.translate_text(
        Text=text,
        SourceLanguageCode=SOURCE_LANGUAGE,
        TargetLanguageCode=target_language
    )
    logging.info(f"Text translated to {target_language}.")
    return result['TranslatedText']

def synthesize_speech(text, target_language):
    """Synthesize speech using Amazon Polly and save to bytes."""
    # Note: Polly language codes may differ slightly (e.g., 'es-ES' instead of 'es')
    # Need to select a voice compatible with the language
    voice_id = 'Lucia' if target_language == 'es' else 'Joanna' # Example voices
    
    response = polly_client.synthesize_speech(
        VoiceId=voice_id,
        OutputFormat='mp3',
        Text=text,
        Engine='neural' # Use 'neural' for more lifelike voice if supported
    )
    logging.info(f"Speech synthesized with Polly using voice {voice_id}.")
    return response['AudioStream'].read()

def process_file(file_path):
    """Main processing function for a single file."""
    filename = os.path.basename(file_path)
    base, ext = os.path.splitext(filename)
    s3_key_audio_input = f'audio-inputs/{filename}'
    transcribe_job_name = f'transcribe-job-{base}-{int(time.time())}'

    # 1. Upload .mp3 file to S3
    if not upload_to_s3(file_path, s3_key_audio_input):
        return

    s3_uri_input = f's3://{S3_BUCKET_NAME}/{ENVIRONMENT}/{s3_key_audio_input}'

    # 2. Call Amazon Transcribe
    start_transcription_job(filename, transcribe_job_name, s3_uri_input)
    transcript_text = get_transcription_result(transcribe_job_name)

    if transcript_text:
        # 3. Call Amazon Translate
        translated_text = translate_text(transcript_text, TARGET_LANGUAGE)
        
        # 4. Call Amazon Polly
        audio_bytes = synthesize_speech(translated_text, TARGET_LANGUAGE)
        
        # 5. Upload all outputs to S3
        # Upload transcript
        s3_key_transcript = f'{ENVIRONMENT}/transcripts/{base}.txt'
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key_transcript, Body=transcript_text)
        logging.info(f"Uploaded transcript to s3://{S3_BUCKET_NAME}/{s3_key_transcript}")

        # Upload translation
        s3_key_translation = f'{ENVIRONMENT}/translations/{base}_{TARGET_LANGUAGE}.txt'
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key_translation, Body=translated_text)
        logging.info(f"Uploaded translation to s3://{S3_BUCKET_NAME}/{s3_key_translation}")

        # Upload synthesized audio
        s3_key_audio_output = f'{ENVIRONMENT}/audio-outputs/{base}_{TARGET_LANGUAGE}.mp3'
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key_audio_output, Body=audio_bytes)
        logging.info(f"Uploaded synthesized audio to s3://{S3_BUCKET_NAME}/{s3_key_audio_output}")

    else:
        logging.error("Could not process file due to transcription failure.")

if __name__ == "__main__":
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith('.mp3'):
            file_path = os.path.join(INPUT_FOLDER, filename)
            logging.info(f"--- Starting processing for {filename} ---")
            process_file(file_path)
            logging.info(f"--- Finished processing for {filename} ---")