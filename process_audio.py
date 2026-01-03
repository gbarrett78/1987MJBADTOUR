import boto3
import os
import time
import logging
import json
import urllib.request
import re
from botocore.exceptions import ClientError

# -----------------------------
# Configuration (env + constants)
# -----------------------------
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "beta")

INPUT_FOLDER = "audio_inputs/"
TARGET_LANGUAGE = "es"   # Spanish
SOURCE_LANGUAGE = "en"   # English

# Fail fast if required env vars are missing
if not S3_BUCKET_NAME:
    raise ValueError("Missing env var S3_BUCKET_NAME. Set it before running.")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# AWS Clients
# -----------------------------
s3_client = boto3.client("s3")
transcribe_client = boto3.client("transcribe")
translate_client = boto3.client("translate")
polly_client = boto3.client("polly")

def download_inputs_from_s3():
    """Download MP3 files from s3://bucket/<env>/audio_inputs/ into local audio_inputs/"""
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    prefix = f"{ENVIRONMENT}/audio_inputs/"
    resp = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)

    contents = resp.get("Contents", [])
    if not contents:
        logging.info(f"No objects found in s3://{S3_BUCKET_NAME}/{prefix}")
        return 0

    downloaded = 0
    for obj in contents:
        key = obj["Key"]
        if key.lower().endswith(".mp3"):
            local_path = os.path.join(INPUT_FOLDER, os.path.basename(key))
            s3_client.download_file(S3_BUCKET_NAME, key, local_path)
            logging.info(f"Downloaded {key} -> {local_path}")
            downloaded += 1

    return downloaded


def upload_to_s3(file_path: str, object_name: str) -> bool:
    """Upload a file to an S3 bucket."""
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, object_name)
        logging.info(f"Uploaded {file_path} to s3://{S3_BUCKET_NAME}/{object_name}")
        return True
    except ClientError as e:
        logging.error(f"S3 Upload Error: {e}", exc_info=True)
        return False


def start_transcription_job(job_name: str, s3_uri: str) -> None:
    """Start an Amazon Transcribe job."""
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat="mp3",
            LanguageCode="en-US",
            # Optional but recommended: put output in your bucket for easier debugging
            OutputBucketName=S3_BUCKET_NAME,
            OutputKey=f"{ENVIRONMENT}/transcribe-output/{job_name}.json",
        )
        logging.info(f"Transcription job {job_name} started.")
    except Exception as e:
        logging.error(f"Error starting transcription job: {e}", exc_info=True)


def get_transcription_result(job_name: str) -> str | None:
    """Wait for transcription job and return the transcript text."""
    while True:
        result = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = result["TranscriptionJob"]["TranscriptionJobStatus"]

        if status in ["COMPLETED", "FAILED"]:
            break

        logging.info(f"Transcription job status: {status}. Waiting 30s...")
        time.sleep(30)

    if status == "FAILED":
        reason = result["TranscriptionJob"].get("FailureReason", "Unknown error")
        logging.error(f"Transcription job failed: {reason}")
        return None

    # If COMPLETED, try to fetch transcript (signed URL) and parse text
    transcript_uri = result["TranscriptionJob"].get("Transcript", {}).get("TranscriptFileUri")
    if not transcript_uri:
        logging.error("Transcript URL not found in completed job.")
        return None

    try:
        with urllib.request.urlopen(transcript_uri) as response:
            transcript_data = json.loads(response.read().decode("utf-8"))
            return transcript_data["results"]["transcripts"][0]["transcript"]
    except Exception as e:
        logging.error(f"Error reading transcript JSON: {e}", exc_info=True)
        return None


def translate_text(text: str, target_language: str) -> str:
    """Translate text using Amazon Translate."""
    result = translate_client.translate_text(
        Text=text,
        SourceLanguageCode=SOURCE_LANGUAGE,
        TargetLanguageCode=target_language,
    )
    return result["TranslatedText"]


def synthesize_speech(text: str, target_language: str) -> bytes:
    """Synthesize speech using Amazon Polly."""
    # Use a commonly-available Spanish voice + fallback if neural isn't supported in region
    voice_id = "Lupe" if target_language == "es" else "Joanna"

    try:
        response = polly_client.synthesize_speech(
            VoiceId=voice_id,
            OutputFormat="mp3",
            Text=text,
            Engine="neural",
        )
    except polly_client.exceptions.InvalidParameterValueException:
        # Fallback to standard engine if neural isn't available for the voice/region
        response = polly_client.synthesize_speech(
            VoiceId=voice_id,
            OutputFormat="mp3",
            Text=text,
            Engine="standard",
        )

    return response["AudioStream"].read()


def process_file(file_path: str) -> None:
    """Main processing function for a single mp3 file."""
    filename = os.path.basename(file_path)
    base, _ = os.path.splitext(filename)

    # Build S3 key for audio input
    s3_key_audio_input = f"{ENVIRONMENT}/audio_inputs/{filename}"

    # Ensure Transcribe job name is valid: letters, numbers, underscore, hyphen
    safe_base = re.sub(r"[^A-Za-z0-9_-]", "-", base)
    transcribe_job_name = f"job-{safe_base}-{int(time.time())}"

    # Upload input to S3
    if not upload_to_s3(file_path, s3_key_audio_input):
        return

    # Start transcription
    s3_uri_input = f"s3://{S3_BUCKET_NAME}/{s3_key_audio_input}"
    start_transcription_job(transcribe_job_name, s3_uri_input)

    # Wait for transcript
    transcript_text = get_transcription_result(transcribe_job_name)
    if not transcript_text:
        logging.error("Processing failed at transcription stage.")
        return

    # Translate + TTS
    try:
        translated_text = translate_text(transcript_text, TARGET_LANGUAGE)
        audio_bytes = synthesize_speech(translated_text, TARGET_LANGUAGE)
    except Exception as e:
        logging.error(f"Translate/Polly stage failed: {e}", exc_info=True)
        return

    # Upload results to S3
    try:
        # 1) Transcript
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{ENVIRONMENT}/transcripts/{base}.txt",
            Body=transcript_text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

        # 2) Translation
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{ENVIRONMENT}/translations/{base}_{TARGET_LANGUAGE}.txt",
            Body=translated_text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

        # 3) Audio Output
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{ENVIRONMENT}/audio-outputs/{base}_{TARGET_LANGUAGE}.mp3",
            Body=audio_bytes,
            ContentType="audio/mpeg",
        )

        logging.info(f"Successfully processed {filename}")
    except ClientError as e:
        logging.error(f"Error uploading results to S3: {e}", exc_info=True)


if __name__ == "__main__":
    download_inputs_from_s3()
    
    if not os.path.exists(INPUT_FOLDER):
        logging.error(f"Input folder '{INPUT_FOLDER}' does not exist.")
    else:
        for fname in os.listdir(INPUT_FOLDER):
            if fname.lower().endswith(".mp3"):
                process_file(os.path.join(INPUT_FOLDER, fname))
