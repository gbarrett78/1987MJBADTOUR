# MJ 1987 Bad Tour – Audio Processing CI/CD Pipeline

This project demonstrates a production-style CI/CD pipeline that processes `.mp3` audio files using AWS managed services.

The pipeline:
- Transcribes audio using Amazon Transcribe
- Translates text using Amazon Translate
- Synthesizes translated speech using Amazon Polly
- Stores all outputs in structured Amazon S3 folders
- Runs automatically via GitHub Actions

---

## Architecture Overview

**Services Used**
- GitHub Actions (CI/CD)
- Amazon S3 (storage)
- Amazon Transcribe (speech-to-text)
- Amazon Translate (language translation)
- Amazon Polly (text-to-speech)
- Python + boto3

---

## Repository Structure


---

## How the Pipeline Works

1. Audio files (`.mp3`) are placed in S3 under:

2. GitHub Actions runs the pipeline:
- Downloads audio inputs from S3
- Uploads audio to S3 (if needed)
- Runs Transcribe → Translate → Polly
- Uploads outputs back to S3

3. Outputs are stored in environment-specific folders:

---

## Environments

### Beta
- Triggered on Pull Requests
- Outputs written to:

### Production
- Triggered on merge to `main`
- Outputs written to:

---

## AWS Setup Requirements

Ensure the following AWS services are available:
- Amazon S3
- Amazon Transcribe
- Amazon Translate
- Amazon Polly

The IAM user must have permissions for:
- S3 read/write
- Transcribe jobs
- Translate text
- Polly synthesis

---

## GitHub Secrets Configuration

Set the following repository secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET_NAME`

**Do not hardcode credentials in the code.**

---

## Running the Pipeline

### Option 1: Via GitHub Actions (Recommended)
- Upload `.mp3` files to:
- Trigger a PR (beta) or merge to `main` (prod)

### Option 2: Local Execution
```bash
pip install -r requirements.txt
python process_audio.py

---

## Step 3: Save, commit, push (last commands of the night)

```bash
git add README.md
git commit -m "Add project README documentation"
git push origin main
