from flask import abort
from google.cloud import storage
from google.oauth2 import service_account
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'flac'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4'}

# Path to your service account JSON key file
SERVICE_ACCOUNT_FILE = os.getenv("CREDENTIALS_FILE")
# Create credentials object explicitly
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

def download_from_bucket(bucket_name, source_blob_name, destination_file_name):

    """Downloads a blob from the bucket."""
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def download_file(url, file_type):
    """Download file from Google Cloud Storage"""
    try:
        # break url
        file_location_data = url.split('/')
        bucket_name = file_location_data[2]
        source_blob_path = "/".join(file_location_data[3:])
        filename = file_location_data[-1]
        destination_file_name, destination_file_ext = filename.split(".")
        destination_file_name = f"{destination_file_name}.{destination_file_ext}"

        if file_type == "audio":
            if not is_allowed_audio_file(destination_file_ext):
                raise Exception(f"Wrong extension. Allowed extension for audio is {ALLOWED_AUDIO_EXTENSIONS}")

        elif file_type == "image":
            if not is_allowed_image_file(destination_file_ext):
                raise Exception(f"Wrong extension. Allowed extension for image is {ALLOWED_IMAGE_EXTENSIONS}")
            
        elif file_type == "image":
            if not is_allowed_video_file(destination_file_ext):
                raise Exception(f"Wrong extension. Allowed extension for image is {ALLOWED_IMAGE_EXTENSIONS}")

        if not os.path.isfile(destination_file_name):
            download_from_bucket(bucket_name, source_blob_path, destination_file_name)
    
        return destination_file_name
    
    except Exception as e:
        abort(404, description=e)

def upload_folder_to_gcs(bucket_name, bucket_folder_path, local_folder_path):
    """Uploads a local folder to GCS and returns the bucket URL
    
    Args:
        bucket_name: Name of the GCS bucket (e.g., 'my-bucket')
        bucket_folder_path: Destination path in bucket (e.g., 'uploads/2024')
        local_folder_path: Local folder path to upload (should contain the main folder)
        
    Returns:
        str: gs:// URL of the uploaded folder
    """
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    
    # Ensure bucket folder path ends with /
    if bucket_folder_path and not bucket_folder_path.endswith('/'):
        bucket_folder_path += '/'
    
    # Get the main folder name
    main_folder_name = os.path.basename(os.path.normpath(local_folder_path))
    
    # Upload all files
    for root, _, files in os.walk(local_folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            # Calculate relative path from the parent of the main folder
            relative_path = os.path.join(main_folder_name, os.path.relpath(local_path, local_folder_path))
            blob_path = os.path.join(bucket_folder_path, relative_path).replace('\\', '/')
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
    
    return f"gs://{bucket_name}/{bucket_folder_path}{main_folder_name}/"

def is_allowed_audio_file(extension):

    """Check if the file is an allowed audio type"""
    if extension in ALLOWED_AUDIO_EXTENSIONS:
        return True
    else:
        abort(404, description=f"Wrong extension. Allowed extension for audio is {ALLOWED_AUDIO_EXTENSIONS}")

def is_allowed_image_file(extension):

    """Check if the file is an allowed audio type"""
    if extension in ALLOWED_IMAGE_EXTENSIONS:
        return True
    else:
        abort(404, description=f"Wrong extension. Allowed extension for audio is {ALLOWED_AUDIO_EXTENSIONS}")

def is_allowed_video_file(extension):

    """Check if the file is an allowed audio type"""
    if extension in ALLOWED_VIDEO_EXTENSIONS:
        return True
    else:
        abort(404, description=f"Wrong extension. Allowed extension for audio is {ALLOWED_AUDIO_EXTENSIONS}")

def file_exists(bucket_name, bucket_dir, file_path):
    bucket_path = f'{bucket_dir}/{file_path}'
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(bucket_path)
    return blob.exists()

