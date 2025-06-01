from celery_tasks import process_video_task, get_pinecode_index
from moviepy import VideoFileClip, concatenate_videoclips
from video_pipeline import semantic_search, save_frame
from flask import Flask, request, jsonify, abort
from vision_pipeline import VisionMetaData
from audio_pipeline import get_audio_data
from utils import upload_folder_to_gcs
from dotenv import load_dotenv
from utils import file_exists
from io import BytesIO
from PIL import Image
from utils import *
from pathlib import Path
from shutil import rmtree
import os
import uuid
import traceback
import json

app = Flask(__name__)

load_dotenv('.env')

BUCKET = os.getenv("BUCKET_NAME")
BUCKET_DIR = os.getenv("BUCKET_DIR")

credentials = { 
    "google_credentials": os.getenv("CREDENTIALS_FILE"),
    "openai_key": os.getenv("OPENAI_API_KEY"),
    "aws_secret_key": os.getenv("AWS_SECRET_KEY"),
    "aws_access_key": os.getenv("AWS_ACCESS_KEY"),
    "azure_api_key" : os.getenv("AZURE_API_KEY"),
    "azure_api_region" : os.getenv("AZURE_API_REGION"),
    "pinecone_api_key" : os.getenv("PINECONE_API_KEY"),
    "pinecone_env": os.getenv("PINECONE_ENV"),
    "templated_api_key": os.getenv('TEMPLATED_API_KEY'),
    "templated_template_id": os.getenv('TEMPLATED_TEMPLATE_ID')

}

imageDataExtractor = VisionMetaData(
    credentials_path=credentials["google_credentials"],
    openai_api_key=credentials["openai_key"],
    aws_access_key=credentials["aws_access_key"],
    aws_secret_key=credentials["aws_secret_key"]
)

@app.route('/health')
def health():
    return 'OK'

@app.route('/process-audio', methods=['POST'])
def process_audio():
    """Endpoint for processing audio files"""
    if 'url' not in request.json:
        abort(400, description="Missing 'url' in request body")
    
    url = request.json['url']

    # check if path exists otherwise throw error
        
    # Download the file
    filepath = download_file(url, file_type="audio") 
    # Process the audio file
    result = get_audio_data(filepath, credentials["openai_key"], credentials["azure_api_key"], credentials["azure_api_region"])
    
    return jsonify({
        "response":result,
        "code": 200
    })

@app.route('/process-image', methods=['POST'])
def process_image():
    
    """Endpoint for processing image files"""
    if 'url' not in request.json:
        abort(400, description="Missing 'url' in request body")

    url = request.json['url']

    # Download the file
    filepath = download_file(url, file_type="image") 

    # # Process the image
    image = Image.open(filepath)
    imageByteArray = BytesIO()
    image.save(imageByteArray, format="JPEG")
    result = imageDataExtractor.get_image_metadata(imageByteArray.getvalue())

    return jsonify({
        "response":result,
        "code": 200,
    })

@app.route('/process-video', methods=['POST'])
def process_video():
    
    """Endpoint for processing image files"""
    if 'url' not in request.json:
        abort(400, description="Missing 'url' in request body")

    url = request.json['url']

    # Download the file
    filepath = download_file(url, file_type="video") 
    
    # initiate task
    jobid = str(uuid.uuid4())
    task = process_video_task.apply_async(args=[filepath, jobid], task_id=jobid)

    response = {
        "jobid":jobid,
        "status":"queued",
        "check_status": f"/status/{jobid}"
    }

    return jsonify({
        "response": response,
        "code": 202
    })
            

@app.route("/status/<jobid>", methods=['GET'])
def check_status(jobid):

    # check task status
    task = process_video_task.AsyncResult(jobid)
    code = 200

    if task.state == 'PENDING':
        response = {
            'status': 'pending',
            'message': 'Job is queued or not started'
        }
    
    elif task.state == 'PROGRESS':
        response = {
            'status': 'processing'
        }

    elif task.state == 'SUCCESS':
        result, gcs_bucket_path, input_file_path = task.get()
        response = {
            'status': 'completed',
            'video_file': f'{input_file_path}',
            'result': str(result),
            'download_data': {
                'path':f"{gcs_bucket_path}"
            },
            'generate_promo':{
                'url': f'/generate-promo/{jobid}'
            },
            'semantic-search':{
                'url': f'/semantic-search/{jobid}'
            }
        }

    else:
        response = {
            'status': 'failed',
            'message': str(task.info)  # contains exception info
        }
        code = 500
 
    return jsonify({
    "response":response,
    "code": code
    })

@app.route('/semantic-search/<jobid>', methods=['POST'])
def search(jobid):
    
    try:
        if 'query' not in request.json:
            abort(400, description="Missing 'query' in request body")
        if 'video_name' not in request.json:
            abort(400, description="Missing 'video_name' in request body")
        
        query = request.json['query']
        video_path = request.json['video_name']
        video_name, _ = video_path.split('.')
        index = get_pinecode_index(credentials)
        local_scenes_filepath = f"{jobid}/scenes/{video_name}.json"
        results = semantic_search(query, index, credentials)
        with open(local_scenes_filepath, 'rb') as fh:
            scenes = json.loads(fh.read())

        searched_scenes = []
        saved_frames = []
        results_path = f"search_results"

        if os.path.isdir(results_path):
            rmtree(results_path)
        Path(results_path).mkdir(exist_ok=True)

        for i, match in enumerate(results["matches"]):
            scene_idx = int(match["id"])
            if 0 <= scene_idx < len(scenes[scene_idx]):
                scene = scenes[scene_idx]
            else:
                continue 
            frame_path = save_frame(video_path, scene["start"], f"{results_path}/match_frame_{i}.jpg")
            searched_scenes.append(scene)
            saved_frames.append(frame_path)

        results_json = f"{results_path}/search_results.json"
        with open(results_json, "w") as f:
            json.dump(searched_scenes, f, indent=2)

        search_gsc_path = upload_folder_to_gcs(BUCKET, f"{BUCKET_DIR}/{jobid}", results_path)

        return jsonify({
        "code": 200,
        "response": "Processed Successfully",
        "results_path": search_gsc_path
        })
    
    except Exception as e:
        return jsonify({
        "code": 500,
        "response": f"{e}",
        })


@app.route('/generate-promo/<jobID>', methods=['POST'])
def generate_promo(jobID):
    
    global BUCKET_DIR, BUCKET

    try:
        if 'video_name' not in request.json:
            abort(400, description="Missing 'video_name' in request body")
        if 'scene_ids' not in request.json:
            abort(400, description="Missing 'scene_ids' in request body")

        video_path = request.json['video_name']
        scene_ids = request.json['scene_ids']
        video_name, _ = video_path.split('.')
        local_scenes_filepath = f"{jobID}/scenes/{video_name}.json"
        if not os.path.isfile(local_scenes_filepath):
            isfileInBucket = file_exists(BUCKET, BUCKET_DIR, local_scenes_filepath)
            if not isfileInBucket:
                abort(400, description="Error no such scene data exists process it first")
            else:
                BUCKET_DIR = f"{BUCKET_DIR}/{jobID}/scenes"
                download_from_bucket(BUCKET, BUCKET_DIR, f"{video_name}.json")

        with open(local_scenes_filepath, 'rb') as fh:
            scenes = json.loads(fh.read())
        clips = []
        for idx in scene_ids:
            sc = scenes[idx]
            clips.append(VideoFileClip(video_path).subclipped(sc['start'], sc['end']))
        promo = concatenate_videoclips(clips)
        promo_path = f"{video_name}_promo.mp4"
        promo_save_dir = f"promo"

        if os.path.isdir(promo_save_dir):
            rmtree(promo_save_dir)
        Path(promo_save_dir).mkdir(exist_ok=True)

        promo.write_videofile(f"{promo_save_dir}/{promo_path}", codec="libx264", audio_codec="aac")

        promo_gsc_path = upload_folder_to_gcs(BUCKET, f"{BUCKET_DIR}/{jobID}", promo_save_dir)
        return jsonify({
        "code": 200,
        "response": "Processed Successfully",
        "file_path": promo_gsc_path
    })

    except Exception as e:
        print(e)
        return jsonify({
        "code": 500,
        "response": f"{e}"
    })


@app.errorhandler(500)
@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Handle all unexpected errors with detailed info in debug mode"""
    status_code = 500
    if hasattr(error, 'code'):
        status_code = error.code

    response_data = {
        "status": "error",
        "message": "An unexpected error occurred"
    }
    
    if app.debug:
        response_data.update({
            "error": str(error),
            "traceback": traceback.format_exc()
        })

    response = jsonify(response_data)
    response.status_code = status_code
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8501)