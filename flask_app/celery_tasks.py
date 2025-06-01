from celery import Celery
import os
from video_pipeline import *
import shutil
from dotenv import load_dotenv
from utils import upload_folder_to_gcs

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

celery = Celery('celery_tasks', 
                broker="amqp://guest:guest@localhost:5672//",
                backend="rpc://",
                include=["celery_tasks"])

def get_pinecode_index(credentials):
    try:
        pc = Pinecone(api_key=credentials['pinecone_api_key'])
        if "video-highlights" not in pc.list_indexes().names():
            pc.create_index(
                name="video-highlights",
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index("video-highlights")
    except:
        index = None
    return index

@celery.task(bind=True)
def process_video_task(self, filepath, jobID):

    # make a save_dir with the name as jobid
    os.mkdir(f"{jobID}")
    # Pinecone

    try:
        self.update_state(
            state="PROGRESS"
            )

        index = get_pinecode_index(credentials)
        dirname, full_transcript, segments = transcribe(filepath, credentials, save_dir=jobID)
        shots = enrich_shots(filepath, segments, credentials=credentials, save_dir=jobID)
        index = index_scenes(shots, index, credentials)
        scene_summaries = [r["summary"] for r in shots]
        chunk_summaries = summarize_chunks_with_gpt4(scene_summaries, credentials)
        full_summary = final_summary_with_openai(chunk_summaries, full_transcript, 
                                                 credentials, save_dir=jobID, video_path=filepath)
        # banner generation
        banner_title = extract_key_line(full_transcript, full_summary, credentials)
        ## pick a representative frame
        res = semantic_search(banner_title, index, credentials, top_k=3)
        print(res)
        valid = [m for m in res.get('matches', []) if m.get('id', '').isdigit()]
        if not valid:
            Exception("No valid scene—re-run analysis.")    
        scene = shots[int(valid[0]['id'])]

        # prepare background image
        frame_path = save_frame(filepath, scene['start'], 'auto_frame.jpg')
        resized_path = 'resized_frame.jpg'
        resize_image(frame_path, resized_path, size=(846, 541))
        if not (is_valid_image(resized_path) and os.path.getsize(resized_path) <= 5 * 1024 * 1024):
            Exception("Invalid or too-large resized frame image")
        bg_data = image_to_base64(resized_path)

        # call Templated API
        api_key = credentials["templated_api_key"]
        template_id = credentials["templated_template_id"]
        
        url = 'https://api.templated.io/v1/render'
        headers = {'Authorization': f'Bearer {api_key}'}
        data = {
            'template': template_id,
            'layers': {
                'shape-2': {},
                'image-2': {
                    'image_url': f'data:image/jpeg;base64,{bg_data}',
                    'width': 846,
                    'height': 541,
                },
                'shape-1': {},
                'text-3': {
                    'text': banner_title,
                    'color': '#2f4f4f',
                    'text_align': 'center',
                    'vertical_align': 'middle'
                }
            }
        }
        # print(data)
        resp = requests.post(url, json=data, headers=headers)
        print('Saving banner')
        print(resp.json())
        if resp.status_code == 200:
            out = resp.json()
            print(out)
            if 'url' in out:
                # download, cache, and display
                img_bytes = requests.get(out['url']).content
                save_banner_dir = f"{jobID}/banner"
                os.mkdir(save_banner_dir)
                save_banner(filepath, img_bytes, save_banner_dir)
            else:
                Exception('Error in banner generation')
        else:
            Exception('Error reponse from banner api')
        
        gcs_folder_path = upload_folder_to_gcs(BUCKET, BUCKET_DIR, jobID)
        return {
            "message": "Processing completed",
            "jobID": jobID
        }, gcs_folder_path, filepath  

    except Exception as e:
        # Clean up on failure
        shutil.rmtree(jobID)
        print(e)