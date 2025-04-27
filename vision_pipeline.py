from google.cloud import vision_v1 as vision
from google.oauth2 import service_account
import openai
import json
import dotenv
import os
from typing import List, Dict, Union
import boto3
# from PIL import Image
# from io import BytesIO
# config = dotenv.load_dotenv()

class VisionMetaData:
    def __init__(self, credentials_path: str, openai_api_key, aws_access_key, aws_secret_key, 
                 model_name: str = "gpt-4o"):
        """Initialize the Vision API client with credentials"""
        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.client = vision.ImageAnnotatorClient(credentials=self.credentials)
        self.model_name = model_name
                # AWS Rekognition Client
        self.rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'  # Change to your preferred region
        )


    def analyze_image(self, content : bytes) -> Dict[str, Union[List[str], Dict]]:
        """
        Perform comprehensive image analysis including:
        - Object recognition
        - Scene detection
        - Text detection
        - Web entities
        """
        # with open(image_path, 'rb') as image_file:
        #     content = image_file.read()   
        image = vision.Image(content=content)

        
        # Configure all features we want to detect
        features = [
            {'type_': vision.Feature.Type.LABEL_DETECTION},       # Objects/scenes
            {'type_': vision.Feature.Type.LOGO_DETECTION},       # Logos
            {'type_': vision.Feature.Type.TEXT_DETECTION},       # OCR
            {'type_': vision.Feature.Type.WEB_DETECTION},        # Web entities
            {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},  # Object positions
        ]
        
        response = self.client.annotate_image({'image': image, 'features': features})
        
        return self._process_response(response)
    
    def _process_response(self, response) -> Dict[str, Union[List[str], Dict]]:
        """Process the API response into organized tags"""
        tags = {
            'objects': [],
            'scenes': [],
            'text': [],
            'web_entities': [],
            'matching_page_titles': []
        }
        
        # Object and scene labels (confidence > 70%)
        tags['objects'] = [
            label.description for label in response.label_annotations 
            if label.score >= 0.7 and ' ' not in label.description
        ]
        
        tags['scenes'] = [
            label.description for label in response.label_annotations 
            if label.score >= 0.7 and ' ' in label.description
        ]
            
        # Detected text
        if response.text_annotations:
            tags['text'] = [text.description for text in response.text_annotations]
        
        # Web entities
        if hasattr(response, 'web_detection'):
            tags['web_entities'] = [
                entity.description for entity in response.web_detection.web_entities
                if entity.score >= 0.6
            ]
            matching_urls = response.web_detection.pages_with_matching_images
            page_titles = []
            if matching_urls:
                for idx, web_page_object in enumerate(matching_urls):
                    page_titles.append(web_page_object.page_title)
                    if idx > 10:
                        break
                tags['matching_page_titles'] = page_titles
        
        return tags, response
    
    def _aws_celebrity_detection(self, image_content: bytes) -> Dict:
        """Detect celebrities using AWS Rekognition"""
        response = self.rekognition_client.recognize_celebrities(
            Image={'Bytes': image_content}
        )
        
        celebrities = []
        for celebrity in response.get('CelebrityFaces', []):
            celeb_info = {
                'name': celebrity['Name'],
                'confidence': celebrity['MatchConfidence'],
                'urls': celebrity.get('Urls', [])
            }
            celebrities.append(celeb_info)
        
        return {
            'celebrity_faces': celebrities,
            'unrecognized_faces': len(response.get('UnrecognizedFaces', []))
        }

    # Initialize OpenAI client (make sure to set your API key)
    def get_image_metadata(self, imageData):
        # Prepare the prompt for OpenAI
        vision_data, _ = self.analyze_image(imageData)
        print(vision_data)
        prompt = f"""
        Based on the following image analysis from Google Vision API:
        {json.dumps(vision_data, indent=2)}
        Please provide the following information STRICTLY based on the provided data. DO NOT hallucinate or add any extra information:

        1. **Event Identification**: Identify the event name from the entry of 'web_entities' or 'matching_page_titles'. If no clear event is found, state general description.
            using matching page title and web_entities to identify event name. Avoid adding a persons name in event name.
        2. **Event Category**: Categorize the event (e.g., political, cultural, sports) based on the event name or context.
        3. **Image Description**: Write a concise, 100-word description focusing on the event, main persons, and context.
        Use matching page titles to create a better summarized description. Don't add unecessary detials that dont make sense. Don't assume anything if it is not happening from image data.
        Exclude trivial details like facial features unless they are iconic (e.g., a celebrity's beard).
        4. **Main Persons**: List the main persons/celebrities from 'web_entities' or 'matching_page_titles'. If none, state 'None'.
        5. **Metadata Tags**: Provide ONLY highly relevant tags for searchability (e.g., event name, main persons, location). Exclude generic tags like 'beard' or 'sleeve'.

        Format your response as a JSON object with these exact keys:
        - "event_identification"
        - "event_category"
        - "image_description"
        - "main_persons"
        - "metadata_tags"
        Example of a good output:
        {{
        "event_identification": "TeamLab Phenomena Abu Dhabi",
        "event_category": "Cultural Exhibition",
        "image_description": "Khaled bin Mohamed bin Zayed attends the inauguration ceremony of TeamLab Phenomena Abu Dhabi, a cutting-edge digital art exhibition.",
        "main_persons": ["Khaled bin Mohamed bin Zayed"],
        "metadata_tags": ["TeamLab Phenomena", "Abu Dhabi", "Khaled bin Mohamed bin Zayed", "inauguration", "digital art"]
        }}

        Think step by step.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name, 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides image search metadata."},
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


if __name__=="__main__":
    from PIL import Image
    import io
    imageMetaData = VisionMetaData(credentials_path="E:\\my_documents\\demoproject-455507-4848ed3c5d27.json")

    
    img = Image.open("media-processing-test-images\\Screenshot 2025-04-18 223127.jpg")    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    face_content = buffer.getvalue()

    imageJsonData = imageMetaData._aws_celebrity_detection(face_content)