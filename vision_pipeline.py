from google.cloud import vision
from google.oauth2 import service_account
import openai
import json
import dotenv
import os
from typing import List, Dict, Union

config = dotenv.load_dotenv()

class VisionMetaData:
    def __init__(self, credentials_path: str, model_name: str = "gpt-4o"):
        """Initialize the Vision API client with credentials"""
        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = vision.ImageAnnotatorClient(credentials=self.credentials)
        self.model_name = model_name
    
    def analyze_image(self, content : bytes) -> Dict[str, Union[List[str], Dict]]:
        """
        Perform comprehensive image analysis including:
        - Object recognition
        - Scene detection
        - Facial recognition
        - Text detection
        - Web entities
        """
        # with open(image_path, 'rb') as image_file:
        #     content = image_file.read()
        
        image = vision.Image(content=content)
        
        # Configure all features we want to detect
        features = [
            {'type_': vision.Feature.Type.LABEL_DETECTION},       # Objects/scenes
            {'type_': vision.Feature.Type.FACE_DETECTION},       # Faces
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
                if entity.score >= 0.7
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

    # Initialize OpenAI client (make sure to set your API key)
    def get_image_metadata(self, imageData):
        # Prepare the prompt for OpenAI
        vision_data, _ = self.analyze_image(imageData)
        prompt = f"""
        Think step by step and re verify if everything makes sense

        Based on the following image analysis from Google Vision API:
        {json.dumps(vision_data, indent=2)}
        
        Please provide:
        1. Event identification and categorization (what event is this likely from, what type of event)
        2. A detailed description of the image that would be good for searchability. Keep it withing 100 words length.
        3. The main persons / celebrities if applicable
        4. Useful meta data tags for precise searching of image
        
        DONOT hallucinate data yourself. Use the relevant information provided from google vision api. And stick to
        the context of image

        Format your response as a JSON object with these keys:
        - "event_identification" -- add correct event name here
        - "event_category"
        - "image_description"
        - "main_persons"
        - "metadata"
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


# if __name__=="__main__":
#     imageMetaData = VisionMetaData(credentials_path="E:\\my_documents\\demoproject-455507-4848ed3c5d27.json")
#     imageJsonData = imageMetaData.get_image_metadata("images.jfif")