import boto3
import json
import base64
import os
import uuid

# Initialize clients
s3 = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')

# Environment variables
# Student must configure this in the Lambda settings
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
YOUR_NAME = os.environ.get('YOUR_NAME')

cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,GET,POST'
    }

def generate_image(prompt):
    """Invokes Bedrock to generate an image."""
    model_id = 'amazon.titan-image-generator-v2:0'
    request_body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": 0
        }
    }
    response = bedrock_runtime.invoke_model(
        body=json.dumps(request_body), modelId=model_id,
        contentType='application/json', accept='application/json'
    )
    response_body = json.loads(response['body'].read())
    return response_body['images'][0]

def upload_to_s3(base64_image):
    """Uploads a base64 image to S3 and returns the public URL."""
    image_data = base64.b64decode(base64_image)
    image_key = f"generated-images/{uuid.uuid4()}.png"
    s3.put_object(
        Bucket=BUCKET_NAME, Key=image_key, Body=image_data, ContentType='image/png'
    )
    return f"https://{YOUR_NAME}.mixtli.cloud/{image_key}"

def list_images():
    """Lists all images in the 'generated-images/' prefix in S3."""
    image_urls = []
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix='generated-images/')
        if 'Contents' in response:
            # Sort by last modified date, newest first
            sorted_objects = sorted(response['Contents'], key=lambda obj: obj['LastModified'], reverse=True)
            for obj in sorted_objects:
                key = obj['Key']
                # Ensure it's a file, not a "folder"
                if not key.endswith('/'):
                    image_urls.append(f"https://{YOUR_NAME}.mixtli.cloud/{key}")
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        # Return an empty list or handle error appropriately
    return image_urls

def lambda_handler(event, context):
    """
    Handles API Gateway requests.
    - POST /api/generate-image: Creates a new image.
    - GET /api/images: Lists all existing images.
    """
    print(f"Received event: {json.dumps(event)}")
    http_method = event.get('httpMethod')

    if not BUCKET_NAME:
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({'message': 'S3_BUCKET_NAME environment variable is not set.'})
        }

    if http_method == 'POST':
        print("The request is a POST, generating an image.")
        try:
            body = json.loads(event.get('body', '{}'))
            prompt = body.get('prompt')
            if not prompt:
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({'message': 'Prompt is required.'})
                }
            
            base64_image = generate_image(prompt)
            image_url = upload_to_s3(base64_image)
            
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps({'imageUrl': image_url})
            }
        except Exception as e:
            print(f"ERROR processing POST request: {e}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'message': f'Error generating image: {e}'})
            }

    elif http_method == 'GET':
        print("The request is a GET, listing all the images.")
        try:
            image_urls = list_images()
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps({'images': image_urls})
                }
        except Exception as e:
            print(f"ERROR processing GET request: {e}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'message': f'Error listing images: {e}'})
                }
            
    else:
        return {
            'statusCode': 405,
            'headers': cors_headers,
            'body': json.dumps({'message': 'Method Not Allowed'})
            }
