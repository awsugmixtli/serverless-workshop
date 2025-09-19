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

def lambda_handler(event, context):
    """
    Handles POST requests to generate an image using Amazon Bedrock
    and upload it to S3.
    """
    print(f"Received event: {json.dumps(event)}")
    
    # --- 1. Validate Input ---
    try:
        body = json.loads(event.get('body', '{}'))
        prompt = body.get('prompt')

        if not prompt:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Prompt is required.'})
            }
        if not BUCKET_NAME:
            raise ValueError("S3_BUCKET_NAME environment variable is not set.")
        if not YOUR_NAME:
            raise ValueError("YOUR_NAME environment variable is not set.")

    except Exception as e:
        print(f"ERROR processing input: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps({'message': f"Invalid request body: {e}"})
        }

    # --- 2. Invoke Bedrock Model ---
    try:
        # Using Titan Image Generator G1
        model_id = 'amazon.titan-image-generator-v1'
        
        # Construct the payload for the model
        request_body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": 8.0,
                "seed": 0 # Use 0 for reproducibility, or remove for randomness
            }
        }
        
        response = bedrock_runtime.invoke_model(
            body=json.dumps(request_body),
            modelId=model_id,
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        base64_image = response_body['images'][0]
        
    except Exception as e:
        print(f"ERROR invoking Bedrock model: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f"Error generating image: {e}"})
        }

    # --- 3. Upload Image to S3 ---
    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_image)
        
        # Generate a unique filename
        image_key = f"generated-images/{uuid.uuid4()}.png"
        
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=image_key,
            Body=image_data,
            ContentType='image/png'
        )
        
        # Construct the public URL of the image
        # Note: Bucket must be in a region that supports this URL format
        # and not have "Block all public access" enabled at the account level.
        image_url = f"https://{YOUR_NAME}.mixtli.cloud/{image_key}"

    except Exception as e:
        print(f"ERROR uploading to S3: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f"Error saving image: {e}"})
        }
        
    # --- 4. Return Success Response ---
    return {
        'statusCode': 200,
        # IMPORTANT: Configure CORS in API Gateway for this to work from a browser
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        'body': json.dumps({'imageUrl': image_url})
    }
