from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import traceback
from google import genai
from google.genai import types
import threading
import time
from tts import OptimizedGeminiCoquiPipeline
from jwt import ExpiredSignatureError, InvalidTokenError
import jwt
from dotenv import load_dotenv
from functools import wraps

load_dotenv()

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

JWT_SECRET = os.getenv("JWT_SECRET")

def verify_token(f):
    """Decorator to verify JWT token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'Authorization header required'}), 401
        
        try:
            # Extract token from "Bearer TOKEN" format
            token = auth_header.split(' ')[1]
        except IndexError:
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        try:
            # Verify and decode the token
            decoded_token = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            
            # Add user info to request context
            request.current_user = decoded_token
            
            return f(*args, **kwargs)
            
        except ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            return jsonify({'error': 'Token verification failed'}), 401
    
    return decorated_function

def check_rate_limit(user_id, max_requests=4, window_minutes=60):
    """Simple rate limiting based on user ID"""
    # In production, use Redis or a proper rate limiting solution
    # This is a simplified example
    from collections import defaultdict
    import time
    
    # Store requests per user (in production, use Redis)
    if not hasattr(check_rate_limit, 'requests'):
        check_rate_limit.requests = defaultdict(list)
    
    now = time.time()
    user_requests = check_rate_limit.requests[user_id]
    
    # Remove old requests outside the window
    cutoff = now - (window_minutes * 60)
    check_rate_limit.requests[user_id] = [req_time for req_time in user_requests if req_time > cutoff]
    
    # Check if user has exceeded rate limit
    if len(check_rate_limit.requests[user_id]) >= max_requests:
        return False
    
    # Add current request
    check_rate_limit.requests[user_id].append(now)
    return True


def extract_text(pdf_file):
    try:
        with open(pdf_file, "rb") as f:  # Note: "rb" for binary mode
            doc_data = f.read()
        prompt = (
            "You are given a large text extracted from a PDF, which includes introductions, exercises, author biography, "
            "and a fictional short story. Extract only the **main fictional story**, excluding any questions, analysis, or introductory content.\n\n"
            "### PDF Content Start ###\n"
            f"{doc_data[:8000]}\n"
            "### PDF Content End ###\n\n"
            "Now extract and return ONLY the main fictional story."
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=doc_data,
                    mime_type='application/pdf',
                ),
                prompt
            ],
            
        )

        text = response.text
        return text
    
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        raise e

@app.route("/v1/generate", methods=["POST"])
@verify_token
def generate_audio():

    path = os.path.dirname(os.path.abspath(__file__))

    upload_pdf_folder = os.path.join(path, "temporary")
    os.makedirs(upload_pdf_folder, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = upload_pdf_folder
    save_pdf_path = os.path.join(app.config.get('UPLOAD_FOLDER'), "temp.pdf")

    upload_audio_folder = os.path.join(path, "temporary_audio")
    os.makedirs(upload_audio_folder, exist_ok=True)
    app.config["UPLOAD_AUDIO_FOLDER"] = upload_audio_folder
    save_audio_path = os.path.join(app.config.get('UPLOAD_AUDIO_FOLDER'), "temp.wav")


    def cleanup_files():
        """Clean up temporary files after sending response"""
        try:
            if save_pdf_path and os.path.exists(save_pdf_path):
                os.unlink(save_pdf_path)
            if save_audio_path and os.path.exists(save_audio_path):
                os.unlink(save_audio_path)
        except Exception as e:
            print(f"Error cleaning up files: {e}")

    try:
        user = request.current_user
        user_id = user.get('userId')
        
        # Rate limiting per user
        if not check_rate_limit(user_id, max_requests=4, window_minutes=60):
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        

        if 'file' not in request.files:
            return jsonify({'message': 'No file provided'}), 400
            
        pdf_file = request.files['file']
        
        if pdf_file.filename == '':
            return jsonify({'message': 'No file selected'}), 400
            
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'message': 'File must be a PDF'}), 400
        
        print(f"Received file: {pdf_file.filename}")
        

        pdf_file.seek(0, 2)  # Seek to end
        file_size = pdf_file.tell()
        pdf_file.seek(0)  # Reset to beginning
        
        print(f"File size: {file_size} bytes")
        
        if file_size < 100:
            return jsonify({'message': 'PDF file appears to be empty or corrupted'}), 400
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return jsonify({'message': 'PDF file is too large. Please use a file smaller than 50MB'}), 400
        
        # Get metadata if provided
        metadata = {}

        if 'metadata' in request.form:
            try:
                metadata = json.loads(request.form['metadata'])
                print(f"Metadata: {metadata}")
            except json.JSONDecodeError:
                print("Invalid metadata JSON, using defaults")

        pdf_file.save(save_pdf_path)
        print(f"Saved PDF to: {save_pdf_path}")
        

        pdf_file.seek(0)  # Reset file pointer
        print("Extracting text from PDF...")
        try:
            extracted_text = extract_text(save_pdf_path)
        except Exception as e:
            cleanup_files()
            error_msg = str(e)
            if "no pages" in error_msg.lower():
                return jsonify({'message': 'The PDF appears to be corrupted or contains no readable pages'}), 400
            elif "invalid_argument" in error_msg.lower():
                return jsonify({'message': 'The PDF format is not supported or the file is corrupted'}), 400
            else:
                return jsonify({'message': f'Could not extract text from PDF: {error_msg}'}), 400

        if not extracted_text or len(extracted_text.strip()) < 10:
            cleanup_files()
            return jsonify({'message': 'Could not extract meaningful text from PDF. The document may be image-based or corrupted.'}), 400
        
        print(f"Extracted text length: {len(extracted_text)} characters")
        print(f"First 200 chars: {extracted_text[:200]}...")
        
        
        # Initialize the TTS pipeline
        try:
            pipeline = OptimizedGeminiCoquiPipeline(
                tts_model="tts_models/en/ljspeech/tacotron2-DDC",
                text=extracted_text,
                use_gpu=True,
                max_workers=4,
                chunk_size=200
            )
        except Exception as e:
            cleanup_files()
            print(f"Error initializing TTS pipeline: {e}")
            return jsonify({'message': 'Failed to initialize text-to-speech engine'}), 500
        

        
        print("Starting audio generation...")
        
        # Generate audio
        try:
            success = pipeline.pdf_to_audiobook_pipeline(
                output_path=save_audio_path,
                chunk_size=350
            )
            
            if not success:
                cleanup_files()
                return jsonify({'message': 'Failed to generate audiobook'}), 500
            
        except Exception as e:
            cleanup_files()
            print(f"Error during audio generation: {e}")
            return jsonify({'message': f'Audio generation failed: {str(e)}'}), 500
        
        # Verify audio file was created
        if not os.path.exists(save_audio_path) or os.path.getsize(save_audio_path) == 0:
            cleanup_files()
            return jsonify({'message': 'Audio file was not generated properly'}), 500
        
        audio_size = os.path.getsize(save_audio_path)
        print(f"Audio generated successfully: {save_audio_path}")
        print(f"Audio file size: {audio_size} bytes")
        
        if save_pdf_path and os.path.exists(save_pdf_path):
                os.unlink(save_pdf_path)
                save_pdf_path = None
        
        def delayed_cleanup():
            time.sleep(30)  # Wait 30 seconds for download to complete
            try:
                if save_audio_path and os.path.exists(save_audio_path):
                    os.unlink(save_audio_path)
                    print(f"Cleaned up audio file: {save_audio_path}")
            except Exception as e:
                print(f"Error cleaning up audio: {e}")
        
        # Start cleanup thread
        threading.Thread(target=delayed_cleanup, daemon=True).start()

        # Return the audio file
        return send_file(
            save_audio_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f"{pdf_file.filename.replace('.pdf', '')}_audiobook.wav"
        )
        
    except Exception as e:
        print(f"Error in creating audio: {str(e)}")
        print(traceback.format_exc())
        
        # Clean up files in case of error
        cleanup_files()
        return jsonify({
            "success": False,
            "message": f"An error occurred: {str(e)}"
        }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"message": "File too large. Please upload a smaller PDF."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"message": "Internal server error. Please try again."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.run(debug=True, port=port, host="0.0.0.0")
