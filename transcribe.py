from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from faster_whisper import WhisperModel
import tempfile, os

model_size = "small"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    print("[DEBUG] Transcribe endpoint called")
    if "audio" not in request.files:
        print("[DEBUG] Error: No audio file in request")
        return jsonify({"error": "File 'audio' tidak ditemukan"}), 400

    file = request.files["audio"]
    print(f"[DEBUG] Received file: {file.filename}, type: {file.content_type}, size: {file.content_length} bytes")
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_file = tmp.name
            file.save(temp_file)
            print(f"[DEBUG] Saved audio to temporary file: {temp_file}")
        
        print("[DEBUG] Starting transcription process...")
        segments, info = model.transcribe(temp_file, language="id", task="transcribe", vad_filter=True, beam_size=5)
        print(f"[DEBUG] Detected language '{info.language}' with probability {info.language_probability}")
        
        text = "".join([seg.text.strip() for seg in segments])
        print(f"[DEBUG] Transcription result:{text}")
        return jsonify({"text": text})
    except Exception as e:
        print(f"[DEBUG] Exception occurred: {type(e).__name__}: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                print(f"[DEBUG] Temporary file {temp_file} removed")
            except Exception as e:
                print(f"[DEBUG] Failed to remove temporary file: {str(e)}")
                pass 

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)