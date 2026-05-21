from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_pymongo import PyMongo
import gridfs
import os
import uuid
import base64
from ultralytics import YOLO
import numpy as np
import cv2
from collections import Counter
from datetime import datetime
from bson import ObjectId

app = Flask(__name__)
CORS(app)

# ============================
# CONFIG MONGODB
# ============================
app.config["MONGO_URI"] = os.environ.get(
    "MONGO_URI", "mongodb://localhost:27017/pest_detector"
)
mongo = PyMongo(app)

def get_fs():
    """GridFS instance (lazy, per-request safe)."""
    return gridfs.GridFS(mongo.db)

# ============================
# MODELO YOLO (lazy load)
# ============================
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

# ============================
# COLORES BGR (generación dinámica por clase)
# ============================
PREDEFINED_COLORS = {
    "Brown Planthopper": (19, 69, 139),
    "Water weevil":      (246, 130, 59),
    "Army worm":         (68, 68, 239),
    "Leaf hopper":       (94, 197, 34),
}

def get_color(class_name, cls_index):
    if class_name in PREDEFINED_COLORS:
        return PREDEFINED_COLORS[class_name]
    hue = (cls_index * 47) % 180
    hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

# ============================
# HELPERS: conversión imagen ↔ bytes (sin disco)
# ============================
def bytes_to_img(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    return img

def img_to_bytes(img: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("No se pudo codificar la imagen")
    return buf.tobytes()

# ============================
# HELPER: dibujar bounding boxes (trabaja sobre np.ndarray)
# ============================
def draw_boxes(img: np.ndarray, yolo_results):
    h_img, w_img, _ = img.shape
    scale  = w_img / 1000
    fscale = max(0.5, 1.2 * scale)
    thick  = max(1, int(4 * scale))
    bthick = max(2, int(6 * scale))
    pad    = int(20 * scale)
    off    = int(10 * scale)

    output = img.copy()
    r      = yolo_results[0]
    counts = Counter()
    detections = []

    if r.boxes and len(r.boxes) > 0:
        for box in r.boxes:
            cls        = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = r.names[cls]
            color      = get_color(class_name, cls)
            counts[class_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, bthick)

            label = f"{class_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick
            )
            cv2.rectangle(
                output,
                (x1, y1 - th - pad), (x1 + tw + pad, y1),
                color, -1
            )
            cv2.putText(
                output, label,
                (x1 + off, y1 - off),
                cv2.FONT_HERSHEY_SIMPLEX,
                fscale, (255, 255, 255), thick
            )

            detections.append({
                "class":      class_name,
                "confidence": round(conf * 100, 2),
                "bbox":       [int(x1), int(y1), int(x2), int(y2)]
            })

    return output, dict(counts), detections


def serialize(doc):
    doc["_id"] = str(doc["_id"])
    for key in ("upload_file_id", "result_file_id"):
        if key in doc:
            doc[key] = str(doc[key])
    return doc

# ============================
# GET /api/images/<file_id>  ← sirve imágenes desde GridFS
# ============================
@app.route("/api/images/<file_id>")
def serve_image(file_id):
    try:
        fs   = get_fs()
        grid = fs.get(ObjectId(file_id))
        return Response(grid.read(), mimetype="image/jpeg")
    except gridfs.errors.NoFile:
        return jsonify({"error": "Imagen no encontrada"}), 404
    except Exception:
        return jsonify({"error": "ID inválido"}), 400

# ============================
# POST /api/analyze
# ============================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    analysis_name = ""

    # ── Leer imagen como bytes ───────────────────────────────────────────
    if "image" in request.files:
        raw_bytes     = request.files["image"].read()
        analysis_name = request.form.get("name", "")
    elif request.is_json and "image_b64" in request.json:
        _, encoded = request.json["image_b64"].split(",", 1)
        raw_bytes     = base64.b64decode(encoded)
        analysis_name = request.json.get("name", "")
    else:
        return jsonify({"error": "No se envió imagen"}), 400

    # ── Decodificar y redimensionar a 416×416 ────────────────────────────
    try:
        img_raw = bytes_to_img(raw_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    img_resized = cv2.resize(img_raw, (416, 416), interpolation=cv2.INTER_LINEAR)

    # ── Inferencia YOLO directamente sobre np.ndarray (sin disco) ────────
    model    = get_model()
    yolo_res = model(img_resized)

    out_img, counts, detections = draw_boxes(img_resized, yolo_res)

    # ── Codificar ambas imágenes a bytes ─────────────────────────────────
    try:
        upload_bytes = img_to_bytes(img_resized)
        result_bytes = img_to_bytes(out_img)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    # ── Guardar en GridFS ─────────────────────────────────────────────────
    fs       = get_fs()
    filename = f"{uuid.uuid4()}.jpg"

    upload_id = fs.put(
        upload_bytes,
        filename=f"upload_{filename}",
        content_type="image/jpeg"
    )
    result_id = fs.put(
        result_bytes,
        filename=f"result_{filename}",
        content_type="image/jpeg"
    )

    # ── Persistir metadatos en MongoDB ────────────────────────────────────
    doc = {
        "name":           analysis_name or f"Análisis {datetime.utcnow().strftime('%d/%m/%Y %H:%M')}",
        "filename":       filename,
        "upload_file_id": upload_id,
        "result_file_id": result_id,
        "upload_url":     f"/api/images/{upload_id}",
        "result_url":     f"/api/images/{result_id}",
        "counts":         counts,
        "detections":     detections,
        "total_objects":  sum(counts.values()),
        "created_at":     datetime.utcnow().isoformat()
    }
    res = mongo.db.analyses.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    doc["upload_file_id"] = str(upload_id)
    doc["result_file_id"] = str(result_id)

    return jsonify(doc), 200

# ============================
# GET /api/history
# ============================
@app.route("/api/history", methods=["GET"])
def history():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 12))
    skip     = (page - 1) * per_page
    total    = mongo.db.analyses.count_documents({})
    docs     = list(
        mongo.db.analyses.find()
        .sort("created_at", -1)
        .skip(skip)
        .limit(per_page)
    )
    return jsonify({
        "total":   total,
        "page":    page,
        "pages":   (total + per_page - 1) // per_page,
        "results": [serialize(d) for d in docs]
    })

# ============================
# GET /api/history/<id>
# ============================
@app.route("/api/history/<aid>", methods=["GET"])
def get_analysis(aid):
    try:
        doc = mongo.db.analyses.find_one({"_id": ObjectId(aid)})
    except Exception:
        return jsonify({"error": "ID inválido"}), 400
    if not doc:
        return jsonify({"error": "No encontrado"}), 404
    return jsonify(serialize(doc))

# ============================
# DELETE /api/history/<id>
# ============================
@app.route("/api/history/<aid>", methods=["DELETE"])
def delete_analysis(aid):
    try:
        doc = mongo.db.analyses.find_one({"_id": ObjectId(aid)})
    except Exception:
        return jsonify({"error": "ID inválido"}), 400
    if not doc:
        return jsonify({"error": "No encontrado"}), 404

    # ── Borrar archivos de GridFS ─────────────────────────────────────────
    fs = get_fs()
    for key in ("upload_file_id", "result_file_id"):
        fid = doc.get(key)
        if fid:
            try:
                fs.delete(ObjectId(fid) if not isinstance(fid, ObjectId) else fid)
            except Exception:
                pass

    mongo.db.analyses.delete_one({"_id": ObjectId(aid)})
    return jsonify({"message": "Eliminado"})

# ============================
# DELETE /api/history (borrar todo)
# ============================
@app.route("/api/history", methods=["DELETE"])
def delete_all_history():
    try:
        fs   = get_fs()
        docs = list(mongo.db.analyses.find({}, {"upload_file_id": 1, "result_file_id": 1}))

        for doc in docs:
            for key in ("upload_file_id", "result_file_id"):
                fid = doc.get(key)
                if fid:
                    try:
                        fs.delete(ObjectId(fid) if not isinstance(fid, ObjectId) else fid)
                    except Exception:
                        pass

        mongo.db.analyses.delete_many({})
        return jsonify({"message": "Historial eliminado"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================
# GET /api/stats
# ============================
@app.route("/api/stats", methods=["GET"])
def stats():
    total    = mongo.db.analyses.count_documents({})
    pipeline = [
        {"$unwind": "$detections"},
        {"$group": {"_id": "$detections.class", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    dist = list(mongo.db.analyses.aggregate(pipeline))
    return jsonify({
        "total_analyses":     total,
        "class_distribution": [{"class": c["_id"], "count": c["count"]} for c in dist]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)