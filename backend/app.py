"""
PlantDoc — app.py  (v6.0)
==========================
Backend for plant leaf disease detection.

Architecture
------------
  Gatekeeper : YOLOv8-cls binary model  (model/leaf_detector.pt)
               → leaf     : forwarded to CNN disease model
               → non-leaf : rejected with HTTP 400

  Disease CNN : TensorFlow/Keras model  (model/leaf_disease_model.h5)

  Recommendations : Gemini 1.5 Flash (falls back gracefully to static text)

Run
---
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

License: MIT
"""

import io
import json
import logging
import os
import time

import numpy as np
from contextlib import asynccontextmanager
from PIL import Image

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from utils.image_utils import preprocess_image

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plantdoc")

# ── Environment ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ .env loaded")
except ImportError:
    pass

YOLO_MODEL_PATH     = os.getenv("YOLO_MODEL_PATH", "model/leaf_detector.pt")
LEAF_CONF_THRESHOLD = float(os.getenv("LEAF_CONFIDENCE_THRESHOLD", "0.60"))
MAX_FILE_SIZE_MB    = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_ORIGINS     = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Globals (populated during startup) ────────────────────────────────────────
disease_model = None
yolo_model    = None
CLASS_NAMES: list = []
GEMINI_AVAILABLE  = False
gemini_client     = None


def _load_classes():
    global CLASS_NAMES
    path = "classes.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"classes.txt not found at '{path}'")
    with open(path, encoding="utf-8") as f:
        CLASS_NAMES = [ln.strip() for ln in f if ln.strip()]
    if not CLASS_NAMES:
        raise ValueError("classes.txt is empty")
    logger.info(f"✅ {len(CLASS_NAMES)} disease classes loaded")


def _load_disease_model():
    global disease_model
    try:
        import tf_keras
        disease_model = tf_keras.models.load_model("model/leaf_disease_model.h5")
        logger.info("✅ Disease CNN loaded via tf_keras")
        return
    except Exception:
        pass
    try:
        import tensorflow as tf
        disease_model = tf.keras.models.load_model("model/leaf_disease_model.h5")
        logger.info("✅ Disease CNN loaded via tf.keras")
    except Exception as exc:
        logger.critical(f"❌ Disease CNN failed: {exc}")
        raise RuntimeError(f"Cannot load disease model: {exc}") from exc


def _load_yolo():
    global yolo_model
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"✅ YOLO leaf detector loaded (threshold={LEAF_CONF_THRESHOLD:.0%})")
    except FileNotFoundError:
        logger.critical(f"❌ YOLO model not found at '{YOLO_MODEL_PATH}'")
        raise
    except ImportError:
        logger.critical("❌ ultralytics not installed — pip install ultralytics")
        raise
    except Exception as exc:
        logger.critical(f"❌ YOLO failed: {exc}")
        raise


def _load_gemini():
    global GEMINI_AVAILABLE, gemini_client
    try:
        from google import genai as google_genai
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            gemini_client    = google_genai.Client(api_key=key)
            GEMINI_AVAILABLE = True
            logger.info("✅ Gemini AI ready")
        else:
            logger.warning("⚠️  No GEMINI_API_KEY — static fallback active")
    except Exception as exc:
        logger.warning(f"⚠️  Gemini init failed ({exc}) — static fallback active")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 PlantDoc starting up…")
    _load_classes()
    _load_disease_model()
    _load_yolo()
    _load_gemini()
    logger.info("🌿 PlantDoc is ready")
    yield
    logger.info("🛑 PlantDoc shutting down")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PlantDoc API",
    description=(
        "Upload a leaf image to detect plant diseases and get treatment advice.\n\n"
        "**Stack:** FastAPI · TensorFlow CNN · YOLOv8 · Gemini 1.5 Flash  \n"
        "**License:** MIT"
    ),
    version="6.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(ms)
    return response


# ── Leaf check ─────────────────────────────────────────────────────────────────
def check_leaf(image: Image.Image) -> tuple:
    """
    YOLO binary classifier.
    Returns (is_leaf: bool, leaf_confidence: float 0-1).
    """
    results     = yolo_model.predict(image, verbose=False)
    r           = results[0]
    class_names = r.names

    leaf_conf = 0.0
    for idx, name in class_names.items():
        if name.lower() == "leaf":
            leaf_conf = float(r.probs.data[idx])
            break

    top1_name = class_names[int(r.probs.top1)]
    top1_conf = float(r.probs.top1conf)
    is_leaf   = leaf_conf >= LEAF_CONF_THRESHOLD

    logger.info(
        f"[YOLO] {'✅ leaf' if is_leaf else '❌ non-leaf'} | "
        f"top1={top1_name} ({top1_conf:.2%}) | "
        f"leaf_conf={leaf_conf:.2%} | thresh={LEAF_CONF_THRESHOLD:.2%}"
    )
    return is_leaf, leaf_conf


# ── Schemas ────────────────────────────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    disease_name: str
    language: str = "en"

    @field_validator("disease_name")
    @classmethod
    def validate_disease(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("disease_name must not be empty")
        if len(v) > 200:
            raise ValueError("disease_name exceeds 200 characters")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        allowed = {"en", "te", "hi", "ml", "ta"}
        v = v.strip().lower()
        return v if v in allowed else "en"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """System info and health check."""
    return {
        "status":  "running",
        "version": "6.0",
        "classes": len(CLASS_NAMES),
        "gemini":  GEMINI_AVAILABLE,
        "gatekeeper": {
            "model":     YOLO_MODEL_PATH,
            "threshold": LEAF_CONF_THRESHOLD,
        },
    }


@app.get("/health", tags=["Health"])
def health():
    """Lightweight liveness check for load balancers / uptime monitors."""
    return {"status": "ok"}


@app.get("/classes", tags=["Info"])
def get_classes():
    """Return all disease class names the model can detect."""
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a plant leaf image and receive a disease prediction.

    - **file**: JPG / PNG / WebP — max 10 MB
    - Returns disease name, confidence %, and leaf gatekeeper confidence.
    """
    ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload JPG, PNG, or WebP.",
        )

    try:
        contents  = await file.read()
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
            )

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=422, detail="Cannot open image — file may be corrupt.")

        # Step 1 — YOLO leaf gatekeeper
        is_leaf, leaf_conf = check_leaf(image)
        if not is_leaf:
            return JSONResponse(
                status_code=400,
                content={
                    "status":          "rejected",
                    "leaf_confidence": round(leaf_conf, 4),
                    "message":         "❌ This does not appear to be a plant leaf. Please upload a clear leaf photo.",
                },
            )

        # Step 2 — CNN disease prediction
        img      = preprocess_image(image)
        preds    = disease_model.predict(img, verbose=0)[0]
        idx      = int(np.argmax(preds))
        conf_pct = float(preds[idx]) * 100

        logger.info(f"[CNN] {CLASS_NAMES[idx]} ({conf_pct:.2f}%)")

        return {
            "status":          "success",
            "disease":         CLASS_NAMES[idx],
            "confidence":      round(conf_pct, 2),
            "leaf_confidence": round(leaf_conf * 100, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Predict error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error. Please try again."},
        )


@app.post("/recommendations", tags=["Recommendations"])
async def recommendations(req: RecommendationRequest):
    """
    Get treatment recommendations for a detected disease.

    - **disease_name**: The disease string returned by /predict
    - Returns pesticide, organic, and prevention advice + YouTube search links.
    """
    disease  = req.disease_name
    language = req.language

    LANGUAGE_NAMES = {
        "en": "English",
        "hi": "Hindi",
        "te": "Telugu",
        "ml": "Malayalam",
        "ta": "Tamil",
    }
    lang_name = LANGUAGE_NAMES.get(language, "English")

    if GEMINI_AVAILABLE and gemini_client is not None:
        try:
            prompt = (
                f'You are a plant pathology expert. The detected disease is: "{disease}".\n\n'
                f'IMPORTANT: Respond entirely in {lang_name} language. All text including names, descriptions, and steps must be in {lang_name}.\n\n'
                "Respond ONLY with a valid JSON object (no markdown, no code fences).\n"
                "Structure:\n"
                "{\n"
                '  "pesticide":  { "name": "...", "description": "...", "steps": ["..."] },\n'
                '  "organic":    { "name": "...", "description": "...", "steps": ["..."] },\n'
                '  "prevention": { "name": "...", "description": "...", "steps": ["..."] },\n'
                '  "youtube_links": [{ "title": "...", "url": "https://..." }]\n'
                "}"
            )
            response = gemini_client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt,
            )
            text = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(text)
        except Exception as exc:
            logger.warning(f"Gemini error: {exc} — using static fallback")

    return _static_recommendations(disease, language)


# ── Static fallback recommendations ───────────────────────────────────────────
def _static_recommendations(disease: str, language: str = "en") -> dict:
    is_healthy = "healthy" in disease.lower()
    q          = disease.replace(" ", "+")

    STATIC_STRINGS = {
        "en": {
            "no_pesticide":    "No pesticide needed",
            "healthy_pest":    "Plant appears healthy — no chemical treatment required.",
            "healthy_steps":   ["Continue your current care routine", "Monitor leaves weekly for early signs", "Ensure proper watering and sunlight"],
            "neem_name":       "Preventive Neem Oil Spray",
            "neem_desc":       "Apply neem oil every 2 weeks as a natural preventive.",
            "neem_steps":      ["Mix 5 ml neem oil with 1 litre of water", "Add 2 drops of dish soap and shake", "Spray lightly on leaves every 14 days"],
            "gen_health":      "General Plant Health",
            "gen_health_desc": "Keep your plant strong to resist future disease.",
            "gen_steps":       ["Water at the base, not overhead", "Remove dead or yellowing leaves promptly", "Ensure good airflow between plants"],
            "healthy_yt":      "How to keep plants healthy",
            "fungicide":       f"Fungicide/Bactericide for {disease}",
            "fungicide_desc":  "Apply suitable chemical treatment based on disease type.",
            "fungicide_steps": ["Identify whether disease is fungal, bacterial, or viral", "Purchase appropriate fungicide from a garden store", "Mix recommended dosage per label instructions", "Spray all affected leaf surfaces (top and bottom)", "Repeat every 7–10 days until symptoms clear", "Wear gloves and mask during application"],
            "neem2_name":      "Neem Oil Spray",
            "neem2_desc":      "A natural broad-spectrum solution effective against most plant diseases.",
            "neem2_steps":     ["Mix 5 ml pure neem oil per 1 litre of water", "Add 2–3 drops of liquid dish soap as emulsifier", "Shake well and pour into a spray bottle", "Spray all leaf surfaces including undersides", "Apply in the evening to prevent leaf burn", "Repeat every 5–7 days for 3 weeks"],
            "prev_name":       "Integrated Disease Management",
            "prev_desc":       "Stop the disease spreading and prevent future outbreaks.",
            "prev_steps":      ["Remove and destroy all visibly infected leaves", "Do not compost infected material — bag and dispose", "Avoid overhead watering; water at soil level only", "Space plants properly to improve air circulation", "Sterilize pruning tools with alcohol between cuts", "Rotate crops each season to break disease cycles"],
            "yt1":             f"How to treat {disease}",
            "yt2":             f"{disease} — identification & cure",
        },
        "hi": {
            "no_pesticide":    "कोई कीटनाशक आवश्यक नहीं",
            "healthy_pest":    "पौधा स्वस्थ दिखता है — कोई रासायनिक उपचार आवश्यक नहीं।",
            "healthy_steps":   ["अपनी वर्तमान देखभाल दिनचर्या जारी रखें", "प्रारंभिक संकेतों के लिए साप्ताहिक पत्तियों की निगरानी करें", "उचित पानी और धूप सुनिश्चित करें"],
            "neem_name":       "निवारक नीम तेल स्प्रे",
            "neem_desc":       "प्राकृतिक रोकथाम के रूप में हर 2 सप्ताह में नीम तेल लगाएं।",
            "neem_steps":      ["1 लीटर पानी में 5 मिली नीम तेल मिलाएं", "2 बूंद डिश साबुन मिलाएं और हिलाएं", "हर 14 दिन में पत्तियों पर हल्का स्प्रे करें"],
            "gen_health":      "सामान्य पौधों का स्वास्थ्य",
            "gen_health_desc": "भविष्य की बीमारी से लड़ने के लिए अपने पौधे को मजबूत रखें।",
            "gen_steps":       ["आधार पर पानी दें, ऊपर से नहीं", "मृत या पीली पत्तियों को तुरंत हटाएं", "पौधों के बीच अच्छा वायु संचार सुनिश्चित करें"],
            "healthy_yt":      "पौधों को स्वस्थ कैसे रखें",
            "fungicide":       f"{disease} के लिए कवकनाशी/जीवाणुनाशक",
            "fungicide_desc":  "रोग के प्रकार के आधार पर उपयुक्त रासायनिक उपचार लागू करें।",
            "fungicide_steps": ["पहचानें कि रोग कवक, जीवाणु या वायरल है", "बगीचे की दुकान से उचित कवकनाशी खरीदें", "लेबल निर्देशों के अनुसार अनुशंसित खुराक मिलाएं", "सभी प्रभावित पत्ती सतहों पर स्प्रे करें (ऊपर और नीचे)", "लक्षण साफ होने तक हर 7-10 दिन में दोहराएं", "उपयोग के दौरान दस्ताने और मास्क पहनें"],
            "neem2_name":      "नीम तेल स्प्रे",
            "neem2_desc":      "अधिकांश पौधों की बीमारियों के खिलाफ प्रभावी प्राकृतिक समाधान।",
            "neem2_steps":     ["1 लीटर पानी में 5 मिली शुद्ध नीम तेल मिलाएं", "2-3 बूंद तरल डिश साबुन इमल्सीफायर के रूप में मिलाएं", "अच्छी तरह हिलाएं और स्प्रे बोतल में डालें", "निचली सतह सहित सभी पत्ती सतहों पर स्प्रे करें", "पत्ती जलने से बचाने के लिए शाम को लगाएं", "3 सप्ताह तक हर 5-7 दिन में दोहराएं"],
            "prev_name":       "एकीकृत रोग प्रबंधन",
            "prev_desc":       "रोग को फैलने से रोकें और भविष्य के प्रकोप को रोकें।",
            "prev_steps":      ["सभी दृश्यमान संक्रमित पत्तियों को हटाएं और नष्ट करें", "संक्रमित सामग्री को खाद में न डालें — बैग में बंद करके फेंकें", "ऊपर से पानी देने से बचें; केवल मिट्टी के स्तर पर पानी दें", "वायु संचार के लिए पौधों को उचित दूरी पर रखें", "कटाई के उपकरणों को उपयोगों के बीच अल्कोहल से साफ करें", "रोग चक्र तोड़ने के लिए हर मौसम में फसल बदलें"],
            "yt1":             f"{disease} का उपचार कैसे करें",
            "yt2":             f"{disease} — पहचान और उपचार",
        },
        "te": {
            "no_pesticide":    "పురుగుమందు అవసరం లేదు",
            "healthy_pest":    "మొక్క ఆరోగ్యంగా కనిపిస్తోంది — రసాయన చికిత్స అవసరం లేదు.",
            "healthy_steps":   ["మీ ప్రస్తుత సంరక్షణ దినచర్యను కొనసాగించండి", "ప్రారంభ సంకేతాల కోసం వారానికి ఆకులను పర్యవేక్షించండి", "సరైన నీరు మరియు సూర్యకాంతి నిర్ధారించండి"],
            "neem_name":       "నివారణ వేప నూనె స్ప్రే",
            "neem_desc":       "సహజ నివారణగా ప్రతి 2 వారాలకు వేప నూనె వేయండి.",
            "neem_steps":      ["1 లీటరు నీటిలో 5 మి.లీ వేప నూనె కలపండి", "2 చుక్కల డిష్ సోప్ కలిపి కదిలించండి", "ప్రతి 14 రోజులకు ఆకులపై తేలికగా స్ప్రే చేయండి"],
            "gen_health":      "సాధారణ మొక్క ఆరోగ్యం",
            "gen_health_desc": "భవిష్యత్ వ్యాధిని నిరోధించడానికి మొక్కను బలంగా ఉంచండి.",
            "gen_steps":       ["అడుగు భాగంలో నీరు పోయండి, పైనుండి కాదు", "చనిపోయిన లేదా పసుపు ఆకులను వెంటనే తొలగించండి", "మొక్కల మధ్య మంచి గాలి ప్రసరణ నిర్ధారించండి"],
            "healthy_yt":      "మొక్కలను ఆరోగ్యంగా ఎలా ఉంచాలి",
            "fungicide":       f"{disease} కోసం శిలీంద్రనాశిని/బాక్టీరియానాశిని",
            "fungicide_desc":  "వ్యాధి రకం ఆధారంగా తగిన రసాయన చికిత్స వేయండి.",
            "fungicide_steps": ["వ్యాధి శిలీంద్రం, బాక్టీరియా లేదా వైరల్ అని గుర్తించండి", "తోట దుకాణం నుండి తగిన శిలీంద్రనాశిని కొనండి", "లేబుల్ సూచనల ప్రకారం మోతాదు కలపండి", "అన్ని ప్రభావిత ఆకు ఉపరితలాలపై స్ప్రే చేయండి", "లక్షణాలు తగ్గే వరకు ప్రతి 7-10 రోజులకు పునరావృతం చేయండి", "వేసేటప్పుడు చేతి తొడుగులు మరియు మాస్క్ ధరించండి"],
            "neem2_name":      "వేప నూనె స్ప్రే",
            "neem2_desc":      "చాలా మొక్కల వ్యాధులకు ప్రభావవంతమైన సహజ పరిష్కారం.",
            "neem2_steps":     ["1 లీటరు నీటిలో 5 మి.లీ స్వచ్ఛమైన వేప నూనె కలపండి", "ఎమల్సిఫైయర్‌గా 2-3 చుక్కల ద్రవ డిష్ సోప్ కలపండి", "బాగా కదిలించి స్ప్రే బాటిల్‌లో పోయండి", "అన్ని ఆకు ఉపరితలాలపై స్ప్రే చేయండి", "ఆకు కాలకుండా సాయంత్రం వేయండి", "3 వారాలు ప్రతి 5-7 రోజులకు పునరావృతం చేయండి"],
            "prev_name":       "సమగ్ర వ్యాధి నిర్వహణ",
            "prev_desc":       "వ్యాధి వ్యాప్తిని ఆపండి మరియు భవిష్యత్ వ్యాప్తిని నివారించండి.",
            "prev_steps":      ["దృశ్యమానంగా సోకిన అన్ని ఆకులను తొలగించి నాశనం చేయండి", "సోకిన పదార్థాన్ని కంపోస్ట్ చేయవద్దు — బ్యాగ్‌లో వేసి పారేయండి", "పైనుండి నీరు పోయడం మానుకోండి", "గాలి ప్రసరణ కోసం మొక్కలను సరిగ్గా అంతరాలలో నాటండి", "కత్తిరింపు పరికరాలను ఆల్కహాల్‌తో శుభ్రం చేయండి", "వ్యాధి చక్రాలు తెంచడానికి ప్రతి సీజన్‌లో పంటలు మారండి"],
            "yt1":             f"{disease} చికిత్స ఎలా చేయాలి",
            "yt2":             f"{disease} — గుర్తింపు మరియు నివారణ",
        },
        "ml": {
            "no_pesticide":    "കീടനാശിനി ആവശ്യമില്ല",
            "healthy_pest":    "സസ്യം ആരോഗ്യകരമായി കാണപ്പെടുന്നു — രാസ ചികിത്സ ആവശ്യമില്ല.",
            "healthy_steps":   ["നിലവിലെ പരിചരണ ദിനചര്യ തുടരുക", "ആദ്യ ലക്ഷണങ്ങൾക്കായി ആഴ്ചതോറും ഇലകൾ നിരീക്ഷിക്കുക", "ശരിയായ ജലസേചനവും സൂര്യപ്രകാശവും ഉറപ്പാക്കുക"],
            "neem_name":       "പ്രതിരോധ വേപ്പ് എണ്ണ സ്‌പ്രേ",
            "neem_desc":       "പ്രകൃതിദത്ത പ്രതിരോധമായി 2 ആഴ്ചയിലൊരിക്കൽ വേപ്പ് എണ്ണ തളിക്കുക.",
            "neem_steps":      ["1 ലിറ്റർ വെള്ളത്തിൽ 5 മില്ലി വേപ്പ് എണ്ണ ചേർക്കുക", "2 തുള്ളി ഡിഷ് സോപ്പ് ചേർത്ത് കുലുക്കുക", "ഓരോ 14 ദിവസവും ഇലകളിൽ ലഘുവായി സ്‌പ്രേ ചെയ്യുക"],
            "gen_health":      "പൊതുവായ സസ്യ ആരോഗ്യം",
            "gen_health_desc": "ഭാവിയിലെ രോഗത്തെ ചെറുക്കാൻ സസ്യത്തെ ശക്തമാക്കുക.",
            "gen_steps":       ["ചുവട്ടിൽ വെള്ളമൊഴിക്കുക, മുകളിൽ നിന്നല്ല", "ഉണങ്ങിയ അല്ലെങ്കിൽ മഞ്ഞ ഇലകൾ ഉടൻ നീക്കം ചെയ്യുക", "സസ്യങ്ങൾ തമ്മിൽ നല്ല വായു സഞ്ചാരം ഉറപ്പാക്കുക"],
            "healthy_yt":      "സസ്യങ്ങളെ ആരോഗ്യകരമായി നിലനിർത്തുന്നത് എങ്ങനെ",
            "fungicide":       f"{disease} നുള്ള കുമിൾനാശിനി/ബാക്ടീരിയനാശിനി",
            "fungicide_desc":  "രോഗത്തിന്റെ തരത്തിന് അനുസരിച്ച് ഉചിതമായ രാസ ചികിത്സ നൽകുക.",
            "fungicide_steps": ["രോഗം കുമിൾ, ബാക്ടീരിയ അല്ലെങ്കിൽ വൈറൽ ആണോ എന്ന് തിരിച്ചറിയുക", "ഒരു ഗാർഡൻ സ്റ്റോറിൽ നിന്ന് ഉചിതമായ കുമിൾനാശിനി വാങ്ങുക", "ലേബൽ നിർദ്ദേശങ്ങൾ അനുസരിച്ച് അളവ് ചേർക്കുക", "ബാധിക്കപ്പെട്ട എല്ലാ ഇല ഉപരിതലങ്ങളിലും സ്‌പ്രേ ചെയ്യുക", "ലക്ഷണങ്ങൾ മാറുന്നതുവരെ ഓരോ 7-10 ദിവസവും ആവർത്തിക്കുക", "ഉപയോഗ സമയത്ത് കൈയ്യുറകളും മാസ്കും ധരിക്കുക"],
            "neem2_name":      "വേപ്പ് എണ്ണ സ്‌പ്രേ",
            "neem2_desc":      "മിക്ക സസ്യ രോഗങ്ങൾക്കും ഫലപ്രദമായ പ്രകൃതിദത്ത പരിഹാരം.",
            "neem2_steps":     ["1 ലിറ്റർ വെള്ളത്തിൽ 5 മില്ലി ശുദ്ധ വേപ്പ് എണ്ണ ചേർക്കുക", "ഇമൽസിഫയറായി 2-3 തുള്ളി ദ്രാവക ഡിഷ് സോപ്പ് ചേർക്കുക", "നന്നായി കുലുക്കി സ്‌പ്രേ ബോട്ടിലിൽ ഒഴിക്കുക", "അടിഭാഗം ഉൾപ്പെടെ എല്ലാ ഇല ഉപരിതലങ്ങളിലും സ്‌പ്രേ ചെയ്യുക", "ഇല കരിയാതിരിക്കാൻ വൈകുന്നേരം പ്രയോഗിക്കുക", "3 ആഴ്ച ഓരോ 5-7 ദിവസവും ആവർത്തിക്കുക"],
            "prev_name":       "സമഗ്ര രോഗ നിയന്ത്രണം",
            "prev_desc":       "രോഗം പടരുന്നത് തടയുകയും ഭാവിയിലെ പൊട്ടിപ്പുറപ്പാടുകൾ തടയുകയും ചെയ്യുക.",
            "prev_steps":      ["ദൃശ്യമായ എല്ലാ ബാധിത ഇലകളും നീക്കം ചെയ്ത് നശിപ്പിക്കുക", "ബാധിത വസ്തുക്കൾ കമ്പോസ്റ്റ് ചെയ്യരുത് — ബാഗിൽ ഇട്ട് നീക്കം ചെയ്യുക", "മുകളിൽ നിന്ന് നനയ്ക്കുന്നത് ഒഴിവാക്കുക", "വായു സഞ്ചാരത്തിനായി സസ്യങ്ങൾ ഉചിതമായ അകലത്തിൽ നടുക", "ഉപകരണങ്ങൾ ഓരോ ഉപയോഗത്തിനിടയിലും ആൽക്കഹോൾ കൊണ്ട് വൃത്തിയാക്കുക", "രോഗ ചക്രങ്ങൾ തകർക്കാൻ ഓരോ സീസണിലും വിളകൾ മാറ്റുക"],
            "yt1":             f"{disease} ചികിത്സിക്കുന്നത് എങ്ങനെ",
            "yt2":             f"{disease} — തിരിച്ചറിയലും ചികിത്സയും",
        },
        "ta": {
            "no_pesticide":    "பூச்சிக்கொல்லி தேவையில்லை",
            "healthy_pest":    "தாவரம் ஆரோக்கியமாக தெரிகிறது — வேதியியல் சிகிச்சை தேவையில்லை.",
            "healthy_steps":   ["உங்கள் தற்போதைய பராமரிப்பை தொடரவும்", "ஆரம்ப அறிகுறிகளுக்கு வாரந்தோறும் இலைகளை கண்காணிக்கவும்", "சரியான நீர்ப்பாசனம் மற்றும் சூரிய ஒளி உறுதி செய்யவும்"],
            "neem_name":       "தடுப்பு வேப்ப எண்ணெய் தெளிப்பு",
            "neem_desc":       "இயற்கை தடுப்பாக 2 வாரங்களுக்கு ஒரு முறை வேப்ப எண்ணெய் தெளிக்கவும்.",
            "neem_steps":      ["1 லிட்டர் நீரில் 5 மிலி வேப்ப எண்ணெய் கலக்கவும்", "2 சொட்டு டிஷ் சோப்பு சேர்த்து குலுக்கவும்", "ஒவ்வொரு 14 நாட்களுக்கும் இலைகளில் லேசாக தெளிக்கவும்"],
            "gen_health":      "பொதுவான தாவர ஆரோக்கியம்",
            "gen_health_desc": "எதிர்கால நோய்களை எதிர்க்க தாவரத்தை வலுவாக வைத்திருக்கவும்.",
            "gen_steps":       ["அடிப்பகுதியில் தண்ணீர் ஊற்றவும், மேலிருந்து அல்ல", "இறந்த அல்லது மஞ்சள் இலைகளை உடனடியாக அகற்றவும்", "தாவரங்களுக்கிடையே நல்ல காற்று சஞ்சாரம் உறுதி செய்யவும்"],
            "healthy_yt":      "தாவரங்களை ஆரோக்கியமாக வைப்பது எப்படி",
            "fungicide":       f"{disease} க்கான பூஞ்சைக்கொல்லி/பாக்டீரியாக்கொல்லி",
            "fungicide_desc":  "நோயின் வகையை பொறுத்து பொருத்தமான வேதியியல் சிகிச்சை அளிக்கவும்.",
            "fungicide_steps": ["நோய் பூஞ்சை, பாக்டீரியா அல்லது வைரல் என்று கண்டறியவும்", "தோட்ட கடையில் இருந்து பொருத்தமான பூஞ்சைக்கொல்லி வாங்கவும்", "லேபிள் வழிமுறைகளின்படி அளவை கலக்கவும்", "பாதிக்கப்பட்ட இலை மேற்பரப்புகளில் தெளிக்கவும்", "அறிகுறிகள் மறையும் வரை ஒவ்வொரு 7-10 நாட்களுக்கும் திரும்பவும்", "பயன்படுத்தும் போது கையுறைகள் மற்றும் மாஸ்க் அணியவும்"],
            "neem2_name":      "வேப்ப எண்ணெய் தெளிப்பு",
            "neem2_desc":      "பெரும்பாலான தாவர நோய்களுக்கு எதிரான இயற்கை தீர்வு.",
            "neem2_steps":     ["1 லிட்டர் நீரில் 5 மிலி தூய வேப்ப எண்ணெய் கலக்கவும்", "ஈமல்சிஃபையராக 2-3 சொட்டு திரவ டிஷ் சோப்பு சேர்க்கவும்", "நன்றாக குலுக்கி ஸ்ப்ரே பாட்டிலில் ஊற்றவும்", "அனைத்து இலை மேற்பரப்புகளிலும் தெளிக்கவும்", "இலை எரிவதை தடுக்க மாலையில் பயன்படுத்தவும்", "3 வாரங்கள் ஒவ்வொரு 5-7 நாட்களுக்கும் திரும்பவும்"],
            "prev_name":       "ஒருங்கிணைந்த நோய் மேலாண்மை",
            "prev_desc":       "நோய் பரவுவதை நிறுத்தவும் எதிர்கால வெடிப்புகளை தடுக்கவும்.",
            "prev_steps":      ["தெரியும் அனைத்து பாதிக்கப்பட்ட இலைகளையும் அகற்றி அழிக்கவும்", "பாதிக்கப்பட்ட பொருட்களை உரமாக்க வேண்டாம்", "மேலிருந்து நீர் பாய்ச்சுவதை தவிர்க்கவும்", "காற்று சஞ்சாரத்திற்காக தாவரங்களை சரியான இடைவெளியில் வைக்கவும்", "கத்தரிக்கும் கருவிகளை ஒவ்வொரு முறையும் ஆல்கஹால் கொண்டு சுத்தம் செய்யவும்", "நோய் சுழற்சிகளை உடைக்க ஒவ்வொரு பருவமும் பயிர்களை மாற்றவும்"],
            "yt1":             f"{disease} சிகிச்சை செய்வது எப்படி",
            "yt2":             f"{disease} — அடையாளம் மற்றும் நிவாரணம்",
        },
    }

    s = STATIC_STRINGS.get(language, STATIC_STRINGS["en"])

    if is_healthy:
        return {
            "pesticide": {"name": s["no_pesticide"],    "description": s["healthy_pest"], "steps": s["healthy_steps"]},
            "organic":   {"name": s["neem_name"],       "description": s["neem_desc"],    "steps": s["neem_steps"]},
            "prevention":{"name": s["gen_health"],      "description": s["gen_health_desc"], "steps": s["gen_steps"]},
            "youtube_links": [{"title": s["healthy_yt"], "url": "https://www.youtube.com/results?search_query=healthy+plant+care+tips"}],
        }

    return {
        "pesticide": {"name": s["fungicide"],  "description": s["fungicide_desc"], "steps": s["fungicide_steps"]},
        "organic":   {"name": s["neem2_name"], "description": s["neem2_desc"],     "steps": s["neem2_steps"]},
        "prevention":{"name": s["prev_name"],  "description": s["prev_desc"],      "steps": s["prev_steps"]},
        "youtube_links": [
            {"title": s["yt1"], "url": f"https://www.youtube.com/results?search_query={q}+treatment"},
            {"title": s["yt2"], "url": f"https://www.youtube.com/results?search_query={q}+plant+disease"},
        ],
    }
