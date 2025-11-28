from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import assist_report
import generate_report
from img_classifier import classify_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# img classification
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = classify_image(image_bytes)
    return result


class AssistInput(BaseModel):
    summary: str
    address: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    severity: str | None = "medium"
    reporter_name: str | None = None

#draft report
@app.post("/assist-report")
async def assist_report_endpoint(data: AssistInput):
    issues = assist_report.load_issues("city_issues.json")
    issues_sample = issues[:5]

    input_fields = {
        "summary": data.summary,
        "address": data.address,
        "description": data.description,
        "tags": data.tags or [],
        "severity": data.severity,
        "reporter_name": data.reporter_name,
    }

    drafted = assist_report.draft_report(input_fields, issues_sample)
    return drafted


class DraftInput(BaseModel):
    summary: str
    lat: float | None = None
    lng: float | None = None

# send report to seeclickfix
@app.post("/generate-draft")
async def generate_draft(data: DraftInput):
    issues = generate_report.load_issues("city_issues.json")

    # Build a Gemini prompt
    prompt = generate_report.build_prompt_for_gemini(
        data.summary,
        data.lat or 33.7701,
        data.lng or -118.1937,
        samples=issues[:5]
    )

    text = generate_report.call_gemini(prompt)

    if not text:
        return {
            "summary": data.summary,
            "description": f"Draft report for '{data.summary}'.",
            "lat": data.lat,
            "lng": data.lng,
            "tags": [data.summary],
            "severity": "medium",
            "reporter_name": "system",
        }

    try:
        import json
        return json.loads(text)
    except:
        return {"raw_output": text}
