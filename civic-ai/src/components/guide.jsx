import React, { useState } from "react";
import '../App.css'

export default function Guide() {
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [draft, setDraft] = useState(null);
  const [step, setStep] = useState("upload");


  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setPrediction(data);
    setStep("review");
  };

  // request auto-filled draft
  const generateDraft = async () => {
    const res = await fetch("http://localhost:8000/assist-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        summary: prediction.category,   // e.g., "pothole"
        address: null,
        description: null,
        tags: [],
        severity: "medium",
        reporter_name: null,
      }),
    });

    const data = await res.json();
    setDraft(data);
    setStep("fill");
  };

  // user fills remaining fields
  const handleSubmit = () => {
    alert("Here you would send to SeeClickFix API or save file.");
  };

// formatting of user process
  return (
    <div style={{ maxWidth: "500px", margin: "auto" }}>
      {step === "upload" && (
        <>
        <div className="uploadCont">
          <text>Upload Image of Issue</text>
          <input type="file" accept="image/*" onChange={handleUpload} />
        </div>
        </>
      )}

      {/*show imgpreview*/}
      {preview && (
        <img src={preview} alt="preview" style={{ width: "100%", marginTop: "15px" }} />
      )}

      {/* show prediction */}
      {step === "review" && prediction && (
        <div style={{ marginTop: "20px" }}>
          <h3>Detected Issue:</h3>
          <p><strong>{prediction.category}</strong></p>

          <button onClick={generateDraft}>Create a Report</button>
          <button onClick={() => setStep("upload")}>Start Over</button>
        </div>
      )}

      {/*show auto-filled draft */}
      {step === "fill" && draft && (
        <div style={{ marginTop: "20px" }}>
          <h3>Auto-Filled Report</h3>

          <label>Issue Type</label>
          <input value={draft.summary} readOnly style={{ width: "100%" }} />

          <label>Description</label>
          <textarea
            defaultValue={draft.description}
            style={{ width: "100%", height: "80px" }}
          />

          <label>Address</label>
          <input
            placeholder="Enter location"
            style={{ width: "100%", marginBottom: "10px" }}
          />

          <button onClick={handleSubmit}>Submit to SeeClickFix</button>
        </div>
      )}
    </div>
  );
}