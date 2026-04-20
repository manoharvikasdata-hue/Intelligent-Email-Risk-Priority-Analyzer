const API_URL = "http://127.0.0.1:8000/analyze";

function getOpenedEmailText() {
  // Gmail renders opened email bodies in multiple possible containers.
  const selectors = [
    "div.a3s.aiL",   // common Gmail message body container
    "div.a3s",       // fallback container
    "div.gs div[dir='ltr']"
  ];

  for (const selector of selectors) {
    const nodes = document.querySelectorAll(selector);
    if (!nodes || nodes.length === 0) continue;

    // Pick the largest non-empty text block as current email body.
    let bestText = "";
    nodes.forEach((node) => {
      const text = (node.innerText || "").trim();
      if (text.length > bestText.length) {
        bestText = text;
      }
    });

    if (bestText) return bestText;
  }

  return "";
}

async function analyzeCurrentEmail() {
  const emailText = getOpenedEmailText();

  if (!emailText) {
    alert("No opened email content found. Open an email in Gmail and try again.");
    return;
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: emailText })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Request failed");
    }

    const data = await response.json();
    const score = Number(data.risk_score).toFixed(2);
    const level = data.risk_level || "Unknown";
    alert(`Phishing Risk Score: ${score}/100\nRisk Level: ${level}`);
  } catch (error) {
    alert(`Could not analyze email.\n${error.message}`);
  }
}

function injectAnalyzeButton() {
  if (document.getElementById("phish-analyze-btn")) return;

  const button = document.createElement("button");
  button.id = "phish-analyze-btn";
  button.textContent = "Analyze Email";
  button.style.position = "fixed";
  button.style.bottom = "20px";
  button.style.right = "20px";
  button.style.zIndex = "99999";
  button.style.padding = "10px 14px";
  button.style.border = "none";
  button.style.borderRadius = "10px";
  button.style.background = "#0d6efd";
  button.style.color = "#fff";
  button.style.fontSize = "14px";
  button.style.cursor = "pointer";
  button.style.boxShadow = "0 4px 10px rgba(0,0,0,0.2)";
  button.title = "Analyze currently opened Gmail email";

  button.addEventListener("click", analyzeCurrentEmail);
  document.body.appendChild(button);
}

injectAnalyzeButton();
