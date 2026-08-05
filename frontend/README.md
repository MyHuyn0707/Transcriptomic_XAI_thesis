

## Run Locally

**Frontend:**

0. `cd LV_code/frontend`
1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`


**Backend:**
0. `cd LV_code`
1. Install dependencies:
   `uv sync`
2. Run the app:
   `uv run uvicorn src.api.main:app --reload`