# Clinic Visit Sorter

Console-based Python tool for the college health centre. It reads the short notes
students write when they arrive, asks an LLM to label each note, and orders the
nurse's queue so the most urgent students are seen first.

**The tool never diagnoses, never gives advice, and never removes anyone from the
queue. The nurse sees every student and makes every medical decision.**

CIT AI-Native Engineering Programme - Semester 1 - Assessment 5 (Capstone)
Made by: Tharika.L (2059)

## What it does
1. Reads `visits.csv` (columns: `id,time,note`). Notes can be English, Tamil or Tanglish.
2. Sends each note to the LLM (Google Gemini) with the system prompt in `system_prompt.txt`.
3. Checks every reply (valid JSON, allowed values). Retries once, then falls back to `today`.
4. Applies Python safety rules:
   - **Red flags** (chest pain, breathing trouble, fainting, heavy bleeding, lip/face
     swelling, "nenju vali", "moochu", ...) force priority to `see_now`.
   - **Priority only goes up.** Python never lowers a priority.
   - **No diagnosis.** A disease name or a guess word ("possible", "likely") in a summary
     is replaced with a neutral summary.
   - **Unknown is never routine.** If the API fails twice, the priority is `today`.
5. Writes two files:
   - `nurse_queue.csv` - every student, most urgent first, then by arrival time, with the
     original note. **For the nurse only.**
   - `summary.txt` - counts by priority, category and hour. No notes. For the in-charge.

## Setup
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## API key (never put it in the code)
Get a free key from Google AI Studio and set it in the terminal:
```
$env:GEMINI_API_KEY="your-key-here"        # PowerShell
$env:CLINIC_MODEL="gemini-3.5-flash-lite"  # a model your key can use
```
Run `python check_models.py` to find a model that works with your key.
Optional: `CLINIC_DELAY` (seconds between notes, default 5) and `CLINIC_RETRY_WAIT`
(seconds before the retry, default 6) help stay inside free-tier limits.

## Run
```
python clinic_sorter.py              # writes nurse_queue.csv and summary.txt
python eval_and_redteam.py           # eval harness (5 notes) + red-team (3 inputs)
```

## Files
| File | Purpose |
|---|---|
| `clinic_sorter.py` | Main program |
| `system_prompt.txt` | LLM prompt (Authority, Exemplar, Constraint, Rubric, Metadata) |
| `visits.csv` | Input: 20 sample notes |
| `nurse_queue.csv`, `summary.txt` | Output files |
| `eval_and_redteam.py` | Eval harness and red-team tests |
| `eval_results_run1.txt`, `eval_results_run2.txt` | Eval scores before and after the prompt fix |
| `check_models.py` | Helper that finds a working Gemini model |
| `requirements.txt`, `.gitignore` | Dependencies; keeps the API key out of Git |

## Testing
- Eval: 5 test notes with expected category and priority. Run 1: 3/5. After improving the
  prompt rubric, run 2: 5/5.
- Red-team: a note asking to be marked routine, a note asking for a diagnosis, and a Tamil
  red-flag note. Result: 3/3.
