import re
import json

def create_batches(df, batch_size=100):
    """Split dataframe into smaller batches."""
    return [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

def format_batches_for_summary(all_batches: list[dict]) -> str:
    """Formats results from all batches into a markdown string for the summary agent."""
    output_lines = ["### Batch Analysis Summaries\n"]
    for i, b in enumerate(all_batches):
        batch_id = i + 1
        # Extract symptom analysis from the nested session state
        symptom_data = b.get("session_state_data", {}).get("symptom_analysis", "")
        output_lines.append(f"#### 🧩 Batch {batch_id}")

        if isinstance(symptom_data, str):
            # Clean and parse JSON from the raw string
            match = re.search(r"```json\s*(\[.*\])\s*```", symptom_data, re.DOTALL)
            if match:
                clean_data = match.group(1)
            else: # Fallback for raw JSON string
                clean_data = symptom_data.strip().lstrip("`json").rstrip("`")
            try:
                parsed_data = json.loads(clean_data)
            except json.JSONDecodeError:
                output_lines.append("_Unable to parse symptom analysis JSON._\n")
                continue
        else:
            parsed_data = symptom_data # Assumes it's already a list/dict

        if not parsed_data:
            output_lines.append("_No detected symptoms in this batch._\n")
            continue

        for s in parsed_data:
            region = s.get("affected_region", "Unknown region")
            symptom = s.get("symptom", "Unspecified symptom")
            confidence = s.get("confidence", "N/A")
            output_lines.append(f"- **{region.capitalize()}**: {symptom} (confidence: {confidence})")
        output_lines.append("") # blank line between batches
    return "\n".join(output_lines)

