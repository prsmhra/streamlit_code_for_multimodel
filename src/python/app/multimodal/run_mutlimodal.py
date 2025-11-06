
import asyncio
import logging
import json
import re
import os

from Config.config import MODEL_NAME
from src.python.app.multimodal.batching import create_multimodal_batches

from src.python.app.multimodal.validation_helpers import (
    validate_medical_relevance_prompt_only,
    validate_csv_has_data,
    validate_audio_content_alignment,
    safe_upload_file
)

from src.python.app.multimodal.gemini_client import (
    create_gemini_client, 
    call_gemini_multimodal,
    upload_file_with_client 
)
from src.python.app.multimodal_audio.orchestrator import run_audio_pipeline_for_batch
from src.python.app.multimodal_vision.run import run_vision_pipeline_for_batch

from src.python.app.multimodal.multimodal_instruction import MULTIMODAL_ANALYSIS_PROMPT,SUMMARY_AGENT_PROMPT

from src.python.app.constants.constants import Constants


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)



def format_batch_result_for_summary(batch_label: str, result_text: str) -> str:
    """
    Parses the raw JSON response from a batch and formats it into
    a clean, human-readable Markdown block for the final summary agent.
    """
    try:
        # Clean up markdown fences first
        match = re.search(r"```json\s*(\{.*?\})\s*```", result_text, re.DOTALL)
        if match:
            result_text = match.group(1)
        
        data = json.loads(result_text)
        
        # Start building the markdown string
        output = [f"### Batch Report: {batch_label}"]
        output.append(f"**Batch Summary:** {data.get(Constants.SUMMARY_KEY, 'No summary provided.')}")
        
        evidence = data.get(Constants.EVIDENCE_LIST_KEY, [])
        if evidence:
            output.append("\n**Key Evidence:**")
            for item in evidence:
                modality = item.get(Constants.MODALITY_KEY, 'unknown').upper()
                interp = item.get(Constants.INTERPRETATION_KEY, 'No interpretation.')
                
                # Get frame/time info
                frame_info = ""
                if item.get(Constants.FRAME_RANGE_KEY):
                    frame_info = f"({item.get(Constants.FRAME_RANGE_KEY)})"
                elif item.get(Constants.FRAME_INDEX_KEY):
                    frame_info = f"(frame {item.get(Constants.FRAME_INDEX_KEY)})"

                output.append(f"* **[{modality}]** {interp} {frame_info}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.warning(f"Failed to parse batch result for summary: {e}. Using raw text.")
        # Fallback to just sending the raw text
        return f"### Batch Report: {batch_label}\n{result_text}"
    

async def run_full_pipeline(
    user_prompt: str,
    master_csv_path: str,
    master_audio_path: str,
    batch_duration_seconds: int = 10
):
    """
    The main orchestrator function.
    """
    
    # --- 1. Setup ---
    logger.info("--- Starting Multimodal Pipeline ---")
    os.makedirs(Constants.AUDIO_FEATURES_DIR, exist_ok=True)
    os.makedirs(Constants.VISION_FEATURES_DIR, exist_ok=True)
    os.makedirs(Constants.MULTIMODAL_RESULTS_DIR, exist_ok=True)
 
    
    gemini_client = create_gemini_client()
    
    if not gemini_client:
        logger.error("Failed to initialize Gemini client. Aborting.")
        return
    


    # --- PRE-VALIDATE PROMPT --
    logger.info("--- Step 1: Validating User Prompt Medical Relevance ---")
    prompt_validation_result = validate_medical_relevance_prompt_only(
        client=gemini_client,
        user_prompt=user_prompt,
        model_name=MODEL_NAME
    )
    
    # --- Meta intent result ---
    meta_intent_log_message = (
        "\n" + "="*30 + " META-INTENT RESULT " + "="*30 +
        "\n" +
        json.dumps(prompt_validation_result, indent=2) +
        "\n" + "="*78 + "\n"
    )
    logger.info(meta_intent_log_message)
    
    if not prompt_validation_result.get(Constants.IS_MEDICAL_KEY):
        logger.error(f"Validation Failed: Prompt is not medical. Reason: {prompt_validation_result.get('reasoning')}")
        logger.error("Aborting pipeline.")
        return

    logger.info("[Completed] Prompt is medically relevant. Proceeding to batching.")
    disease_focus = prompt_validation_result.get("disease_focus") # Get disease focus


    # --- 2. Create Batches ---
    logger.info("--- Step 1: Creating Batches ---")
    batches = create_multimodal_batches(
        master_csv_path=master_csv_path,
        master_audio_path=master_audio_path,
        batch_duration_seconds=batch_duration_seconds,
    )
    
    if not batches:
        logger.error("No batches were created. Aborting.")
        return

    # --- 3. Process Batches in a Loop ---
    
    if Constants.DEFAULT_FPS <= 0:
        logger.error(f"Invalid INPUT_FPS in config: {Constants.DEFAULT_FPS}. Aborting.")
        return
        
    optimal_window_sec = Constants.CONSTANT_ONE / Constants.DEFAULT_FPS
    optimal_hop_ratio =Constants.CONSTANT_ONE-optimal_window_sec
    
    logger.info(f"--- Step 2: Processing {len(batches)} Batches ---")
    logger.info(f"Using dynamic audio window: {optimal_window_sec:.4f}s (1 per frame)")
    logger.info(f"Using dynamic audio hop ratio: {optimal_hop_ratio:.2f} (no overlap)")
    
    batch_results = []
    
    for batch_label, batch_csv_path, batch_audio_path in batches:
        logger.info(f"--- Processing Batch: {batch_label} ---")

        audio_output_dir = f"{Constants.AUDIO_FEATURES_DIR}{os.sep}{batch_label}"
        vision_output_dir = f"{Constants.VISION_FEATURES_DIR}{os.sep}{batch_label}"
        # --- STEP 3.5: PER-BATCH VALIDATION ---
        logger.info(f"Validating data for batch {batch_label}...")
        
        # 3.5a: Validate CSV
        csv_validation = validate_csv_has_data(batch_csv_path)
        if not csv_validation.get(Constants.IS_VALID_KEY):
            logger.warning(f"Batch {batch_label} SKIPPED: CSV validation failed. Error: {csv_validation.get(Constants.ERROR_KEY)}")
            continue # Skip to the next batch

        logger.info("[Completed] Batch CSV has data.")


        # 3.5b: Upload & Validate Audio
        try:
            uploaded_audio = safe_upload_file(gemini_client, batch_audio_path, Constants.BATCH_AUDIO)
            audio_validation = validate_audio_content_alignment(
                client=gemini_client,
                uploaded_audio=uploaded_audio,
                user_prompt=user_prompt,
                disease_focus=disease_focus,
                model_name=MODEL_NAME
            )
            if not audio_validation.get(Constants.IS_ALIGNED_KEY):
                logger.warning(f"Batch {batch_label} SKIPPED: Audio content validation failed. Reason: {audio_validation.get(Constants.REASONING_KEY)}")
                continue # Skip to the next batch
            
            logger.info("[Completed] Batch Audio is aligned.")

        except Exception as e:
            logger.warning(f"Batch {batch_label} SKIPPED: Audio upload/validation failed. Error: {e}")
            continue # Skip to the next batch

        logger.info(f"Starting audio and vision pipelines for batch {batch_label} in parallel...")
        
       
        vision_task= run_vision_pipeline_for_batch(
            disease_focus=disease_focus,
            input_csv_path=batch_csv_path,
            output_dir=vision_output_dir,
            batch_label=batch_label
        )
        
        audio_task= asyncio.to_thread(run_audio_pipeline_for_batch,
            audio_path=batch_audio_path,
            windows_to_try=[optimal_window_sec],
            out_dir=audio_output_dir,
            batch_label=batch_label,
            client=gemini_client,
            hop_ratio=optimal_hop_ratio # Pass our new hop ratio
        
        )

        # Wait for both tasks to complete
        results = await asyncio.gather(vision_task, audio_task)
        
        (vision_csv_output_path, disease_focus) = results[0]
        (audio_contents, audio_meta) = results[1]
        
        if not vision_csv_output_path:
            logger.warning(f"Vision pipeline failed for batch {batch_label}. Skipping.")
            continue


        # --- 3c. Combine Payloads and Call Gemini ---
        logger.info("Combining payloads for multimodal call...")
        
        final_prompt_parts = [ MULTIMODAL_ANALYSIS_PROMPT ]
        
        # 2. Add batch-specific context 
        final_prompt_parts.append(
            f"CONTEXT FOR THIS BATCH:\n"
            f"- Batch Label: {batch_label}\n"
            f"- Disease Focus: {disease_focus or 'Not specified'}"
        )
        
        final_prompt_parts.extend(audio_contents)
        
        try:
            logger.info(f"Uploading final vision CSV: {vision_csv_output_path}")
            vision_file_upload = upload_file_with_client(
                client=gemini_client,
                file_path=vision_csv_output_path,
                display_name=f"{batch_label}_{Constants.VISION_FEATURES_CSV}"
            )
            if vision_file_upload:
                final_prompt_parts.append(f"Final Vision Feature CSV for {batch_label}:")
                final_prompt_parts.append(vision_file_upload)
            else:
                # If upload returns None, raise an error to be caught below
                raise Exception(f"File upload returned None for {vision_csv_output_path}")

        except Exception as e:
            logger.error(f"Failed to upload vision CSV {vision_csv_output_path}: {e}")
            
            try:
                with open(vision_csv_output_path, 'r') as f:
                    csv_preview = f.read(Constants.VISION_DEFAULT_CSV_READ_NUM)
                final_prompt_parts.append(f"VISION CSV PREVIEW: {csv_preview}...")
            except Exception as e_preview:
                logger.error(f"Failed to even read CSV for preview: {e_preview}")
                pass # Skip if we can't even read it
    

        batch_result_text = await call_gemini_multimodal(
            client=gemini_client,
            model_name=MODEL_NAME,
            prompt_parts=final_prompt_parts
        )
        
        if batch_result_text:

            # ---PRINT THE RAW BATCH RESPONSE ---
            logger.info(f"---Raw Gemini Response for {batch_label} ---")
            logger.info(batch_result_text)
            logger.info("-------------------------------------------------")


            logger.info(f"Batch {batch_label} analysis complete.")
            
            formatted_summary = format_batch_result_for_summary(batch_label, batch_result_text)
            batch_results.append(formatted_summary)
            
            # batch_result_file = out_dir / Constants.BATCH_RESULTS_KEY / f"{batch_label}_{Constants.RESULT_JSON_SUFFIX}"
            # batch_result_file.parent.mkdir(parents=True, exist_ok=True)
            batch_result_file = Constants.MULTIMODAL_RESULTS_DIR / f"{batch_label}_result.json"
            batch_result_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                match = re.search(r"```json\s*(\{.*?\})\s*```", batch_result_text, re.DOTALL)
                if match:
                    batch_result_text = match.group(1)
                
                with open(batch_result_file, Constants.WRITE_MODE, encoding=Constants.UTF_8) as f:
                    json.dump(json.loads(batch_result_text), f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to parse/save batch JSON, saving raw text: {e}")
                batch_result_file.with_suffix(Constants.TXT_EXTENSION).write_text(batch_result_text, encoding=Constants.UTF_8)
        else:
            logger.warning(f"Batch {batch_label} returned no result from Gemini.")


    # --- 4. Run Final Summary Agent ---
    logger.info(f"--- Step 4: Summarizing {len(batch_results)} Batch Results ---")
    
    if not batch_results:
        logger.error("No batch results to summarize. Exiting.")
        return



    summary_payload = [
        SUMMARY_AGENT_PROMPT,
        "--- BATCH REPORTS ---",
        "\n\n".join(batch_results)
    ]
    

    final_summary_text = await call_gemini_multimodal(
        client=gemini_client,
        model_name=MODEL_NAME,
        prompt_parts=summary_payload
    )


    if final_summary_text:
        logger.info("--- Pipeline Complete: Final Summary ---")

        summary_log_message = (
            "\n" + "="*30 +
            "\n--- FINAL SUMMARY ---\n" +
            final_summary_text +
            "\n" + "="*30 + "\n"
        )
        logger.info(summary_log_message)
        
       
        
        # 1. Define the new report file path (using .txt)
        report_file = Constants.BASE_OUTPUT_DIR / Constants.FINAL_REPORT_TXT
        
        try:
            # 2. Just write the raw text directly to the file
            with open(report_file, Constants.WRITE_MODE, encoding=Constants.UTF_8) as f:
                f.write(final_summary_text)
            logger.info(f"Final summary report saved to {report_file}")
            
        except Exception as e:
            # This is just a fallback in case writing fails
            logger.error(f"Failed to save final summary report: {e}")
            
            
    else:
        logger.error("Failed to generate final summary.")
    
