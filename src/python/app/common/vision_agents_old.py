import os
import io
import re
import gc
import ast
import json
import random
import logging
import asyncio
import aiohttp

import pandas as pd
from datetime import datetime
from google.genai import types
from dotenv import load_dotenv
from pydantic import PrivateAttr
import plotly.graph_objects as go


from collections import defaultdict
from google.adk.events import Event
from google.adk.runners import Runner
from typing_extensions import override
from google.api_core import exceptions 
from google.genai.errors import ClientError 
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.sessions import InMemorySessionService

from src.python.app.utils.state_summary import build_state_summary
from src.python.app.tools.frame_prefilter_tools import prefilter_frames_function_tool
from src.python.app.tools.csv_filter_tools import csv_filter_tool
from src.python.app.tools.sample_function_tools import sample_csv_function_tool
from src.python.app.instructions.agent_instructions import (META_INTENT_INSTRUCTION, CSV_SAMPLER_INSTRUCTION,
                                                            PREFILTER_INSTRUCTION, REGION_DETECTOR_INSTRUCTION,
                                                            SYMPTOM_ANALYZER_INSTRUCTION, ORCHESTRATOR_INSTRUCTION)

from src.python.app.constants.constants import Constants

from Config import config

INPUT_FPS=config.INPUT_FPS
MODEL_NAME =config.MODEL_NAME

class MetaIntentAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()

    def __init__(self, llm_agent: LlmAgent, name=Constants.METAINTENTTOOL_KEY):
        super().__init__(name=name, sub_agents=[llm_agent])
        self._llm_agent = llm_agent

   
    # @retry_with_exponential_backoff(max_retries=5)
    @override
    async def _run_async_impl(self, ctx):
        # Extract latest user message (handle various sources safely)
        user_msg = Constants.INVERTED_STRING
        if hasattr(ctx, Constants.USER_CONTENT) and ctx.user_content and ctx.user_content.parts:
            user_msg = ctx.user_content.parts[Constants.ZERO].text.strip()
        
        if not user_msg:
            yield Event(
                content=types.Content(
                    role=Constants.ASSISTENT_KEY,
                    parts=[types.Part(text=Constants.METAINTENT_NO_ANALYSIS_STR)]
                ),
                author=self.name,
            )
            return

        # Fill instruction
        instruction_filled = META_INTENT_INSTRUCTION

        # Temporary LLM agent for classification
        temp_llm_agent = LlmAgent(
            name=Constants.METAINTENTLLM_KEY,
            model=MODEL_NAME,
            instruction=instruction_filled,
            output_key=Constants.METAINTENT_RES_KEY,
            tools=[]
        )

        
        # Run classification
        async for ev in temp_llm_agent.run_async(ctx):
            
            if not ev.author:
                ev.author = temp_llm_agent.name
            yield ev

        # Retrieve and process result
        raw = ctx.session.state.get(Constants.METAINTENT_RES_KEY, Constants.INVERTED_STRING)
        
        if not raw:
            yield Event(
                content=types.Content(
                    role=Constants.ASSISTENT_KEY,
                    parts=[types.Part(text=Constants.METAINTENT_NO_RESULT_STR)]
                ),
                author=self.name,
            )
            return

        # Try to parse JSON result

        match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw_json = match.group(Constants.ONE)
        else:
            raw_json = raw

        try:
            result = json.loads(raw_json)
        
        except Exception:
            result = {"intent_type": "invalid_input", "disease_focus": Constants.INVERTED_STRING, "reason": "Could not parse LLM output."}

        # Save disease focus (if found) to session state for downstream agents
        if result.get("disease_focus"):
            ctx.session.state["disease_focus"] = result["disease_focus"]

        msg = f"✅  MetaIntent analysis completed\n\n{result}"
        # Log the structured result
        yield Event(
            content=types.Content(
                role=Constants.ASSISTENT_KEY,
                parts=[types.Part(text=msg)]
            ),
            author=self.name,
        )



class CSVSamplerAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()

    def __init__(self, llm_agent: LlmAgent, name="FrameSamplerTool"):
        super().__init__(name=name, sub_agents=[llm_agent])
        self._llm_agent = llm_agent

    @override
    async def _run_async_impl(self, ctx):
        csv_path = ctx.session.state.get("blendshape_csv_path")
        out_dir = ctx.session.state.get("work_dir", ".")
        input_fps = INPUT_FPS

        if not csv_path or not os.path.exists(csv_path):
            msg = "❌ Input CSV not found for sampling."
            yield Event(
                content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
                author=self.name
            )
            return

        # Store context for tool
        ctx.session.state.update({
            "csv_path": csv_path,
            "out_dir": out_dir,
            "original_fps": input_fps
        })




        df = pd.read_csv(csv_path)
        markdown_table = df.to_markdown(index=False)  # preview small subset

        # Fill LLM instruction
        instruction_filled = CSV_SAMPLER_INSTRUCTION.format(
            original_fps=input_fps,
            markdown_table=markdown_table
        )

        # Create a temporary LLM agent with the tool
        temp_llm_agent = LlmAgent(
            name=self._llm_agent.name,
            model=MODEL_NAME,
            instruction=instruction_filled,
            tools=[sample_csv_function_tool],
        )

        # Run LLM (it will call sample_csv_function_tool)
        async for ev in temp_llm_agent.run_async(ctx):
            if not ev.author:
                ev.author = temp_llm_agent.name
            yield ev

        # Retrieve result
        result = ctx.session.state.get("csv_sampler_result")
        if not result:
            msg = "❌ CSV sampler tool was not called by the LLM."
            yield Event(
                content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
                author=self.name
            )
            return

        if not result.get("success"):
            msg = f"🚫 CSV sampling failed. Reason: {result.get('reason', 'Unknown')}"
            yield Event(
                content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
                author=self.name
            )
            return

        ctx.session.state["sampled_blendshape_csv_path"] = result["sampled_csv_path"]
        ctx.session.state["target_fps"] = result["target_fps"]

        msg = f"✅ CSV sampled to ~{result['target_fps']} FPS. Saved to {result['sampled_csv_path']}. Reason: {result.get('reason','')}"
        yield Event(
            content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
            author=self.name
        )


class FramePrefilterAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()
    
    def __init__(self, llm_agent: LlmAgent, name="FramePrefilterTool"):
        super().__init__(name=name, sub_agents=[llm_agent])
        self._llm_agent = llm_agent
    
    @override
    async def _run_async_impl(self, ctx):
        csv_path = ctx.session.state.get("sampled_blendshape_csv_path")

        if csv_path is None:
            csv_path = ctx.session.state.get("blendshape_csv_path")

    
        
        # Save context variables for the tool
        out_dir = ctx.session.state.get("work_dir", ".")
        
        ctx.session.state.update({
            "csv_path": csv_path,
            "out_dir": out_dir
        })
        
        df = pd.read_csv(csv_path)
        markdown_table = df.to_markdown(index=False)
        ctx.session.state["prefilter_preview_markdown"] = markdown_table
        
        # Generate prefilter instruction dynamically
        instruction_filled = PREFILTER_INSTRUCTION.format(original_fps=INPUT_FPS,markdown_table=markdown_table)

        
        # Create LLM agent with the tool
        temp_llm_agent = LlmAgent(
            name=self._llm_agent.name,
            model=MODEL_NAME,
            instruction=instruction_filled,
            tools=[prefilter_frames_function_tool],
        )
        
        # Run LLM asynchronously
        async for ev in temp_llm_agent.run_async(ctx):
            if not ev.author:
                ev.author = temp_llm_agent.name
            yield ev
        
        # Retrieve tool result from where the TOOL stored it
        result = ctx.session.state.get("prefilter_result")
        
        if result is None:
            msg = "❌ Prefilter tool was not called. No result found."
            yield Event(
                content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
                author=self.name
            )
            return
        
        if not isinstance(result, dict):
            msg = f"❌ Tool returned unexpected type: {type(result)}"
            yield Event(
                content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
                author=self.name
            )
            return
        
        if not result.get("useful"):
            msg = f"🚫 Prefilter flagged data as not useful. Reason: {result.get('reason', 'Unknown')}"
            yield Event(
                content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
                author=self.name
            )
            ctx.session.state["prefiltered_csv_path"] = None
            return
        
        # Success - store results
        ctx.session.state["prefiltered_csv_path"] = result["filtered_csv_path"]
        ctx.session.state["prefilter_summary"] = f"Kept ranges: {result['kept_ranges']}"
        
        msg = f"✅ Prefilter kept frames {result['kept_ranges']}, saved to {result['filtered_csv_path']}"
        yield Event(
            content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
            author=self.name
        )


class CSVFilterAgent(BaseAgent):
    def __init__(self, region_llm_agent: LlmAgent, name="FeaturesSelectionTool"):
        super().__init__(name=name, sub_agents=[region_llm_agent])
        self._region_llm_agent = region_llm_agent

        self._region_map = {
            "eyes": [
                "eyeBlinkLeft","eyeBlinkRight","eyeSquintLeft","eyeSquintRight",
                "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
                "eyeWideLeft","eyeWideRight"
            ],
            "mouth": [
                "mouthSmileLeft","mouthSmileRight","mouthPucker","jawOpen","mouthFunnel",
                "mouthLowerDownLeft","mouthLowerDownRight","mouthPressLeft","mouthPressRight",
                "mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
                "mouthStretchLeft","mouthStretchRight","mouthUpperUpLeft","mouthUpperUpRight"
            ],
            "nose": [
                "cheekSquintLeft","cheekSquintRight","noseSneerLeft","noseSneerRight",
                "browDownLeft","browDownRight","mouthUpperUpLeft","mouthUpperUpRight",
                "eyeSquintLeft","eyeSquintRight"
            ],
        }

        # Predefined region-to-AU mappings (example — adjust as per your AU naming convention)
        self._au_region_map = {
            "eyes": ["AU01- Inner Brow Raiser", "AU02- Outer Brow Raiser", "AU04- Brow Lower", "AU05- Upper Lid Raiser", 
                     "AU07- Lid Tightener","AU41- Eye Droop","AU42- Eye Slit","AU43- Eyes Close","AU44- Squint",
                     "AU45- Blink","AU46- Wink","AU61- Eye Turn Left","AU62- Eye Turn Right","AU63- Eye Turn Up","AU64- Eye Turn Down"],  

            "mouth": ["AU16- Lower Lip Depressor", "AU17- Chin Raiser", "AU12- Lip Corner Puller", "AU13- Cheek Puffer", "AU14- Dimpler",
                      "AU15- Lip Corner Depressor","AU10- Upper Lip Raiser","AU18- Lip Puckerer","AU20- Lip Stretcher","AU22- Lip Funneler",
                      "AU23- Lip Tightener","AU24- Lip Pressor","AU25- Lips Part","AU26- Jaw Drop","AU27- Mouth Stretch","AU28- Lip Suck"],  

            "nose": ["AU06- Cheek Raiser", "AU11- Nasolabial Deepener","AU09- Nose Wrinkler"],  
        }

        # Fixed emotion columns
        self._emotion_cols = ["Joy", "Sadness", "Surprise", "Fear", "Anger", "Disgust", "Contempt"]



    async def _run_async_impl(self, ctx):
        
        csv_path = ctx.session.state.get("prefiltered_csv_path")

        if csv_path is None:
            csv_path = ctx.session.state.get("blendshape_csv_path")


        if not csv_path or not os.path.exists(csv_path):
            msg = "❌ Prefiltered CSV not found."
            yield Event(content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]), author=self.name)
            return

        # Save context variables
        out_dir = ctx.session.state.get("work_dir", ".")
        ctx.session.state.update({
            "csv_path": csv_path,
            "region_map": self._region_map,
            "au_region_map": self._au_region_map,
            "emotion_cols": self._emotion_cols,
            "out_dir": out_dir
        })

        # Convert CSV to markdown for LLM analysis
        df = pd.read_csv(csv_path)
        markdown_table = df.to_markdown(index=False)

        
        instruction_filled = REGION_DETECTOR_INSTRUCTION.format(original_fps=INPUT_FPS,markdown_table=markdown_table)

        
    

        # Create LLM agent with the tool
        temp_llm_agent = LlmAgent(
            name="RegionDetectorLLM",
            model=MODEL_NAME,
            instruction=instruction_filled,
            tools=[csv_filter_tool],  # provide the tool
            
        )

        # Run LLM asynchronously
        async for ev in temp_llm_agent.run_async(ctx):
            if not ev.author:
                ev.author = temp_llm_agent.name
            yield ev



        # Retrieve tool result
        result = ctx.session.state.get("filtered_csv_result")
        
    
        if result is None:
            msg = "❌ Tool was not called. No result found."
            yield Event(content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]), author=self.name)
            return
        
        if isinstance(result, str):
            msg = f"❌ Tool returned string instead of dict: {result}"
            yield Event(content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]), author=self.name)
            return
        
        if not result.get("success"):
            msg = f"🚫 CSV filtering failed: {result.get('reason', 'unknown error')}"
            yield Event(content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]), author=self.name)
            return

        ctx.session.state["filtered_blendshape_csv_path"] = result["filtered_csv_path"]
        msg = f"✅ Filtered CSV saved: {result['filtered_csv_path']}\nSelected regions: {result.get('regions')}"
        yield Event(content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]), author=self.name)




    
class SymptomAnalyzerAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()

    def __init__(self, llm_agent: LlmAgent, name="SymptomAnalyzerTool"):
        super().__init__(name=name, sub_agents=[llm_agent])
        self._llm_agent = llm_agent

    @override
    async def _run_async_impl(self, ctx):
        csv_path = ctx.session.state.get("filtered_blendshape_csv_path")
        if not csv_path or not os.path.exists(csv_path):
            msg = "❌ Filtered CSV not found."
            yield Event(content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
                        author=self.name)
            return

        df = pd.read_csv(csv_path)
        # df_sampled = df.iloc[::10, :]
        markdown_table = df.to_markdown(index=False)
        ctx.session.state["symptom_analysis_markdown"] = markdown_table

        disease_focus = ctx.session.state.get("disease_focus", Constants.INVERTED_STRING)
        if disease_focus:
            instruction_extra = f"\n\n⚕️ Focus particularly on findings relevant to **{disease_focus}**."
        else:
            instruction_extra = Constants.INVERTED_STRING


        instruction_filled = SYMPTOM_ANALYZER_INSTRUCTION.format(original_fps=INPUT_FPS,markdown_table=markdown_table)+ instruction_extra
        temp_llm_agent = LlmAgent(
            name="SymptomReasonerLLM",
            model=MODEL_NAME,
            instruction=instruction_filled,
            input_schema=None,
            output_key="symptom_analysis"
        )

        async for ev in temp_llm_agent.run_async(ctx):
            # pass
            if not ev.author:
                ev.author = temp_llm_agent.name
            yield ev

        # 1. Retrieve the raw output from the session state
        raw_output = ctx.session.state.get("symptom_analysis", Constants.INVERTED_STRING)
        
        # 2. OPTIONAL: Save the raw output to a backup key for separate access if needed
        #    This is good practice, but for the UI fix, we embed it in the message.
        #    ctx.session.state["final_symptom_analysis_raw"] = raw_output 

        # 3. Create the final completion message by appending the raw output.
        #    This ensures the JSON is visible in the Streamlit message variable.
        msg = f"✅ Symptom analysis process completed.\n\n{raw_output}"
        
        yield Event(content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]),
                    author=self.name)



class LlmOrchestratorAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()
    _tools: dict = PrivateAttr()

    def __init__(self, llm_agent: LlmAgent, tools: list[BaseAgent], name="LlmOrchestrator"):
        super().__init__(name=name, sub_agents=[llm_agent] + tools)
        self._llm_agent = llm_agent
        self._tools = {t.name: t for t in tools}

    @override
    async def _run_async_impl(self, ctx):
        yield Event(
            content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text="🧭 Orchestrator is starting...")]),
            author=self.name
        )
        history = []

        # 🔹 STEP 1: Run MetaIntent classification first
        meta_intent_tool = self._tools.get("MetaIntentTool")
        if meta_intent_tool:
            async for ev in meta_intent_tool.run_async(ctx):
                yield ev

        intent_raw = ctx.session.state.get(Constants.METAINTENT_RES_KEY, Constants.INVERTED_STRING)
        try:
            match = re.search(r"```json\s*(\{.*\})\s*```", intent_raw, re.DOTALL)
            intent = json.loads(match.group(1)) if match else {}
        except Exception:
            intent = {"intent_type": "invalid_input"}

        intent_type = intent.get("intent_type", "invalid_input")
        disease_name = intent.get("disease_name")
        reason = intent.get("reason", Constants.INVERTED_STRING)

        # 🔹 STEP 2: Handle invalid or non-medical inputs
        if intent_type in ["non_medical", "invalid_input"]:
            msg = f"🚫 Input not suitable for medical analysis. Reason: {reason}"
            yield Event(content=types.Content(role=Constants.ASSISTENT_KEY, parts=[types.Part(text=msg)]), author=self.name)
            return

        # 🔹 STEP 3: Store context if disease-specific
        if intent_type == "disease_specific" and disease_name:
            ctx.session.state["disease_focus"] = disease_name

        # 🔹 STEP 4: Continue normal orchestration loop
        while True:
            ctx.session.state["orchestrator_memory_summary"] = build_state_summary(ctx.session.state)
            state_summary = ctx.session.state["orchestrator_memory_summary"]

            self._llm_agent.instruction = ORCHESTRATOR_INSTRUCTION.format(summary=state_summary)
            async for ev in self._llm_agent.run_async(ctx):
                if not ev.author:
                    ev.author = self._llm_agent.name
                yield ev

            decision_raw = ctx.session.state.get("orchestrator_decision", Constants.INVERTED_STRING)
            try:
                match = re.search(r"```json\s*(\{.*\})\s*```", decision_raw, re.DOTALL)
                decision = json.loads(match.group(1)) if match else {}
            except Exception:
                decision = {"next_tool": "STOP"}

            next_tool = decision.get("next_tool", "STOP")
            reason = decision.get("reason", Constants.INVERTED_STRING)

            yield Event(
                content=types.Content(
                    role=Constants.ASSISTENT_KEY,
                    parts=[types.Part(text=f"🧭 Orchestrator chose: {next_tool} ({reason})")]
                ),
                author=self.name
            )

            if next_tool == "STOP":
                break
            if next_tool not in self._tools:
                yield Event(
                    content=types.Content(
                        role=Constants.ASSISTENT_KEY,
                        parts=[types.Part(text=f"❌ Unknown tool {next_tool}, stopping.")]
                    ),
                    author=self.name
                )
                break

            tool = self._tools[next_tool]

            
            async for ev in tool.run_async(ctx):
                if not ev.author:
                    ev.author = tool.name
                yield ev

            if next_tool == "FramePrefilterTool":
                prefiltered_csv = ctx.session.state.get("prefiltered_csv_path")
                if prefiltered_csv is None:
                    yield Event(
                        content=types.Content(
                            role=Constants.ASSISTENT_KEY,
                            parts=[types.Part(
                                text="🚫 Prefilter detected no useful frames. Stopping pipeline."
                            )]
                        ),
                        author=self.name
                    )
                    break  


            ctx.session.state["orchestrator_memory_summary"] = build_state_summary(ctx.session.state)
            history.append(f"Ran {next_tool}")

        yield Event(
            content=types.Content(
                role=Constants.ASSISTENT_KEY,
                parts=[types.Part(text="✅ Orchestrator finished flow.")]
            ),
            author=self.name
        )
