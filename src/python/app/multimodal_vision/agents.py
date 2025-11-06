import os
import re
import json
import logging

import pandas as pd
from google.genai import types
from pydantic import PrivateAttr

from google.adk.events import Event
from typing_extensions import override
from google.adk.agents import BaseAgent, LlmAgent

from src.python.app.multimodal_vision.state_summary import build_state_summary
from src.python.app.multimodal_vision.tools.frame_prefilter_tools import prefilter_frames_function_tool
from src.python.app.multimodal_vision.tools.csv_filter_tools import csv_filter_tool
from src.python.app.multimodal_vision.tools.sample_function_tools import sample_csv_function_tool
from src.python.app.multimodal_vision.instructions import META_INTENT_INSTRUCTION,CSV_SAMPLER_INSTRUCTION,PREFILTER_INSTRUCTION,REGION_DETECTOR_INSTRUCTION,ORCHESTRATOR_INSTRUCTION

from Config.config import MODEL_NAME
from src.python.app.constants.constants import Constants

logger = logging.getLogger(__name__)



class MetaIntentAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()

    def __init__(self, llm_agent: LlmAgent, name=Constants.METAINTENTTOOL_KEY):
        super().__init__(name=name, sub_agents=[llm_agent])
        self._llm_agent = llm_agent

    @override
    async def _run_async_impl(self, ctx):
        user_msg = Constants.EMPTY_STRING
        if hasattr(ctx, Constants.USER_CONTENT) and ctx.user_content and ctx.user_content.parts:
            user_msg = ctx.user_content.parts[0].text.strip()
        
        if not user_msg:
            yield Event(
                content=types.Content(
                    role=Constants.ASSISTANT_ROLE,
                    parts=[types.Part(text="[Warning] No user message found for MetaIntent analysis.")]
                ),
                author=self.name,
            )
            return

        instruction_filled = META_INTENT_INSTRUCTION

        temp_llm_agent = LlmAgent(
            name=Constants.METAINTENTLLM_KEY,
            model=MODEL_NAME, 
            instruction=instruction_filled,
            output_key=Constants.METAINTENT_RES_KEY,
            tools=[]
        )
        
        async for ev in temp_llm_agent.run_async(ctx):
            if not ev.author:
                ev.author = temp_llm_agent.name
            yield ev

        raw = ctx.session.state.get(Constants.METAINTENT_RES_KEY, Constants.EMPTY_STRING)
        
        if not raw:
            yield Event(
                content=types.Content(
                    role=Constants.ASSISTANT_ROLE,
                    parts=[types.Part(text="[Warning] No meta-intent result returned by LLM.")]
                ),
                author=self.name,
            )
            return

        match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw_json = match.group(1)
        else:
            raw_json = raw

        try:
            result = json.loads(raw_json)
        except Exception:
            
            result = {Constants.INTENT_TYPE: "invalid_input", Constants.DISEASE_FOCUS_KEY: "", Constants.REASON_KEY: "Could not parse LLM output."}

        # Save result for the runner to inspect
        ctx.session.state[Constants.METAINTENT_ANALYSIS_KEY] = result

        if result.get(Constants.DISEASE_FOCUS_KEY):
            ctx.session.state[Constants.DISEASE_FOCUS_KEY] = result[Constants.DISEASE_FOCUS_KEY]

        yield Event(
            content=types.Content(
                role=Constants.ASSISTANT_ROLE,
                parts=[types.Part(text=f"MetaIntent analysis completed: {result.get(Constants.INTENT_TYPE)}")
            ]),
            author=self.name,
        )



class CSVSamplerAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()

    def __init__(self, llm_agent: LlmAgent, name=Constants.FRAMESAMPLER_TOOL):
        super().__init__(name=name, sub_agents=[llm_agent])
        self._llm_agent = llm_agent

    @override
    async def _run_async_impl(self, ctx):
        csv_path = ctx.session.state.get(Constants.BLENDSHAPE_CSV_PATH_KEY)
        out_dir = ctx.session.state.get(Constants.WORK_DIR_KEY, Constants.CURRENT_DIR)
        input_fps = Constants.DEFAULT_FPS

        if not csv_path or not os.path.exists(csv_path):
            msg = "[Failed] Input CSV not found for sampling."
            yield Event(
                content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]),
                author=self.name
            )
            return

        # Store context for tool
        ctx.session.state.update({
            Constants.CSV_PATH_KEY: csv_path,
            Constants.OUT_DIR_KEY: out_dir,
            Constants.ORIGINAL_FPS_KEY: input_fps
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
        result = ctx.session.state.get(Constants.CSV_SAMPLER_RESULT_KEY)
        if not result:
            msg = "[Failed] CSV sampler tool was not called by the LLM."
            yield Event(
                content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]),
                author=self.name
            )
            return

        if not result.get(Constants.SUCCESS_KEY):
            msg = f"[Failed] CSV sampling failed. Reason: {result.get(Constants.REASON_KEY, 'Unknown')}"
            yield Event(
                content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]),
                author=self.name
            )
            return

        ctx.session.state[Constants.SAMPELED_BLENDSAHPE_CSV_PATH_KEY] = result[Constants.SAMPELED_CSV_PATH_KEY]
        ctx.session.state[Constants.TARGET_FPS_KEY] = result[Constants.TARGET_FPS_KEY]

        msg = f"[Completed] CSV sampled to ~{result[Constants.TARGET_FPS_KEY]} FPS. Saved to {result[Constants.SAMPELED_CSV_PATH_KEY]}. Reason: {result.get(Constants.REASON_KEY,Constants.EMPTY_STRING)}"
        yield Event(
            content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]),
            author=self.name
        )


class FramePrefilterAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()
    
    def __init__(self, llm_agent: LlmAgent, name=Constants.FRAME_PREFILTER_TOOL):
        super().__init__(name=name, sub_agents=[llm_agent])
        self._llm_agent = llm_agent

    @override
    async def _run_async_impl(self, ctx):

        input_csv_path = ctx.session.state.get(Constants.SAMPELED_BLENDSAHPE_CSV_PATH_KEY)
        if input_csv_path is None:
            input_csv_path = ctx.session.state.get(Constants.BLENDSHAPE_CSV_PATH_KEY)

        if not input_csv_path or not os.path.exists(input_csv_path):
             msg = "[Failed] Input CSV not found for Prefilter."
             yield Event(
                 content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]),
                 author=self.name
             )
             # Even if it fails, pass the (bad) path along so the next agent can log an error
             ctx.session.state[Constants.PREFILTER_CSV_PATH_KEY] = input_csv_path
             ctx.session.state[Constants.PREFILTER_SUMMARY_KEY] = "Error: Input CSV not found."
             return
        
        # Save context variables for the tool
        out_dir = ctx.session.state.get(Constants.WORK_DIR_KEY, Constants.CURRENT_DIR)
        
        ctx.session.state.update({
            Constants.CSV_PATH_KEY: input_csv_path, # Use the definite input path
            Constants.OUT_DIR_KEY: out_dir
        })
        
        df = pd.read_csv(input_csv_path)
        markdown_table = df.to_markdown(index=False)
        
        # Generate prefilter instruction dynamically
        instruction_filled = PREFILTER_INSTRUCTION.format(
            original_fps=Constants.DEFAULT_FPS,
            markdown_table=markdown_table
        )
        
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
        
        result = ctx.session.state.get(Constants.PREFILTER_RESULT_KEY)
        
        if result is None or not isinstance(result, dict) or not result.get(Constants.USEFUL_KEY):

            msg = f"[Warning] Prefilter tool failed, found no useful frames, or was not called. Using original sampled CSV for next step."
            yield Event(
                content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]),
                author=self.name
            )

            ctx.session.state[Constants.PREFILTER_CSV_PATH_KEY] = input_csv_path
            ctx.session.state[Constants.PREFILTER_SUMMARY_KEY] = "No frames prefiltered; using original."
        else:
            ctx.session.state[Constants.PREFILTER_CSV_PATH_KEY] = result[Constants.FILTERED_CSV_PATH_KEY]
            ctx.session.state[Constants.PREFILTER_SUMMARY_KEY] = f"Kept ranges: {result[Constants.KEPT_RANGES_KEY]}"
            msg = f"[Completed] Prefilter kept frames {result[Constants.KEPT_RANGES_KEY]}, saved to {result[Constants.FILTERED_CSV_PATH_KEY]}"
            yield Event(
                content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]),
                author=self.name
            )



class CSVFilterAgent(BaseAgent):
    def __init__(self, region_llm_agent: LlmAgent, name=Constants.FEATURE_SELECTION_TOOL):
        super().__init__(name=name, sub_agents=[region_llm_agent])
        self._region_llm_agent = region_llm_agent

        #  predefined maps
        self._region_map = {
            Constants.EYES_KEY: Constants.EYE_BLENDSHAPES,
            Constants.MOUTH_KEY: Constants.MOUTH_BLENDSHAPES,
            Constants.NOSE_KEY: Constants.NOSE_BLENDSHAPES,
        }

        # Predefined region-to-AU mappings (example — adjust as per your AU naming convention)
        self._au_region_map = {
            Constants.EYES_KEY: Constants.EYE_AUS,  
            Constants.MOUTH_KEY: Constants.MOUTH_AUS,  
            Constants.NOSE_KEY: Constants.NOSE_AUS,  
        }

        # Fixed emotion columns
        self._emotion_cols = Constants.EMOTIONS



    async def _run_async_impl(self, ctx):
        
        csv_path = ctx.session.state.get(Constants.PREFILTER_CSV_PATH_KEY)

        if csv_path is None:
            csv_path = ctx.session.state.get(Constants.BLENDSHAPE_CSV_PATH_KEY)


        if not csv_path or not os.path.exists(csv_path):
            msg = "[Failed] Prefiltered CSV not found."
            yield Event(content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]), author=self.name)
            return

        # Save context variables
        out_dir = ctx.session.state.get(Constants.WORK_DIR_KEY, Constants.CURRENT_DIR)
        ctx.session.state.update({
            Constants.CSV_PATH_KEY: csv_path,
            Constants.REGION_MAP_KEY: self._region_map,
            Constants.AU_REGION_MAP_KEY: self._au_region_map,
            Constants.EMOTION_COLS_KEY: self._emotion_cols,
            Constants.OUT_DIR_KEY: out_dir
        })

        # Convert CSV to markdown for LLM analysis
        df = pd.read_csv(csv_path)
        markdown_table = df.to_markdown(index=False)

        
        instruction_filled = REGION_DETECTOR_INSTRUCTION.format(original_fps=Constants.DEFAULT_FPS,markdown_table=markdown_table)

        
    

        # Create LLM agent with the tool
        temp_llm_agent = LlmAgent(
            name=Constants.REGIONDETECTOR_AGENT,
            model=MODEL_NAME,
            instruction=instruction_filled,
            tools=[csv_filter_tool],
            
        )

        # Run LLM asynchronously
        async for ev in temp_llm_agent.run_async(ctx):
            if not ev.author:
                ev.author = temp_llm_agent.name
            yield ev



        # Retrieve tool result
        result = ctx.session.state.get(Constants.FILTERED_CSV_RESULT_KEY)
        
    
        if result is None:
            msg = "[Failed] Tool was not called. No result found."
            yield Event(content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]), author=self.name)
            return
        
        if isinstance(result, str):
            msg = f"[Failed] Tool returned string instead of dict: {result}"
            yield Event(content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]), author=self.name)
            return
        
        if not result.get("success"):
            msg = f"[Failed] CSV filtering failed: {result.get(Constants.REASON_KEY, 'unknown error')}"
            yield Event(content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]), author=self.name)
            return

        ctx.session.state[Constants.FILTERED_BLENDSHAPE_CSV_PATH_KEY] = result[Constants.FILTERED_CSV_PATH_KEY]
        msg = f"[Completed] Filtered CSV saved: {result[Constants.FILTERED_CSV_PATH_KEY]}\nSelected regions: {result.get(Constants.REGIONS_KEY)}"
        yield Event(content=types.Content(role=Constants.ASSISTANT_ROLE, parts=[types.Part(text=msg)]), author=self.name)

 

class LlmOrchestratorAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()
    _tools: dict = PrivateAttr()

    def __init__(self, llm_agent: LlmAgent, tools: list[BaseAgent], name=Constants.LLM_ORCHESTRATOR):
        super().__init__(name=name, sub_agents=[llm_agent] + tools)
        self._llm_agent = llm_agent
        self._tools = {t.name: t for t in tools}

    @override
    async def _run_async_impl(self, ctx):
        
        
        disease_focus = ctx.session.state.get(Constants.DISEASE_FOCUS_KEY, Constants.EMPTY_STRING)
        if disease_focus:
            yield Event(
                content=types.Content(
                    role=Constants.ASSISTANT_ROLE,
                    parts=[types.Part(text=f"⚕️ Disease focus confirmed: {disease_focus}")]
                ),
                author=self.name
            )

        while True:
            ctx.session.state[Constants.ORCHESTRATOR_MEMORY_SUMMARY_KEY] = build_state_summary(ctx.session.state)
            state_summary = ctx.session.state[Constants.ORCHESTRATOR_MEMORY_SUMMARY_KEY]

            self._llm_agent.instruction = ORCHESTRATOR_INSTRUCTION.format(summary=state_summary)
            async for ev in self._llm_agent.run_async(ctx):
                if not ev.author:
                    ev.author = self._llm_agent.name
                yield ev

            decision_raw = ctx.session.state.get(Constants.ORCHESTRATOR_KEY, Constants.EMPTY_STRING)
            try:
                match = re.search(r"```json\s*(\{.*?\})\s*```", decision_raw, re.DOTALL)
                decision = json.loads(match.group(1)) if match else {}
            except Exception:
                decision = {Constants.NEXT_TOOL_KEY: Constants.STOP, Constants.REASON_KEY: "Could not parse decision."}

            next_tool = decision.get(Constants.NEXT_TOOL_KEY, Constants.STOP)
            reason = decision.get(Constants.REASON_KEY, Constants.EMPTY_STRING)

            yield Event(
                content=types.Content(
                    role=Constants.ASSISTANT_ROLE,
                    parts=[types.Part(text=f"[Action] Orchestrator chose: {next_tool} ({reason})")]
                ),
                author=self.name
            )

            if next_tool == Constants.STOP:
                break
            
            if next_tool not in self._tools:
                yield Event(
                    content=types.Content(
                        role=Constants.ASSISTANT_ROLE,
                        parts=[types.Part(text=f"[Failed] Unknown tool {next_tool}, stopping.")]
                    ),
                    author=self.name
                )
                break

            tool = self._tools[next_tool]
            
            async for ev in tool.run_async(ctx):
                if not ev.author:
                    ev.author = tool.name
                yield ev
        
        yield Event(
            content=types.Content(
                role=Constants.ASSISTANT_ROLE,
                parts=[types.Part(text="[Completed] Vision pipeline processing finished.")]
            ),
            author=self.name
        )



