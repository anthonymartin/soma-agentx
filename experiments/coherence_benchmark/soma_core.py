"""
This file is part of SOMA (Self-Organizing Memory Architecture).

Copyright © 2025 Cadenzai, Inc.

SOMA is free software: you can redistribute it and/or modify it under the terms of the 
GNU Affero General Public License as published by the Free Software Foundation, either 
version 3 of the License, or (at your option) any later version.

SOMA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with SOMA. 
If not, see <https://www.gnu.org/licenses/>.

SOMA (Self-Organizing Memory Architecture) Implementation with LangGraph

Updated to use dictionary-style state compatible with current LangGraph version.
Added persistence for coherence benchmarks to CSV file and debug logging functionality.
"""

import os
import json
import uuid
import time
import csv
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import argparse
import hashlib

# Import for custom LLM client
from openai import OpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END

# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logger(debug_mode=False, log_file='soma_debug.log'):
    """Configure logging with file and console output."""
    # Create logger
    logger = logging.getLogger('soma')
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# =============================================================================
# Coherence Benchmark Persistence
# =============================================================================

class CoherenceBenchmarkPersistence:
    """Handles persistence of coherence benchmark data to CSV."""
    
    def __init__(self, csv_file='coherence_benchmarks.csv'):
        """Initialize persistence with file path."""
        self.csv_file = csv_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 
                    'session_id', 
                    'turn_number', 
                    'intent_type',
                    'referential_integrity', 
                    'lexical_consistency', 
                    'contextual_relevance', 
                    'structural_continuity', 
                    'tone_consistency',
                    'directive_adherence',
                    'overall_score',
                    'directives'
                ])
    
    def save_benchmark(self, session_id: str, turn_number: int, 
                      intent_type: str, coherence_score: Dict[str, Any],
                      directives: List[str] = None):
        """Save a coherence benchmark data point to CSV."""
        if not coherence_score:
            return
        
        # Format directives for CSV storage
        directive_text = "|".join(directives) if directives else ""
        
        # Format data for CSV
        row_data = [
            datetime.now().isoformat(),
            session_id,
            turn_number,
            intent_type,
            coherence_score.get('referential_integrity', 0),
            coherence_score.get('lexical_consistency', 0),
            coherence_score.get('contextual_relevance', 0),
            coherence_score.get('structural_continuity', 0),
            coherence_score.get('tone_consistency', 0),
            coherence_score.get('directive_adherence', 0),
            coherence_score.get('overall_score', 0),
            directive_text
        ]
        
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

# =============================================================================
# Custom LLM Client
# =============================================================================

class LLMClient:
    """Custom LLM client using the provided interface."""
    
    def __init__(self, temperature=0.3, logger=None):
        """Initialize LLM client with provided interface."""
        self.temperature = temperature
        self.OPENAI_BASE = "https://xxxxxxx"
        self.OPENAI_API_KEY = 'nothing'
        self.MODEL_NAME = "gaunernst/gemma-3-27b-it-int4-awq"
        self.client = OpenAI(api_key=self.OPENAI_API_KEY, base_url=self.OPENAI_BASE)
        self.logger = logger
    
    def chat(self, system: str, user: str) -> str:
        """Call the LLM with system and user prompts."""
        try:
            if self.logger:
                self.logger.debug(f"LLM Request - System: {system[:100]}... User: {user[:100]}...")
            
            start_time = time.time()
            resp = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                temperature=self.temperature,
                messages=[{"role": "system", "content": system},
                        {"role": "user", "content": user}]
            )
            elapsed_time = time.time() - start_time
            
            if self.logger:
                self.logger.debug(f"LLM Response time: {elapsed_time:.2f}s")
            
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM Error: {str(e)}")
            raise

# =============================================================================
# Data Models
# =============================================================================

class Owner(str, Enum):
    """Ownership of a memory structure."""
    SYSTEM = "system"
    AGENT = "agent"
    USER = "user"

@dataclass
class SymbolicTrigger:
    """A symbolic trigger pattern for an EMU."""
    type: str
    value: Any
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if this trigger matches the given context."""
        if self.type == "InteractionTurn":
            return context.get("turn_number") == self.value
        elif self.type == "IntentType":
            return context.get("intent_type") == self.value
        elif self.type == "Tag":
            return self.value in context.get("tags", [])
        elif self.type == "KeywordPresent":
            return self.value.lower() in context.get("text", "").lower()
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "value": self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolicTrigger':
        """Create from dictionary."""
        return cls(
            type=data.get("type", ""),
            value=data.get("value", None)
        )

@dataclass
class MemoryFragment:
    """An immutable memory fragment."""
    id: str
    content: Dict[str, Any]
    owner: Owner
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "owner": self.owner,
            "created_at": self.created_at,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryFragment':
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", {}),
            owner=data.get("owner", Owner.AGENT),
            created_at=data.get("created_at", datetime.now().isoformat()),
            tags=data.get("tags", [])
        )

@dataclass
class EMU:
    """Executable Memory Unit."""
    id: str
    type: str
    trigger: SymbolicTrigger
    predicate: str
    value: Any
    owner: Owner
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    utility_rationale: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    perspective: str = "agent"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "trigger": self.trigger.to_dict(),
            "predicate": self.predicate,
            "value": self.value,
            "owner": self.owner,
            "confidence": self.confidence,
            "tags": self.tags,
            "utility_rationale": self.utility_rationale,
            "timestamp": self.timestamp,
            "perspective": self.perspective
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EMU':
        """Create from dictionary."""
        trigger_data = data.get("trigger", {})
        trigger = SymbolicTrigger.from_dict(trigger_data)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data.get("type", ""),
            trigger=trigger,
            predicate=data.get("predicate", ""),
            value=data.get("value", ""),
            owner=data.get("owner", Owner.AGENT),
            confidence=data.get("confidence", 1.0),
            tags=data.get("tags", []),
            utility_rationale=data.get("utility_rationale", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            perspective=data.get("perspective", "agent")
        )

@dataclass
class EventEnvelope:
    """Container for event information with semantic metadata."""
    text: str
    tags: List[str]
    intent_type: str
    turn_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sentiment: str = "neutral"
    entities: List[Dict[str, str]] = field(default_factory=list)
    context_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "tags": self.tags,
            "intent_type": self.intent_type,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "sentiment": self.sentiment,
            "entities": self.entities,
            "context_info": self.context_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventEnvelope':
        """Create from dictionary."""
        return cls(
            text=data.get("text", ""),
            tags=data.get("tags", []),
            intent_type=data.get("intent_type", ""),
            turn_number=data.get("turn_number", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            sentiment=data.get("sentiment", "neutral"),
            entities=data.get("entities", []),
            context_info=data.get("context_info", {})
        )

@dataclass
class CoherenceScore:
    """Scores for evaluating conversation coherence."""
    referential_integrity: int = 3  # 1-5 scale
    lexical_consistency: int = 3    # 1-5 scale
    contextual_relevance: int = 3   # 1-5 scale
    structural_continuity: int = 3  # 1-5 scale
    tone_consistency: int = 3       # 1-5 scale
    directive_adherence: int = 3    # 1-5 scale - How well response follows the EMU directive
    
    def average(self) -> float:
        """Calculate average score."""
        total = (self.referential_integrity + 
                self.lexical_consistency + 
                self.contextual_relevance + 
                self.structural_continuity + 
                self.tone_consistency + 
                self.directive_adherence)
        return total / 6.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "referential_integrity": self.referential_integrity,
            "lexical_consistency": self.lexical_consistency,
            "contextual_relevance": self.contextual_relevance,
            "structural_continuity": self.structural_continuity,
            "tone_consistency": self.tone_consistency,
            "directive_adherence": self.directive_adherence,
            "overall_score": self.average()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoherenceScore':
        """Create from dictionary."""
        return cls(
            referential_integrity=data.get("referential_integrity", 3),
            lexical_consistency=data.get("lexical_consistency", 3),
            contextual_relevance=data.get("contextual_relevance", 3),
            structural_continuity=data.get("structural_continuity", 3),
            tone_consistency=data.get("tone_consistency", 3),
            directive_adherence=data.get("directive_adherence", 3)
        )

# =============================================================================
# Memory Registry
# =============================================================================

class MemoryRegistry:
    """Storage for EMUs and Memory Fragments."""
    
    def __init__(self, logger=None):
        self.emus: Dict[str, EMU] = {}
        self.fragments: Dict[str, MemoryFragment] = {}
        self.merkle_roots: Dict[str, str] = {}
        self.logger = logger
        
    def add_emu(self, emu: EMU) -> None:
        """Add an EMU to the registry."""
        self.emus[emu.id] = emu
        # Generate provenance commitment
        content_str = json.dumps(emu.to_dict(), sort_keys=True)
        self.merkle_roots[emu.id] = hashlib.sha256(content_str.encode()).hexdigest()
        
        if self.logger:
            self.logger.debug(f"Added EMU: {emu.id} - Type: {emu.type} - Predicate: {emu.predicate}")
        
    def add_fragment(self, fragment: MemoryFragment) -> None:
        """Add a memory fragment to the registry."""
        self.fragments[fragment.id] = fragment
        # Generate provenance commitment
        content_str = json.dumps(fragment.to_dict(), sort_keys=True)
        self.merkle_roots[fragment.id] = hashlib.sha256(content_str.encode()).hexdigest()
        
        if self.logger:
            self.logger.debug(f"Added Fragment: {fragment.id} - Tags: {fragment.tags}")
        
    def get_emu(self, emu_id: str) -> Optional[EMU]:
        """Get an EMU by ID."""
        return self.emus.get(emu_id)
    
    def get_fragment(self, fragment_id: str) -> Optional[MemoryFragment]:
        """Get a memory fragment by ID."""
        return self.fragments.get(fragment_id)
    
    def get_emus_by_tags(self, tags: List[str]) -> List[EMU]:
        """Get EMUs with specified tags."""
        return [emu for emu in self.emus.values() 
                if any(tag in emu.tags for tag in tags)]
    
    def get_fragments_by_tags(self, tags: List[str]) -> List[MemoryFragment]:
        """Get memory fragments with specified tags."""
        return [frag for frag in self.fragments.values() 
                if any(tag in frag.tags for tag in tags)]
    
    def retrieve_emus_by_trigger(self, context: Dict[str, Any]) -> List[EMU]:
        """Retrieve EMUs that match the given context."""
        matching_emus = []
        for emu in self.emus.values():
            if emu.trigger.matches(context):
                matching_emus.append(emu)
                if self.logger:
                    self.logger.debug(f"Matched EMU: {emu.id} - For context: {context}")
        return matching_emus
    
    def verify_provenance(self, memory_id: str, expected_root: str) -> bool:
        """Verify the provenance of a memory structure."""
        return self.merkle_roots.get(memory_id) == expected_root

# =============================================================================
# Domain Classifier
# =============================================================================

class DomainClassifier:
    """Simplified domain classifier for EMU triggering."""
    
    def __init__(self, llm_client: LLMClient, logger=None):
        self.llm_client = llm_client
        self.logger = logger
    
    def classify(self, text: str, turn_number: int) -> EventEnvelope:
        """Classify text and generate event envelope - focused on EMU triggers."""
        # Simple system prompt focused on essential triggers
        system_prompt = """
        Analyze the user message and extract only:
        
        1. INTENT: Select one: fact_seeking, instructional, reflective, creative, social, 
           opinion_seeking, troubleshooting, clarification, metaconversational, emotional
        
        2. TAGS: 3-5 relevant keywords that characterize the message
        
        Format as simple JSON: {"intent": "intent_type", "tags": ["tag1", "tag2", "tag3"]}
        Keep your response concise - just the JSON with no explanation or markdown formatting.
        """
        
        # Use LLM client with minimal request
        result = self.llm_client.chat(system_prompt, text)
        
        try:
            # Clean up any markdown formatting
            cleaned_result = result
            if "```" in result:
                # Extract content between code blocks
                import re
                code_block_matches = re.findall(r"```(?:json)?\s*(.*?)```", result, re.DOTALL)
                if code_block_matches:
                    cleaned_result = code_block_matches[0].strip()
            
            # Parse the minimal output
            parsed = json.loads(cleaned_result)
            intent_type = parsed.get("intent", "social")
            tags = parsed.get("tags", ["unclassified"])
            
            if self.logger:
                self.logger.debug(f"Classification: Intent={intent_type}, Tags={tags}")
                
        except json.JSONDecodeError as e:
            # Simple fallbacks
            intent_type = "social"
            tags = ["unclassified"]
            
            if self.logger:
                self.logger.error(f"Classification parsing error: {str(e)}, Response: {result}")
        
        # Create envelope with just essential data for EMU triggering
        return EventEnvelope(
            text=text,
            tags=tags,
            intent_type=intent_type,
            turn_number=turn_number
        )

# =============================================================================
# Retrieval Module
# =============================================================================

class RetrievalModule:
    """Handles retrieval of EMUs based on context."""
    
    def __init__(self, registry: MemoryRegistry, logger=None):
        self.registry = registry
        self.logger = logger
        
    def retrieve(self, context: Dict[str, Any]) -> List[EMU]:
        """Retrieve relevant EMUs using the Trigger Channel."""
        emus = self.registry.retrieve_emus_by_trigger(context)
        
        if self.logger:
            self.logger.debug(f"Retrieved {len(emus)} EMUs for context: {context}")
            
        return emus

# =============================================================================
# Response Generator
# =============================================================================

class ResponseGenerator:
    """Generates responses based on retrieved EMUs and context."""
    
    def __init__(self, llm_client: LLMClient, logger=None):
        self.llm_client = llm_client
        self.logger = logger
    
    def generate(self, user_message: str, directives: List[str], conversation_history: List[Dict[str, Any]]) -> str:
        """Generate a response based on directives and context."""
        # Format directives
        directive_text = "\n".join(directives) if directives else "Respond naturally to the user's message."
        
        # Format history for context
        history_str = ""
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n\n"
        
        # Comprehensive system prompt
        system_prompt = f"""
        You are an adaptive conversational agent operating under SOMA (Self-Organizing Memory Architecture).
        Your behavior is guided by Executable Memory Units (EMUs) with symbolic triggers.
        
        Current conversation history:
        {history_str}
        
        The current directives from active EMUs are:
        {directive_text}
        
        Use these directives to shape your response for this turn.
        
        Remember to maintain coherence by applying the directive naturally. Do not explicitly
        reference being guided by EMUs or directives unless directly asked about your operation.
        """
        
        if self.logger:
            self.logger.debug(f"Generating response with {len(directives)} directives")
            
        # Call LLM for response
        response = self.llm_client.chat(system_prompt, user_message)
        
        if self.logger:
            self.logger.debug(f"Generated response: {response[:100]}...")
            
        return response

# =============================================================================
# Coherence-Onboarding Protocol (COP)
# =============================================================================

class CoherenceOnboardingProtocol:
    """Implements the COP for the first 10 interactions."""
    
    def __init__(self, registry: MemoryRegistry, logger=None):
        self.registry = registry
        self.logger = logger
        self._initialize_cop_emus()
    
    def _initialize_cop_emus(self):
        """Initialize EMUs for the first 10 interactions."""
        if self.logger:
            self.logger.info("Initializing COP EMUs for the first 10 interactions")
            
        # Turn 1: Anchor and reframe user input
        self.registry.add_emu(EMU(
            id="emu_cop_t1",
            type="directive",
            trigger=SymbolicTrigger(type="InteractionTurn", value=1),
            predicate="coherence_directive",
            value="Anchor to user's language and reframe their statement",
            owner=Owner.AGENT,
            tags=["cop", "coherence", "first_interaction"],
            utility_rationale="Anchoring improves early grounding and minimizes ambiguity"
        ))
        
        # Turn 2: Echo key terms and ask clarification
        self.registry.add_emu(EMU(
            id="emu_cop_t2",
            type="directive",
            trigger=SymbolicTrigger(type="InteractionTurn", value=2),
            predicate="coherence_directive",
            value="Echo key terms and ask clarification",
            owner=Owner.AGENT,
            tags=["cop", "coherence"],
            utility_rationale="Echoing confirms understanding and builds shared vocabulary"
        ))
        
        # Turn 3: Propose structure or next steps
        self.registry.add_emu(EMU(
            id="emu_cop_t3",
            type="directive",
            trigger=SymbolicTrigger(type="InteractionTurn", value=3),
            predicate="coherence_directive",
            value="Propose structure or next steps for the conversation",
            owner=Owner.AGENT,
            tags=["cop", "coherence"],
            utility_rationale="Establishing direction creates conversational momentum"
        ))
        
        # Turns 4-5: Maintain lexical consistency and begin inference
        self.registry.add_emu(EMU(
            id="emu_cop_t4_5",
            type="directive",
            trigger=SymbolicTrigger(type="InteractionTurn", value=4),
            predicate="coherence_directive",
            value="Maintain lexical consistency and begin drawing inferences",
            owner=Owner.AGENT,
            tags=["cop", "coherence"],
            utility_rationale="Consistent terminology with added insights builds depth"
        ))
        
        self.registry.add_emu(EMU(
            id="emu_cop_t4_5_2",
            type="directive",
            trigger=SymbolicTrigger(type="InteractionTurn", value=5),
            predicate="coherence_directive",
            value="Maintain lexical consistency and begin drawing inferences",
            owner=Owner.AGENT,
            tags=["cop", "coherence"],
            utility_rationale="Consistent terminology with added insights builds depth"
        ))
        
        # Turns 6-8: Reference earlier turns and extend topic
        for turn in range(6, 9):
            self.registry.add_emu(EMU(
                id=f"emu_cop_t{turn}",
                type="directive",
                trigger=SymbolicTrigger(type="InteractionTurn", value=turn),
                predicate="coherence_directive",
                value="Reference earlier turns and extend the topic",
                owner=Owner.AGENT,
                tags=["cop", "coherence"],
                utility_rationale="Callbacks to previous content reinforces memory and continuity"
            ))
        
        # Turn 9: Summarize progression and confirm alignment
        self.registry.add_emu(EMU(
            id="emu_cop_t9",
            type="directive",
            trigger=SymbolicTrigger(type="InteractionTurn", value=9),
            predicate="coherence_directive",
            value="Summarize conversation progression and confirm alignment",
            owner=Owner.AGENT,
            tags=["cop", "coherence"],
            utility_rationale="Summaries build shared understanding of conversational progress"
        ))
        
        # Turn 10: Invite reflection and offer topic continuation or closure
        self.registry.add_emu(EMU(
            id="emu_cop_t10",
            type="directive",
            trigger=SymbolicTrigger(type="InteractionTurn", value=10),
            predicate="coherence_directive",
            value="Invite reflection and offer topic continuation or closure",
            owner=Owner.AGENT,
            tags=["cop", "coherence", "last_interaction"],
            utility_rationale="Reflective closure solidifies learning and sets future expectations"
        ))
        
        # Meta-EMU for turn 10 evaluation
        self.registry.add_emu(EMU(
            id="meta_emu_cop_aggregate",
            type="meta-evaluation",
            trigger=SymbolicTrigger(type="InteractionTurn", value=10),
            predicate="coherence_score",
            value={
                "referential_integrity": 0,
                "lexical_consistency": 0,
                "contextual_relevance": 0,
                "structural_continuity": 0,
                "tone_consistency": 0
            },
            owner=Owner.AGENT,
            tags=["cop", "evaluation", "meta"],
            utility_rationale="Tracking coherence metrics enables continuous improvement"
        ))

    def get_directive_for_turn(self, turn_number: int) -> Optional[str]:
        """Get the coherence directive for a specific turn."""
        context = {"turn_number": turn_number}
        emus = self.registry.retrieve_emus_by_trigger(context)
        
        for emu in emus:
            if emu.predicate == "coherence_directive":
                if self.logger:
                    self.logger.debug(f"COP directive for turn {turn_number}: {emu.value}")
                return emu.value
        
        return None
        
    def update_coherence_score(self, scores: CoherenceScore) -> None:
        """Update the coherence score in the meta-EMU."""
        meta_emu = self.registry.get_emu("meta_emu_cop_aggregate")
        if meta_emu:
            meta_emu.value = {
                "referential_integrity": scores.referential_integrity,
                "lexical_consistency": scores.lexical_consistency,
                "contextual_relevance": scores.contextual_relevance,
                "structural_continuity": scores.structural_continuity,
                "tone_consistency": scores.tone_consistency
            }
            self.registry.add_emu(meta_emu)  # Update in registry
            
            if self.logger:
                self.logger.info(f"Updated meta-EMU coherence scores: {scores.average():.2f}/5.0")

# =============================================================================
# Coherence Evaluator
# =============================================================================

class CoherenceEvaluator:
    """Evaluates conversation coherence using the scoring rubric."""
    
    def __init__(self, llm_client: LLMClient, logger=None):
        self.llm_client = llm_client
        self.logger = logger
    
    def evaluate(self, history: List[Dict[str, Any]], user_message: str, ai_response: str, directives: List[str] = None) -> CoherenceScore:
        """Evaluate coherence of the current response."""
        # Format history for prompt
        formatted_history = ""
        for entry in history:
            if entry["role"] == "user":
                formatted_history += f"User: {entry['content']}\n"
            else:
                formatted_history += f"AI: {entry['content']}\n"
        
        # Format directives if provided
        directive_text = ""
        if directives and len(directives) > 0:
            directive_text = "Response should follow these directives:\n" + "\n".join([f"- {d}" for d in directives])
        
        system_prompt = f"""
        You are a specialized coherence evaluator for conversational AI.
        Score the conversation on a scale of 1-5 (where 5 is best) for each:
        
        1. Referential Integrity: How well references entities from previous turns
        2. Lexical Consistency: Consistency in terminology and vocabulary
        3. Contextual Relevance: Alignment with topic and context
        4. Structural Continuity: Logical flow and structure
        5. Tone Consistency: Consistency in voice and style
        6. Directive Adherence: How well the response follows the given directives
        
        {directive_text}
        
        Return ONLY a JSON object with numerical scores, with no markdown formatting or explanation:
        {{"referential_integrity": 4, "lexical_consistency": 5, "contextual_relevance": 3, "structural_continuity": 4, "tone_consistency": 5, "directive_adherence": 4}}
        """
        
        user_prompt = f"""
        Conversation History:
        {formatted_history}
        
        Current User Message:
        {user_message}
        
        Current AI Response:
        {ai_response}
        """
        
        if self.logger:
            self.logger.debug("Evaluating coherence of response")
            if directives:
                self.logger.debug(f"Evaluating adherence to directives: {directives}")
            
        # Invoke LLM for evaluation
        result = self.llm_client.chat(system_prompt, user_prompt)
        
        try:
            # Clean up the response in case it contains markdown code blocks
            cleaned_result = result
            if "```" in result:
                # Extract content between code blocks
                import re
                code_block_matches = re.findall(r"```(?:json)?\s*(.*?)```", result, re.DOTALL)
                if code_block_matches:
                    cleaned_result = code_block_matches[0].strip()
            
            # Parse JSON response
            parsed = json.loads(cleaned_result)
            
            # Create coherence score
            coherence_score = CoherenceScore(
                referential_integrity=parsed.get("referential_integrity", 3),
                lexical_consistency=parsed.get("lexical_consistency", 3),
                contextual_relevance=parsed.get("contextual_relevance", 3),
                structural_continuity=parsed.get("structural_continuity", 3),
                tone_consistency=parsed.get("tone_consistency", 3),
                directive_adherence=parsed.get("directive_adherence", 3)
            )
            
            if self.logger:
                self.logger.debug(f"Coherence evaluation: {coherence_score.to_dict()}")
                
            return coherence_score
            
        except json.JSONDecodeError as e:
            # Default scores on parsing error
            if self.logger:
                self.logger.error(f"Coherence evaluation parsing error: {str(e)}, Response: {result}")
                
            return CoherenceScore()

# =============================================================================
# SOMA Core
# =============================================================================

class SOMACore:
    """Core SOMA implementation."""
    
    def __init__(self, llm_client: LLMClient, logger=None, benchmark_persistence=None):
        self.llm_client = llm_client
        self.logger = logger
        self.benchmark_persistence = benchmark_persistence
        self.session_id = str(uuid.uuid4())
        
        if self.logger:
            self.logger.info(f"Initializing SOMA Core - Session ID: {self.session_id}")
        
        self.registry = MemoryRegistry(logger=logger)
        self.domain_classifier = DomainClassifier(llm_client, logger=logger)
        self.retrieval_module = RetrievalModule(self.registry, logger=logger)
        self.response_generator = ResponseGenerator(llm_client, logger=logger)
        self.coherence_protocol = CoherenceOnboardingProtocol(self.registry, logger=logger)
        self.coherence_evaluator = CoherenceEvaluator(llm_client, logger=logger)
        self.conversation_history = []
        self.turn_counter = 0
        self.state = {"turn_number": 0}  # Add persistent state tracking
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message and generate a response."""
        # Increment turn counter
        self.turn_counter += 1
        self.state["turn_number"] = self.turn_counter  # Update persistent state
        
        if self.logger:
            self.logger.info(f"Processing message - Turn {self.turn_counter}")
            self.logger.debug(f"User message: {message[:100]}...")
        
        # Store user message in history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Classify the message
        envelope = self.domain_classifier.classify(message, self.turn_counter)
        
        # Retrieve relevant EMUs
        emus = self.retrieval_module.retrieve(envelope.to_dict())
        
        # Extract directives from EMUs
        directives = [emu.value for emu in emus if emu.predicate == "coherence_directive"]
        
        # Get specific COP directive for this turn if available
        cop_directive = self.coherence_protocol.get_directive_for_turn(self.turn_counter)
        if cop_directive and cop_directive not in directives:
            directives.append(cop_directive)
        
        # Generate response
        response = self.response_generator.generate(
            message, directives, self.conversation_history
        )
        
        # Store response in history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Evaluate coherence if we have enough history
        coherence_score = None
        if len(self.conversation_history) >= 2:
            coherence_score = self.coherence_evaluator.evaluate(
                self.conversation_history[:-2],  # History excluding latest exchange
                message,
                response,
                directives  # Pass directives to evaluator
            )
            
            # Update coherence scores in meta-EMU for turn 10
            if self.turn_counter == 10:
                self.coherence_protocol.update_coherence_score(coherence_score)
            
            # Save coherence benchmark if persistence is enabled
            if self.benchmark_persistence and coherence_score:
                self.benchmark_persistence.save_benchmark(
                    self.session_id,
                    self.turn_counter,
                    envelope.intent_type,
                    coherence_score.to_dict(),
                    directives  # Pass directives to save in CSV
                )
        
        # Return the response and other information
        return {
            "response": response,
            "turn": self.turn_counter,
            "intent_type": envelope.intent_type,
            "tags": envelope.tags,
            "directives": directives,
            "coherence_score": coherence_score.to_dict() if coherence_score else None
        }

# =============================================================================
# LangGraph Integration
# =============================================================================

class SOMALangGraph:
    """SOMA implementation as a LangGraph."""
    
    def __init__(self, llm_client: LLMClient, debug: bool = False, 
                 benchmark_file: str = 'coherence_benchmarks.csv', 
                 log_file: str = 'soma_debug.log'):
        """Initialize the LangGraph."""
        # Setup logger
        self.logger = setup_logger(debug, log_file)
        self.logger.info("Initializing SOMA LangGraph")
        
        self.debug = debug
        self.llm_client = llm_client
        llm_client.logger = self.logger
        
        # Setup benchmark persistence
        self.benchmark_persistence = CoherenceBenchmarkPersistence(benchmark_file)
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())
        self.logger.info(f"New session started - ID: {self.session_id}")
        
        # Initialize SOMA components
        self.soma = SOMACore(
            llm_client, 
            logger=self.logger, 
            benchmark_persistence=self.benchmark_persistence
        )
        
        # Build and compile the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        # Initialize session state to track conversation across calls
        self.session_state = {
            "turn_number": 0,
            "messages": []
        }
        
    def get_graph_visualization(self, output_file=None):
        """Generate a visualization of the LangGraph structure."""
        return visualize_graph(self.graph, output_file)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph with dictionary state."""
        self.logger.info("Building LangGraph")
        
        # Create graph with dictionary state
        graph = StateGraph(dict)
        
        # Add nodes
        graph.add_node("classify", self.classify_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("generate", self.generate_node)
        graph.add_node("evaluate", self.evaluate_node)
        
        # Add edges - the core SOMA pattern
        graph.add_edge("classify", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_edge("evaluate", END)
        
        # Set entry point
        graph.set_entry_point("classify")
        
        return graph
    
    def classify_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the user message and create event envelope."""
        # Extract current message
        current_message = state.get("current_message", "")
        if not current_message:
            return state
        
        # Use the session state's turn counter and messages
        turn_number = self.session_state["turn_number"] + 1
        self.session_state["turn_number"] = turn_number  # Update session state
        
        messages = self.session_state.get("messages", [])
        
        # Add user message to history
        messages.append({
            "role": "user",
            "content": current_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update session messages
        self.session_state["messages"] = messages
        
        # Classify message
        envelope = self.soma.domain_classifier.classify(
            current_message, turn_number
        )
        
        # Update state for this graph execution
        state["turn_number"] = turn_number
        state["messages"] = messages.copy()  # Copy to avoid reference issues
        state["intent_type"] = envelope.intent_type
        state["tags"] = envelope.tags
        state["event_envelope"] = envelope.to_dict()
        
        if self.debug:
            self.logger.info(f"Turn {turn_number}")
            self.logger.info(f"Intent: {envelope.intent_type}")
            self.logger.info(f"Tags: {', '.join(envelope.tags)}")
        
        return state
    
    def retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve EMUs based on the event envelope."""
        # Extract event envelope
        event_envelope = state.get("event_envelope")
        if not event_envelope:
            return state
        
        # Get turn number
        turn_number = state.get("turn_number", 0)
        
        # Retrieve EMUs
        emus = self.soma.retrieval_module.retrieve(event_envelope)
        
        # Extract directives
        directives = [emu.value for emu in emus 
                      if emu.predicate == "coherence_directive"]
        
        # Add COP directive for this turn if available
        cop_directive = self.soma.coherence_protocol.get_directive_for_turn(turn_number)
        if cop_directive and cop_directive not in directives:
            directives.append(cop_directive)
        
        # Update state
        state["directives"] = directives
        state["active_emus"] = [emu.to_dict() for emu in emus]
        
        if self.debug:
            self.logger.info(f"Retrieved {len(emus)} EMUs")
            self.logger.info(f"Directives: {directives}")
        
        return state
    
    def generate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on directives."""
        # Extract needed data
        current_message = state.get("current_message", "")
        if not current_message:
            return state
        
        directives = state.get("directives", [])
        messages = state.get("messages", [])
        
        # Generate response
        response = self.soma.response_generator.generate(
            current_message, 
            directives, 
            messages
        )
        
        # Update state
        state["response"] = response
        
        # Add to history in both state and session
        messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        state["messages"] = messages
        self.session_state["messages"] = messages.copy()  # Update session messages
        
        if self.debug:
            self.logger.info(f"Generated response: {response[:100]}...")
        
        return state
    
    def evaluate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate coherence of the response."""
        # Extract needed data
        messages = state.get("messages", [])
        if len(messages) < 2:
            return state
        
        current_message = state.get("current_message", "")
        response = state.get("response", "")
        turn_number = state.get("turn_number", 0)
        intent_type = state.get("intent_type", "")
        directives = state.get("directives", [])
        
        # Prepare history for evaluation (exclude current exchange)
        history = messages[:-2] if len(messages) > 2 else []
        
        # Evaluate coherence, passing the directives
        coherence_score = self.soma.coherence_evaluator.evaluate(
            history, current_message, response, directives
        )
        
        # Update state
        state["coherence_score"] = coherence_score.to_dict()
        
        # Update meta-EMU for turn 10
        if turn_number == 10:
            self.soma.coherence_protocol.update_coherence_score(coherence_score)
        
        # Save coherence benchmark
        self.benchmark_persistence.save_benchmark(
            self.session_id,
            turn_number,
            intent_type,
            coherence_score.to_dict()
        )
        
        if self.debug and turn_number > 0:
            self.logger.info(f"Coherence score: {coherence_score.average():.2f}/5.0")
            if coherence_score.directive_adherence:
                self.logger.info(f"Directive adherence score: {coherence_score.directive_adherence}/5.0")
        
        return state
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message through the graph."""
        # Create initial state
        state = {"current_message": message}
        
        # Log the incoming message
        self.logger.info(f"Processing message: {message[:50]}...")
        
        # Run the graph
        result_state = self.app.invoke(state)
        
        # Log the result
        if "coherence_score" in result_state and result_state["coherence_score"]:
            self.logger.info(f"Processed message with coherence score: {result_state['coherence_score'].get('overall_score', 0):.2f}/5.0")
        
        # Return relevant information from dict state
        return {
            "response": result_state.get("response", ""),
            "turn": result_state.get("turn_number", 0),
            "intent_type": result_state.get("intent_type", ""),
            "tags": result_state.get("tags", []),
            "directives": result_state.get("directives", []),
            "coherence_score": result_state.get("coherence_score")
        }
    
    def run_interactive_session(self, max_turns: int = 10):
        """Run an interactive chat session."""
        print("\n==== SOMA Coherence-Onboarding Protocol (COP) Chatbot ====")
        print("This chatbot implements the SOMA architecture with COP.")
        print("Type 'exit' to end the conversation.")
        print("=" * 60)
        
        self.logger.info(f"Starting interactive session - Max turns: {max_turns}")
        
        turn_counter = 0
        
        while turn_counter < max_turns:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                self.logger.info("User ended conversation with 'exit' command")
                break
            
            # Process message
            result = self.process_message(user_input)
            turn_counter += 1
            
            # Display response
            print(f"\nAssistant ({turn_counter}/{max_turns}): {result['response']}")
            
            # Display coherence score
            if self.debug and "coherence_score" in result and result["coherence_score"]:
                score = result["coherence_score"]["overall_score"]
                print(f"\n[Coherence Score: {score:.2f}/5.0]")
                # Display individual dimensions if in debug mode
                print("[Component Scores]")
                for key, value in result["coherence_score"].items():
                    if key != "overall_score":
                        print(f"  - {key.replace('_', ' ').title()}: {value:.1f}")
        
        self.logger.info("Interactive session ended")
        print("\nConversation ended. Thank you for using the SOMA COP Chatbot!")
        print(f"Coherence scores saved to: {self.benchmark_persistence.csv_file}")
        print(f"Debug logs saved to: {self.logger.handlers[0].baseFilename}")

# =============================================================================
# Graph Visualization Functions
# =============================================================================

def visualize_graph(graph, output_file=None):
    """Visualize the LangGraph structure."""
    try:
        # Import graphviz for visualization
        import graphviz
        
        # Create a new Digraph
        dot = graphviz.Digraph(comment="SOMA LangGraph")
        
        # Based on the debug output, graph.nodes and graph.edges are available
        # Add nodes
        for node_name, node_spec in graph.nodes.items():
            # Use 'classify' as the entry point since we set it as such in our code
            if node_name == "classify":
                dot.node(node_name, f"{node_name} (Entry)", style="filled", fillcolor="lightblue")
            else:
                dot.node(node_name, node_name)
        
        # Add special nodes for start and end
        dot.node("__start__", "START", style="filled", fillcolor="yellow")
        dot.node("__end__", "END", style="filled", fillcolor="lightgreen")
        
        # Add edges - edges in the graph are stored as a set of tuples (source, target)
        for edge in graph.edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                source, target = edge
                dot.edge(source, target)
        
        # Render the graph
        if output_file:
            try:
                dot.render(output_file, format='png', cleanup=True)
                print(f"Graph visualization saved to {output_file}.png")
            except Exception as e:
                if "failed to execute" in str(e) and "dot" in str(e):
                    print("\nGraphviz Error: The 'dot' executable was not found on your system PATH.")
                    print("\nTo install Graphviz:")
                    print("  - On macOS: brew install graphviz")
                    print("  - On Ubuntu/Debian: sudo apt-get install graphviz")
                    print("  - On CentOS/RHEL: sudo yum install graphviz")
                    print("  - On Windows: Download from https://graphviz.org/download/")
                    print("\nAlternatively, here's a text representation of the graph structure:")
                    print("\nNodes:")
                    for node in graph.nodes:
                        print(f"  - {node}")
                    print("\nEdges:")
                    for edge in graph.edges:
                        if isinstance(edge, tuple) and len(edge) == 2:
                            print(f"  - {edge[0]} → {edge[1]}")
                else:
                    print(f"Error rendering graph: {str(e)}")
        
        # Return the dot object for display options
        return dot
    
    except ImportError:
        print("\nGraphviz Python package is not installed.")
        print("You can install it with: pip install graphviz")
        print("\nIn addition, you will need the Graphviz system package:")
        print("  - On macOS: brew install graphviz")
        print("  - On Ubuntu/Debian: sudo apt-get install graphviz")
        print("  - On CentOS/RHEL: sudo yum install graphviz") 
        print("  - On Windows: Download from https://graphviz.org/download/")
        
        # Provide a text representation as fallback
        print("\nHere's a text representation of the graph structure:")
        print("\nNodes:")
        for node in graph.nodes:
            print(f"  - {node}")
        print("\nEdges:")
        for edge in graph.edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                print(f"  - {edge[0]} → {edge[1]}")
        
        return None
    except Exception as e:
        print(f"Error generating graph visualization: {str(e)}")
        print("Dumping graph structure for debugging:")
        for attr in dir(graph):
            if not attr.startswith('__'):  # Skip dunder methods
                try:
                    attr_value = getattr(graph, attr)
                    # For methods, just print that it's a method
                    if callable(attr_value):
                        print(f"graph.{attr} = <method>")
                    else:
                        print(f"graph.{attr} = {attr_value}")
                except Exception as attr_error:
                    print(f"graph.{attr} = <error accessing: {str(attr_error)}>")
        return None

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="SOMA COP Chatbot")
    parser.add_argument("--temp", type=float, default=0.3, help="Temperature setting")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--turns", type=int, default=10, help="Maximum number of turns")
    parser.add_argument("--benchmark-file", type=str, default="coherence_benchmarks.csv", 
                      help="Path to save benchmark data")
    parser.add_argument("--log-file", type=str, default="soma_debug.log", 
                      help="Path to save debug logs")
    parser.add_argument("--visualize", action="store_true", help="Visualize the LangGraph structure")
    parser.add_argument("--graph-output", type=str, default="soma_graph", 
                      help="Output file for graph visualization (without extension)")
    args = parser.parse_args()
    
    # Initialize LLM client
    llm_client = LLMClient(temperature=args.temp)
    
    # Initialize and run app
    app = SOMALangGraph(
        llm_client=llm_client, 
        debug=args.debug,
        benchmark_file=args.benchmark_file,
        log_file=args.log_file
    )
    
    # Visualize the graph if requested
    if args.visualize:
        visualize_graph(app.graph, args.graph_output)
        print(f"Graph structure visualization created at {args.graph_output}.png")
    
    # Run interactive session
    app.run_interactive_session(max_turns=args.turns)

if __name__ == "__main__":
    main()
