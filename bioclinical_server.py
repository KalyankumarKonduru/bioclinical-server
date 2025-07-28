#!/usr/bin/env python3
"""
BioClinicalBERT MCP Server
Exposes BioClinicalBERT NER capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

# MCP Server imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool,
    TextContent,
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
)

# ML imports
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BioClinicalBERTService:
    def __init__(self):
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.is_loaded = False
        
        # Medical entity mapping (dynamically loaded from model)
        self.entity_mapping = {}
        self.medical_categories = {
            'B-PROBLEM': 'CONDITION',
            'I-PROBLEM': 'CONDITION', 
            'B-TREATMENT': 'PROCEDURE',
            'I-TREATMENT': 'PROCEDURE',
            'B-TEST': 'PROCEDURE',
            'I-TEST': 'PROCEDURE',
            'B-MEDICATION': 'MEDICATION',
            'I-MEDICATION': 'MEDICATION',
            'B-DOSAGE': 'MEASUREMENT',
            'I-DOSAGE': 'MEASUREMENT'
        }
        
    async def load_model(self):
        """Load BioClinicalBERT model dynamically"""
        if self.is_loaded:
            return
            
        try:
            logger.info(f"Loading BioClinicalBERT model: {self.model_name}...")
            start_time = time.time()
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            # Create NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Dynamically get label mapping from model config
            if hasattr(self.model.config, 'id2label'):
                self.entity_mapping = self.model.config.id2label
                logger.info(f"Model label mapping: {self.entity_mapping}")
            else:
                logger.warning("Model config doesn't have id2label, using default mapping")
            
            load_time = time.time() - start_time
            logger.info(f"BioClinicalBERT loaded successfully in {load_time:.2f}s")
            logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load BioClinicalBERT: {str(e)}")
            raise e
    
    def extract_medical_entities(self, text: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Extract medical entities using BioClinicalBERT"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Run NER pipeline
            raw_entities = self.ner_pipeline(text)
            
            # Process entities
            processed_entities = []
            for entity in raw_entities:
                if entity.get('score', 0) >= confidence_threshold:
                    # Map entity label to medical category
                    entity_label = entity.get('entity', entity.get('entity_group', 'O'))
                    mapped_label = self.medical_categories.get(entity_label, entity_label)
                    
                    if mapped_label != 'O':  # Skip outside tokens
                        processed_entity = {
                            'text': entity['word'].replace('##', '').strip(),
                            'label': mapped_label,
                            'confidence': float(entity['score']),
                            'start': int(entity.get('start', 0)),
                            'end': int(entity.get('end', 0)),
                            'context': self._get_context(text, entity.get('start', 0), entity.get('end', 0))
                        }
                        processed_entities.append(processed_entity)
            
            # Post-process: merge adjacent entities and remove duplicates
            return self._postprocess_entities(processed_entities)
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            raise e
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _postprocess_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge adjacent entities and remove duplicates"""
        if not entities:
            return entities
        
        # Sort by position
        entities.sort(key=lambda x: x.get('start', 0))
        
        # Merge adjacent entities of same type
        merged = []
        current = entities[0] if entities else None
        
        for next_entity in entities[1:]:
            if (current and 
                current.get('end', 0) >= next_entity.get('start', 0) - 2 and 
                current['label'] == next_entity['label']):
                # Merge
                current['text'] = f"{current['text']} {next_entity['text']}".strip()
                current['end'] = next_entity.get('end', current.get('end', 0))
                current['confidence'] = max(current['confidence'], next_entity['confidence'])
            else:
                if current:
                    merged.append(current)
                current = next_entity
        
        if current:
            merged.append(current)
        
        return merged

# Initialize the BioClinicalBERT service
bio_service = BioClinicalBERTService()

# Create MCP Server
server = Server("bioclinical-bert-server")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="extractMedicalEntities",
            description="Extract medical entities using BioClinicalBERT",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Medical text to analyze"
                    },
                    "confidenceThreshold": {
                        "type": "number",
                        "description": "Minimum confidence score",
                        "default": 0.5
                    },
                    "entityTypes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity types to filter"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="getModelInfo",
            description="Get BioClinicalBERT model information",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "extractMedicalEntities":
        try:
            # Ensure model is loaded
            if not bio_service.is_loaded:
                await bio_service.load_model()
            
            text = arguments.get("text", "")
            confidence_threshold = arguments.get("confidenceThreshold", 0.5)
            entity_types = arguments.get("entityTypes")
            
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            start_time = time.time()
            entities = bio_service.extract_medical_entities(text, confidence_threshold)
            
            # Filter by entity types if specified
            if entity_types:
                entities = [e for e in entities if e['label'] in entity_types]
            
            processing_time = int((time.time() - start_time) * 1000)
            avg_confidence = sum(e['confidence'] for e in entities) / len(entities) if entities else 0
            
            result = {
                "success": True,
                "entitiesFound": len(entities),
                "confidence": avg_confidence,
                "processingTimeMs": processing_time,
                "model": "BioClinicalBERT",
                "entities": entities
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "model": "BioClinicalBERT"
            }
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    elif name == "getModelInfo":
        try:
            model_info = {
                "modelName": bio_service.model_name,
                "isLoaded": bio_service.is_loaded,
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "supportedEntities": list(set(bio_service.medical_categories.values())),
                "torchVersion": torch.__version__,
                "entityMapping": bio_service.entity_mapping
            }
            
            return [TextContent(type="text", text=json.dumps(model_info, indent=2))]
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e)
            }
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point"""
    logger.info("Starting BioClinicalBERT MCP Server...")
    
    # Load model on startup
    try:
        await bio_service.load_model()
        logger.info("✅ BioClinicalBERT MCP Server ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize BioClinicalBERT: {e}")
        raise
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="bioclinical-bert-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())