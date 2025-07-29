#!/usr/bin/env python3
"""
BioClinicalBERT MCP Server - Basic Version
Compatible with standard MCP Python SDK
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

# Basic MCP imports - try different import paths
try:
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
except ImportError:
    try:
        from mcp.server.lowlevel import NotificationOptions, Server
        from mcp.server.models import InitializationOptions
        import mcp.server.stdio
        import mcp.types as types
    except ImportError:
        # Final fallback - try the most basic import structure
        from mcp import server
        from mcp import types
        from mcp.server.models import InitializationOptions
        import mcp.server.stdio
        Server = server.Server
        NotificationOptions = getattr(server, 'NotificationOptions', None)

# ML imports
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BioClinicalBERTService:
    def __init__(self):
        # Using Clinical-AI-Apollo/Medical-NER - best available medical NER model
        self.model_name = "Clinical-AI-Apollo/Medical-NER"
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.is_loaded = False
        
        # Medical entity mapping (dynamically loaded from model)
        self.entity_mapping = {}
        # Updated mapping for Clinical-AI-Apollo/Medical-NER model
        self.medical_categories = {
            'B-PROBLEM': 'PROBLEM',
            'I-PROBLEM': 'PROBLEM',
            'B-TREATMENT': 'TREATMENT', 
            'I-TREATMENT': 'TREATMENT',
            'B-TEST': 'TEST',
            'I-TEST': 'TEST',
            'O': 'OTHER'
        }
        
    async def load_model(self):
        """Load BioClinicalBERT model dynamically"""
        if self.is_loaded:
            return
            
        try:
            logger.info(f"Loading Medical NER model: {self.model_name}...")
            start_time = time.time()
            
            # Load tokenizer and model for medical NER
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            # Create NER pipeline with optimized settings for medical text
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",  # Groups B- and I- tags
                device=0 if torch.cuda.is_available() else -1,
                ignore_labels=["O"]  # Skip 'Outside' tokens
            )
            
            # Get label mapping from model config
            if hasattr(self.model.config, 'id2label'):
                self.entity_mapping = self.model.config.id2label
                logger.info(f"Model label mapping: {self.entity_mapping}")
                
                # Update medical_categories based on actual model labels
                if self.entity_mapping:
                    # Clear default mapping and use model's actual labels
                    self.medical_categories = {}
                    for label_id, label_name in self.entity_mapping.items():
                        if label_name.startswith('B-') or label_name.startswith('I-'):
                            base_label = label_name[2:]  # Remove B- or I- prefix
                            self.medical_categories[label_name] = base_label.upper()
                        else:
                            self.medical_categories[label_name] = label_name.upper()
                    
                    logger.info(f"Updated medical categories: {set(self.medical_categories.values())}")
            else:
                logger.warning("Model config doesn't have id2label, using default mapping")
            
            load_time = time.time() - start_time
            logger.info(f"Medical NER model loaded successfully in {load_time:.2f}s")
            logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            logger.info(f"Supported entities: {list(set(self.medical_categories.values()))}")
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load Medical NER model: {str(e)}")
            raise e
    
    def extract_medical_entities(self, text: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Extract medical entities using Clinical-AI-Apollo Medical NER model"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Run NER pipeline
            raw_entities = self.ner_pipeline(text)
            
            # Process entities
            processed_entities = []
            for entity in raw_entities:
                confidence = entity.get('score', 0)
                if confidence >= confidence_threshold:
                    # Get entity label and map to medical category
                    entity_label = entity.get('entity_group', entity.get('entity', 'O'))
                    mapped_label = self.medical_categories.get(entity_label, entity_label)
                    
                    # Skip 'O' (Outside) tokens and low-confidence predictions
                    if mapped_label not in ['O', 'OTHER', 'OUTSIDE']:
                        processed_entity = {
                            'text': entity['word'].replace('##', '').strip(),
                            'label': mapped_label,
                            'confidence': float(confidence),
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

# Create MCP Server with flexible initialization
try:
    server = Server("bioclinical-bert-server")
except TypeError:
    # Some versions might not need the name parameter
    server = Server()

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
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
        types.Tool(
            name="getModelInfo",
            description="Get BioClinicalBERT model information",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
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
                "model": "Clinical-AI-Apollo/Medical-NER",
                "entities": entities
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "model": "Clinical-AI-Apollo/Medical-NER"
            }
            return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
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
            
            return [types.TextContent(type="text", text=json.dumps(model_info, indent=2))]
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e)
            }
            return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point"""
    logger.info("Starting Medical NER MCP Server...")
    
    # Load model on startup
    try:
        await bio_service.load_model()
        logger.info("✅ Medical NER MCP Server ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Medical NER model: {e}")
        raise
    
    # Run the server with flexible capabilities handling
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        try:
            # Try with NotificationOptions if available
            if NotificationOptions:
                capabilities = server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            else:
                # Fallback to basic capabilities
                capabilities = server.get_capabilities()
                
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="medical-ner-server",
                    server_version="1.0.0",
                    capabilities=capabilities,
                ),
            )
        except TypeError as e:
            # Handle different MCP SDK versions
            logger.warning(f"Capabilities initialization failed, trying alternative: {e}")
            await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())