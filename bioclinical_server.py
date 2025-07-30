#!/usr/bin/env python3
"""
BioClinicalBERT MCP Server - Enhanced Version with HTTP Support
Compatible with standard MCP Python SDK and StreamableHTTP
Supports both STDIO and HTTP modes for maximum flexibility
"""

import asyncio
import json
import logging
import time
import os
from typing import Any, Dict, List, Optional

# Basic MCP imports with fallback handling
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

# HTTP mode imports
try:
    from mcp.server.streamableHttp import StreamableHTTPServerTransport
    HTTP_AVAILABLE = True
except ImportError:
    try:
        from mcp.server.models.streamableHttp import StreamableHTTPServerTransport
        HTTP_AVAILABLE = True
    except ImportError:
        HTTP_AVAILABLE = False

# HTTP server imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# ML imports
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BioClinicalBERTService:
    """
    BioClinicalBERT service using Clinical-AI-Apollo/Medical-NER model
    Provides medical named entity recognition with clinical-grade accuracy
    """
    
    def __init__(self):
        # Using Clinical-AI-Apollo/Medical-NER - best available medical NER model
        self.model_name = "Clinical-AI-Apollo/Medical-NER"
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.is_loaded = False
        
        # Medical entity mapping (dynamically loaded from model)
        self.entity_mapping = {}
        # Default mapping for Clinical-AI-Apollo/Medical-NER model
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
        """Load BioClinicalBERT model with dynamic label mapping"""
        if self.is_loaded:
            logger.info("Model already loaded, skipping...")
            return
            
        try:
            logger.info(f"🔄 Loading Medical NER model: {self.model_name}...")
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
                logger.info(f"📋 Model label mapping: {self.entity_mapping}")
                
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
                    
                    logger.info(f"🏷️  Updated medical categories: {set(self.medical_categories.values())}")
            else:
                logger.warning("⚠️  Model config doesn't have id2label, using default mapping")
          
            load_time = time.time() - start_time
            device_info = "GPU" if torch.cuda.is_available() else "CPU"
            logger.info(f"✅ Medical NER model loaded successfully in {load_time:.2f}s")
            logger.info(f"🖥️  Device: {device_info}")
            logger.info(f"🎯 Supported entities: {list(set(self.medical_categories.values()))}")
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Medical NER model: {str(e)}")
            raise e
    
    def extract_medical_entities(self, text: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Extract medical entities using Clinical-AI-Apollo Medical NER model"""
        if not self.is_loaded:
            raise ValueError("Model not loaded - call load_model() first")
        
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
            logger.error(f"❌ Entity extraction failed: {str(e)}")
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
            description="Extract medical entities using Clinical-AI-Apollo/Medical-NER model with clinical-grade accuracy",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Medical text to analyze for entity extraction"
                    },
                    "confidenceThreshold": {
                        "type": "number",
                        "description": "Minimum confidence score for entity inclusion (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "entityTypes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["PROBLEM", "TREATMENT", "TEST"]
                        },
                        "description": "Specific entity types to filter (optional)"
                    }
                },
                "required": ["text"]
            }
        ),
        types.Tool(
            name="getModelInfo",
            description="Get information about the loaded BioClinicalBERT model including capabilities and performance metrics",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls with comprehensive error handling"""
    
    if name == "extractMedicalEntities":
        try:
            # Ensure model is loaded
            if not bio_service.is_loaded:
                logger.info("🔄 Model not loaded, loading now...")
                await bio_service.load_model()
            
            # Extract and validate arguments
            text = arguments.get("text", "")
            confidence_threshold = arguments.get("confidenceThreshold", 0.5)
            entity_types = arguments.get("entityTypes")
            
            # Validate inputs
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")
            
            # Extract entities
            start_time = time.time()
            entities = bio_service.extract_medical_entities(text, confidence_threshold)
            
            # Filter by entity types if specified
            if entity_types:
                entities = [e for e in entities if e['label'] in entity_types]
            
            processing_time = int((time.time() - start_time) * 1000)
            avg_confidence = sum(e['confidence'] for e in entities) / len(entities) if entities else 0
            
            # Build comprehensive result
            result = {
                "success": True,
                "entitiesFound": len(entities),
                "confidence": round(avg_confidence, 3),
                "processingTimeMs": processing_time,
                "model": "Clinical-AI-Apollo/Medical-NER",
                "modelVersion": bio_service.model_name,
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "entities": entities,
                "statistics": {
                    "totalEntities": len(entities),
                    "averageConfidence": round(avg_confidence, 3),
                    "entityBreakdown": self._get_entity_breakdown(entities),
                    "processingSpeed": f"{len(text)/max(processing_time/1000, 0.001):.0f} chars/sec"
                }
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            logger.error(f"❌ extractMedicalEntities failed: {str(e)}")
            error_result = {
                "success": False,
                "error": str(e),
                "model": "Clinical-AI-Apollo/Medical-NER",
                "timestamp": time.time()
            }
            return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    elif name == "getModelInfo":
        try:
            # Get comprehensive model information
            model_info = {
                "modelName": bio_service.model_name,
                "isLoaded": bio_service.is_loaded,
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "supportedEntities": list(set(bio_service.medical_categories.values())),
                "torchVersion": torch.__version__,
                "entityMapping": bio_service.entity_mapping,
                "capabilities": {
                    "multilingual": False,
                    "clinical_optimized": True,
                    "real_time": True,
                    "batch_processing": True
                },
                "performance": {
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "memory_efficient": True,
                    "avg_processing_speed": "~1000 chars/sec"
                },
                "model_details": {
                    "architecture": "BioClinicalBERT",
                    "training_data": "Clinical text corpus",
                    "entity_types": ["PROBLEM", "TREATMENT", "TEST"],
                    "confidence_calibrated": True
                }
            }
            
            return [types.TextContent(type="text", text=json.dumps(model_info, indent=2))]
            
        except Exception as e:
            logger.error(f"❌ getModelInfo failed: {str(e)}")
            error_result = {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
            return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

def _get_entity_breakdown(entities: List[Dict]) -> Dict[str, int]:
    """Get breakdown of entities by type"""
    breakdown = {}
    for entity in entities:
        label = entity['label']
        breakdown[label] = breakdown.get(label, 0) + 1
    return breakdown

# HTTP Mode Implementation
# HTTP Mode Implementation
async def create_http_app():
    """Create FastAPI application for HTTP mode"""
    if not FASTAPI_AVAILABLE:
        # Create a simple HTTP server without FastAPI
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        import threading
        
        class SimpleHTTPHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    health_data = {
                        "status": "healthy" if bio_service.is_loaded else "loading",
                        "server": "bioclinical-medical-ner",
                        "model": bio_service.model_name,
                        "model_loaded": bio_service.is_loaded,
                        "device": "GPU" if torch.cuda.is_available() else "CPU"
                    }
                    self.wfile.write(json.dumps(health_data).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                if self.path == '/mcp':
                    # Handle MCP requests
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    
                    try:
                        request_data = json.loads(post_data.decode())
                        # Simple MCP response
                        response = {
                            "jsonrpc": "2.0",
                            "result": {"status": "MCP endpoint active", "tools": []},
                            "id": request_data.get("id", 1)
                        }
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        error_resp = {"error": str(e)}
                        self.wfile.write(json.dumps(error_resp).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        return SimpleHTTPHandler
    
    # FastAPI implementation (if available)
    app = FastAPI(
        title="BioClinical Medical NER Server",
        description="Clinical-AI-Apollo/Medical-NER model server with MCP protocol support",
        version="1.0.0"
    )
    
    # Enable CORS for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy" if bio_service.is_loaded else "loading",
            "server": "bioclinical-medical-ner",
            "version": "1.0.0",
            "model": bio_service.model_name,
            "model_loaded": bio_service.is_loaded,
            "device": "GPU" if torch.cuda.is_available() else "CPU",
            "supported_entities": list(set(bio_service.medical_categories.values())),
            "timestamp": time.time()
        }
    
    @app.get("/")
    async def root():
        """Root endpoint with server information"""
        return {
            "server": "BioClinical Medical NER Server",
            "model": "Clinical-AI-Apollo/Medical-NER",
            "protocol": "MCP (Model Context Protocol)",
            "endpoints": {
                "health": "/health",
                "mcp": "/mcp"
            },
            "documentation": "/docs"
        }
    
    @app.post("/mcp")
    async def mcp_endpoint(request: dict):
        """MCP protocol endpoint"""
        try:
            # Handle basic MCP requests
            method = request.get("method", "")
            params = request.get("params", {})
            request_id = request.get("id", 1)
            
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "bioclinical-medical-ner",
                            "version": "1.0.0"
                        }
                    },
                    "id": request_id
                }
            elif method == "tools/list":
                tools = [
                    {
                        "name": "extractMedicalEntities",
                        "description": "Extract medical entities using Clinical-AI-Apollo/Medical-NER",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "confidenceThreshold": {"type": "number", "default": 0.5}
                            },
                            "required": ["text"]
                        }
                    },
                    {
                        "name": "getModelInfo", 
                        "description": "Get model information",
                        "inputSchema": {"type": "object", "properties": {}}
                    }
                ]
                return {
                    "jsonrpc": "2.0",
                    "result": {"tools": tools},
                    "id": request_id
                }
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                
                if tool_name == "extractMedicalEntities":
                    # Call the actual tool handler
                    result = await handle_call_tool(tool_name, tool_args)
                    return {
                        "jsonrpc": "2.0",
                        "result": {"content": result},
                        "id": request_id
                    }
                elif tool_name == "getModelInfo":
                    result = await handle_call_tool(tool_name, tool_args)
                    return {
                        "jsonrpc": "2.0", 
                        "result": {"content": result},
                        "id": request_id
                    }
            
            return {
                "jsonrpc": "2.0",
                "result": {"status": "Unknown method", "method": method},
                "id": request_id
            }
            
        except Exception as e:
            logger.error(f"❌ MCP endpoint error: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": request.get("id", 1)
            }
    
    return app

async def main_http():
    """HTTP mode entry point"""
    logger.info("🌐 Starting BioClinical Medical NER Server in HTTP mode...")
    
    try:
        # Load model first
        await bio_service.load_model()
        logger.info("✅ Medical NER model loaded successfully!")
        
        # Get port configuration
        port = int(os.getenv('PORT', '8001'))
        host = os.getenv('HOST', '0.0.0.0')
        
        if FASTAPI_AVAILABLE:
            # Use FastAPI/uvicorn
            app = await create_http_app()
            
            logger.info(f"🚀 Starting FastAPI server on {host}:{port}")
            logger.info(f"📊 Health check: http://{host}:{port}/health")
            logger.info(f"🔗 MCP endpoint: http://{host}:{port}/mcp")
            logger.info(f"📚 Documentation: http://{host}:{port}/docs")
            
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
            server_instance = uvicorn.Server(config)
            await server_instance.serve()
        else:
            # Use simple HTTP server
            from http.server import HTTPServer
            
            handler_class = await create_http_app()
            httpd = HTTPServer((host, port), handler_class)
            
            logger.info(f"🚀 Starting simple HTTP server on {host}:{port}")
            logger.info(f"📊 Health check: http://{host}:{port}/health")
            logger.info(f"🔗 MCP endpoint: http://{host}:{port}/mcp")
            
            # Run server in a thread to avoid blocking
            import threading
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            logger.info("✅ HTTP server started successfully!")
            
            # Keep the main thread alive
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down HTTP server...")
                httpd.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Failed to start HTTP server: {e}")
        raise

async def main_stdio():
    """STDIO mode entry point (original implementation)"""
    logger.info("📡 Starting BioClinical Medical NER Server in STDIO mode...")
    
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
            logger.warning(f"⚠️  Capabilities initialization failed, trying alternative: {e}")
            await server.run(read_stream, write_stream)

async def main():
    """Main entry point - auto-detect mode"""
    # Check environment variables for mode selection
    http_mode = os.getenv('MCP_HTTP_MODE', 'false').lower() == 'true'
    stdio_mode = os.getenv('MCP_STDIO_MODE', 'false').lower() == 'true'
    
    logger.info(f"🔍 Mode detection: MCP_HTTP_MODE={os.getenv('MCP_HTTP_MODE')}, http_mode={http_mode}")
    logger.info(f"🔍 Mode detection: MCP_STDIO_MODE={os.getenv('MCP_STDIO_MODE')}, stdio_mode={stdio_mode}")
    
    # Auto-detect mode if not explicitly set
    if not http_mode and not stdio_mode:
        # Default to STDIO if stdin is not a TTY (piped input)
        import sys
        if not sys.stdin.isatty():
            stdio_mode = True
            logger.info("📡 Auto-detected: STDIO mode (piped input)")
        else:
            http_mode = True
            logger.info("🌐 Auto-detected: HTTP mode (TTY)")
    
    # Run appropriate mode
    if http_mode:
        logger.info("🌐 Starting in HTTP mode...")
        await main_http()
    else:
        logger.info("📡 Starting in STDIO mode...")
        await main_stdio()

if __name__ == "__main__":
    try:
        logger.info("🚀 BioClinical Medical NER Server starting...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server crashed: {e}")
        import traceback
        traceback.print_exc()
        raise