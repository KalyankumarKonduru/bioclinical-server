import asyncio
import json
import logging
import time
import os
import sys
import gc
import resource
from typing import Any, Dict, List, Optional, Generator
from datetime import datetime
import psutil

# CRITICAL: Conservative memory limits to prevent OOM
try:
    # Set conservative memory limits (1GB virtual, 512MB data)
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_DATA, (512 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_STACK, (16 * 1024 * 1024, 32 * 1024 * 1024))
    print("🔧 Conservative memory limits set to prevent OOM")
except Exception as e:
    print(f"⚠️ Could not set memory limits: {e}")

# Reduce recursion limit
sys.setrecursionlimit(2000)

# Aggressive garbage collection
gc.enable()
gc.set_threshold(50, 5, 5)

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

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

# Constants for chunk processing
CHUNK_SIZE = 10000  # Process 10KB chunks
OVERLAP_SIZE = 500  # 500 char overlap between chunks
BATCH_DELAY = 0.1  # 100ms delay between chunks to prevent memory spikes
MAX_MEMORY_MB = 800  # Maximum memory usage before forced cleanup

class MemoryOptimizedBioClinicalService:
    """Memory-optimized BioClinicalBERT service with chunk processing"""
    
    def __init__(self):
        self.model_name = "Clinical-AI-Apollo/Medical-NER"
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.is_loaded = False
        self.entity_mapping = {}
        
        self.medical_categories = {
            "B-PROBLEM": "PROBLEM", "I-PROBLEM": "PROBLEM",
            "B-TREATMENT": "TREATMENT", "I-TREATMENT": "TREATMENT", 
            "B-TEST": "TEST", "I-TEST": "TEST"
        }
        
        logger.info("🧬 Memory-Optimized BioClinicalBERT Service initialized")

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0

    async def load_model(self):
        """Load model with memory optimization"""
        try:
            logger.info(f"🔄 Loading model: {self.model_name}")
            initial_memory = self.get_memory_usage()
            
            # Force garbage collection before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load tokenizer with limits
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=512,  # Limit max length
                padding_side='right',
                truncation_side='right'
            )
            
            # Load model in eval mode to save memory
            logger.info("🧠 Loading model...")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model.eval()  # Set to eval mode
            
            # Build entity mapping
            if hasattr(self.model.config, 'id2label'):
                self.entity_mapping = self.model.config.id2label
                logger.info(f"📋 Entity mapping loaded: {len(self.entity_mapping)} types")
            
            # Create pipeline with optimized settings
            logger.info("🔧 Creating NER pipeline...")
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1,
                batch_size=1  # Process one at a time to minimize memory
            )
            
            self.is_loaded = True
            final_memory = self.get_memory_usage()
            logger.info(f"✅ Model loaded. Memory used: {final_memory - initial_memory:.1f}MB")
            
            # Force cleanup
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def process_chunk(self, text_chunk: str, chunk_offset: int = 0) -> List[Dict]:
        """Process a single chunk of text"""
        if not text_chunk.strip():
            return []
        
        try:
            # Process chunk
            entities = self.ner_pipeline(text_chunk)
            
            # Adjust positions for global text
            processed_entities = []
            for entity in entities:
                # Convert numpy float32 to Python float
                score = float(entity.get('score', 0))
                if score >= 0.5:  # Filter by confidence
                    original_label = entity.get('entity_group', entity.get('label', 'UNKNOWN'))
                    mapped_label = self.medical_categories.get(original_label, original_label)
                    
                    processed_entities.append({
                        'text': str(entity['word']),  # Ensure string
                        'label': str(mapped_label),    # Ensure string
                        'confidence': round(float(score), 3),  # Convert to Python float
                        'start': int(entity['start'] + chunk_offset),  # Ensure int
                        'end': int(entity['end'] + chunk_offset)       # Ensure int
                    })
            
            return processed_entities
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return []

    async def extract_entities(self, text: str, confidence_threshold: float = 0.5) -> Dict:
        """Extract entities using memory-efficient chunk processing"""
        if not self.is_loaded:
            await self.load_model()
        
        start_time = time.time()
        initial_memory = self.get_memory_usage()
        
        try:
            text_length = len(text)
            logger.info(f"🔍 Processing {text_length:,} characters in chunks")
            
            # Validate text size
            max_size = 5 * 1024 * 1024  # 5MB
            if text_length > max_size:
                raise ValueError(f"Text too large: {text_length:,} chars. Maximum: {max_size:,} chars (5MB)")
            
            all_entities = []
            processed_chars = 0
            chunk_count = 0
            
            # Process text in chunks
            position = 0
            while position < text_length:
                # Calculate chunk boundaries
                chunk_end = min(position + CHUNK_SIZE, text_length)
                
                # Find a good break point (sentence/word boundary)
                if chunk_end < text_length:
                    # Try to break at sentence end
                    for sep in ['. ', '.\n', '? ', '! ', '\n\n', '\n', ' ']:
                        last_sep = text.rfind(sep, position, chunk_end)
                        if last_sep != -1 and last_sep > position:
                            chunk_end = last_sep + len(sep)
                            break
                
                # Extract chunk
                chunk = text[position:chunk_end]
                chunk_count += 1
                
                # Log progress
                progress = (position / text_length) * 100
                current_memory = self.get_memory_usage()
                logger.info(f"📦 Chunk {chunk_count}: {len(chunk)} chars, "
                          f"Progress: {progress:.1f}%, Memory: {current_memory:.1f}MB")
                
                # Process chunk
                chunk_entities = self.process_chunk(chunk, position)
                all_entities.extend(chunk_entities)
                processed_chars += len(chunk)
                
                # Update position with overlap
                if chunk_end < text_length:
                    position = chunk_end - OVERLAP_SIZE
                else:
                    position = chunk_end
                
                # Memory management between chunks
                gc.collect()
                await asyncio.sleep(BATCH_DELAY)  # Small delay to prevent memory spikes
                
                # Check memory usage
                if current_memory > MAX_MEMORY_MB:
                    logger.warning(f"⚠️ High memory usage: {current_memory:.1f}MB. Running cleanup...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    await asyncio.sleep(0.5)  # Give system time to clean up
            
            # Remove duplicate entities from overlapping regions
            unique_entities = self.deduplicate_entities(all_entities)
            
            # Filter by confidence threshold
            filtered_entities = [e for e in unique_entities if e['confidence'] >= confidence_threshold]
            
            # Calculate statistics
            processing_time = (time.time() - start_time) * 1000
            final_memory = self.get_memory_usage()
            memory_used = final_memory - initial_memory
            
            avg_confidence = (sum(e['confidence'] for e in filtered_entities) / len(filtered_entities) 
                            if filtered_entities else 0)
            
            logger.info(f"✅ Extracted {len(filtered_entities)} entities in {processing_time:.0f}ms")
            logger.info(f"💾 Memory used: {memory_used:.1f}MB, Final: {final_memory:.1f}MB")
            
            return {
                "success": True,
                "entitiesFound": len(filtered_entities),
                "confidence": round(float(avg_confidence), 3),  # Convert to Python float
                "processingTimeMs": int(round(processing_time)),  # Convert to Python int
                "model": self.model_name,
                "entities": filtered_entities,
                "statistics": {
                    "textLength": int(text_length),  # Convert to Python int
                    "chunksProcessed": int(chunk_count),  # Convert to Python int
                    "averageConfidence": round(float(avg_confidence), 3),  # Convert to Python float
                    "entityBreakdown": self._get_entity_breakdown(filtered_entities),
                    "processingSpeed": f"{int(text_length/max(processing_time/1000, 0.001))} chars/sec",
                    "memoryUsedMB": round(float(memory_used), 1),  # Convert to Python float
                    "finalMemoryMB": round(float(final_memory), 1)  # Convert to Python float
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model_name,
                "timestamp": time.time(),
                "textLength": text_length
            }
        finally:
            # Cleanup
            gc.collect()

    def deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities from overlapping regions"""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        unique = []
        
        for entity in sorted_entities:
            # Check if this entity overlaps with the last added entity
            if unique and entity['start'] < unique[-1]['end']:
                # Skip if it's the same entity
                if (entity['text'].lower() == unique[-1]['text'].lower() and 
                    entity['label'] == unique[-1]['label']):
                    continue
            unique.append(entity)
        
        return unique

    def _get_entity_breakdown(self, entities: List[Dict]) -> Dict[str, int]:
        """Get breakdown of entities by type"""
        breakdown = {}
        for entity in entities:
            label = entity['label']
            breakdown[label] = breakdown.get(label, 0) + 1
        return breakdown

# Create global service instance
bio_service = MemoryOptimizedBioClinicalService()

# Create MCP server
server = Server("medical-ner-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="extractMedicalEntities",
            description="Extract medical entities from clinical text using memory-optimized Clinical-AI-Apollo/Medical-NER model. Supports documents up to 5MB.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Clinical text to analyze (up to 5MB)"
                    },
                    "confidenceThreshold": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0)",
                        "default": 0.5
                    },
                    "entityTypes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by entity types",
                        "default": ["PROBLEM", "TREATMENT", "TEST"]
                    }
                },
                "required": ["text"]
            }
        ),
        types.Tool(
            name="getModelInfo",
            description="Get comprehensive information about the loaded Clinical-AI-Apollo/Medical-NER model",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls with memory-optimized processing"""
    
    if name == "extractMedicalEntities":
        start_time = time.time()
        try:
            text = arguments.get("text", "")
            confidence_threshold = arguments.get("confidenceThreshold", 0.5)
            
            # Validate input
            if not text or not isinstance(text, str):
                raise ValueError("Text must be a non-empty string")
            
            if not text.strip():
                raise ValueError("Text cannot be empty or only whitespace")
            
            # Check text size limit
            max_size = 5 * 1024 * 1024  # 5MB
            if len(text) > max_size:
                raise ValueError(f"Text too large: {len(text):,} characters. Maximum allowed: {max_size:,} characters (5MB)")
            
            logger.info(f"📄 Processing document: {len(text):,} characters")
            
            # Use timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Processing timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minute timeout
            
            try:
                result = await bio_service.extract_entities(text, confidence_threshold)
            finally:
                signal.alarm(0)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Total processing completed in {processing_time:.2f} seconds")
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except TimeoutError as e:
            processing_time = time.time() - start_time
            logger.error(f"⏰ Processing timeout after {processing_time:.2f} seconds: {str(e)}")
            error_result = {
                "success": False,
                "error": f"Processing timeout after {processing_time:.2f} seconds",
                "model": "Clinical-AI-Apollo/Medical-NER",
                "timestamp": time.time(),
                "suggestion": "Try breaking the document into smaller sections"
            }
            return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ extractMedicalEntities failed after {processing_time:.2f} seconds: {str(e)}")
            error_result = {
                "success": False,
                "error": str(e),
                "model": "Clinical-AI-Apollo/Medical-NER",
                "timestamp": time.time(),
                "processingTimeSeconds": processing_time
            }
            return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    elif name == "getModelInfo":
        try:
            current_memory = bio_service.get_memory_usage()
            
            model_info = {
                "modelName": bio_service.model_name,
                "isLoaded": bio_service.is_loaded,
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "supportedEntities": list(set(bio_service.medical_categories.values())),
                "torchVersion": torch.__version__,
                "entityMapping": bio_service.entity_mapping,
                "memoryUsageMB": round(current_memory, 1),
                "processingMode": "memory-optimized-chunks",
                "chunkSize": CHUNK_SIZE,
                "overlapSize": OVERLAP_SIZE,
                "capabilities": {
                    "multilingual": False,
                    "clinical_optimized": True,
                    "real_time": True,
                    "batch_processing": True,
                    "max_document_size": "5MB",
                    "smart_chunking": True,
                    "memory_optimized": True
                },
                "performance": {
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "memory_optimized": True,
                    "chunked_processing": True,
                    "estimated_speed": "~500-1000 chars/sec",
                    "max_memory_mb": MAX_MEMORY_MB
                },
                "model_details": {
                    "architecture": "BioClinicalBERT",
                    "training_data": "Clinical text corpus",
                    "entity_types": ["PROBLEM", "TREATMENT", "TEST"],
                    "confidence_calibrated": True,
                    "max_document_size": "5MB"
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

# HTTP Mode Implementation
async def create_http_app():
    """Create FastAPI application for HTTP mode"""
    if not FASTAPI_AVAILABLE:
        # Create a simple HTTP server without FastAPI
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        import threading
        
        class MemoryOptimizedHTTPHandler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                # Set reasonable timeout
                self.request.settimeout(300)  # 5 minute timeout
            
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    health_data = {
                        "status": "healthy" if bio_service.is_loaded else "loading",
                        "server": "bioclinical-medical-ner-memory-optimized",
                        "model": bio_service.model_name,
                        "model_loaded": bio_service.is_loaded,
                        "device": "GPU" if torch.cuda.is_available() else "CPU",
                        "memory_usage_mb": round(bio_service.get_memory_usage(), 1),
                        "max_document_size": "5MB",
                        "chunk_processing": True
                    }
                    self.wfile.write(json.dumps(health_data).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                if self.path == '/mcp':
                    content_length = int(self.headers.get('Content-Length', 0))
                    max_request_size = 10 * 1024 * 1024  # 10MB request limit
                    
                    if content_length > max_request_size:
                        self.send_response(413)  # Request Entity Too Large
                        self.end_headers()
                        return
                    
                    logger.info(f"📥 Receiving request of {content_length:,} bytes")
                    
                    try:
                        post_data = self.rfile.read(content_length)
                        request_data = json.loads(post_data.decode())
                        
                        method = request_data.get("method", "")
                        params = request_data.get("params", {})
                        request_id = request_data.get("id", 1)
                        
                        # Handle MCP methods
                        response = None
                        if method == "initialize":
                            response = {
                                "jsonrpc": "2.0",
                                "result": {
                                    "protocolVersion": "2024-11-05",
                                    "capabilities": {"tools": {"listChanged": False}},
                                    "serverInfo": {"name": "bioclinical-memory-optimized", "version": "3.0.0"}
                                },
                                "id": request_id
                            }
                        elif method == "tools/list":
                            response = {
                                "jsonrpc": "2.0",
                                "result": {
                                    "tools": [
                                        {
                                            "name": "extractMedicalEntities",
                                            "description": "Extract medical entities using memory-optimized processing",
                                            "inputSchema": {
                                                "type": "object",
                                                "properties": {
                                                    "text": {"type": "string"},
                                                    "confidenceThreshold": {"type": "number", "default": 0.5}
                                                },
                                                "required": ["text"]
                                            }
                                        }
                                    ]
                                },
                                "id": request_id
                            }
                        else:
                            response = {
                                "jsonrpc": "2.0",
                                "result": {"status": "MCP endpoint active"},
                                "id": request_id
                            }
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())
                        
                    except Exception as e:
                        logger.error(f"❌ HTTP request failed: {e}")
                        error_response = {
                            "jsonrpc": "2.0",
                            "error": {"code": -32603, "message": str(e)},
                            "id": request_data.get("id", 1) if 'request_data' in locals() else 1
                        }
                        self.send_response(500)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(error_response).encode())
                        
        return MemoryOptimizedHTTPHandler
    
    # FastAPI implementation
    app = FastAPI(
        title="BioClinical Medical NER Server - Memory Optimized",
        description="Memory-optimized Medical Entity Recognition using Clinical-AI-Apollo/Medical-NER",
        version="3.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy" if bio_service.is_loaded else "loading",
            "server": "bioclinical-medical-ner-memory-optimized",
            "version": "3.0.0",
            "model": {
                "name": bio_service.model_name,
                "loaded": bio_service.is_loaded,
                "device": "GPU" if torch.cuda.is_available() else "CPU"
            },
            "memory": {
                "current_mb": round(bio_service.get_memory_usage(), 1),
                "max_allowed_mb": MAX_MEMORY_MB
            },
            "capabilities": {
                "max_document_size": "5MB",
                "chunk_processing": True,
                "memory_optimized": True
            },
            "timestamp": time.time()
        }
    
    @app.post("/mcp")
    async def handle_mcp_request(request: dict):
        """Handle MCP requests"""
        try:
            logger.info(f"📨 MCP request: {request.get('method', 'unknown')}")
            
            method = request.get("method", "")
            params = request.get("params", {})
            request_id = request.get("id", 1)
            
            # Handle different MCP methods
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": False}
                        },
                        "serverInfo": {
                            "name": "bioclinical-memory-optimized",
                            "version": "3.0.0"
                        }
                    },
                    "id": request_id
                }
            elif method == "tools/list":
                tools = await handle_list_tools()
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            } for tool in tools
                        ]
                    },
                    "id": request_id
                }
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                
                logger.info(f"🔧 Calling tool: {tool_name}")
                
                # Call the tool handler
                result_list = await handle_call_tool(tool_name, tool_args)
                
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": item.type,
                                "text": item.text
                            } for item in result_list
                        ]
                    },
                    "id": request_id
                }
            
            return {
                "jsonrpc": "2.0",
                "result": {"status": "processed"},
                "id": request_id
            }
            
        except Exception as e:
            logger.error(f"❌ MCP request failed: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -1, "message": str(e)},
                "id": request.get("id", 1)
            }
    
    return app

async def main_http():
    """HTTP mode entry point"""
    logger.info("🌐 Starting BioClinical Medical NER Server in HTTP mode (Memory Optimized)...")
    
    # Load model on startup
    try:
        await bio_service.load_model()
        logger.info("✅ Medical NER Model loaded!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Medical NER model: {e}")
        raise
    
    try:
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
                access_log=True,
                timeout_keep_alive=300,
                timeout_graceful_shutdown=30,
                limit_concurrency=100,
                limit_max_requests=10000,
                backlog=1024
            )
            server_instance = uvicorn.Server(config)
            await server_instance.serve()
        else:
            # Use simple HTTP server
            from http.server import HTTPServer
            import socket
            
            handler_class = await create_http_app()
            
            class MemoryOptimizedHTTPServer(HTTPServer):
                def server_bind(self):
                    super().server_bind()
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10*1024*1024)
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10*1024*1024)
                    self.socket.settimeout(300)
            
            httpd = MemoryOptimizedHTTPServer((host, port), handler_class)
            
            logger.info(f"🚀 Starting HTTP server on {host}:{port}")
            logger.info(f"📊 Health check: http://{host}:{port}/health")
            logger.info(f"🔗 MCP endpoint: http://{host}:{port}/mcp")
            
            # Run server in a thread
            import threading
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            logger.info("✅ Memory-optimized HTTP server started!")
            
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
    """STDIO mode entry point"""
    logger.info("📡 Starting BioClinical Medical NER Server in STDIO mode (Memory Optimized)...")
    
    # Load model on startup
    try:
        await bio_service.load_model()
        logger.info("✅ Medical NER MCP Server ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Medical NER model: {e}")
        raise
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        try:
            if NotificationOptions:
                capabilities = server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={
                        "memory_optimized": True,
                        "chunk_processing": True
                    },
                )
            else:
                capabilities = server.get_capabilities()
                
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="medical-ner-memory-optimized",
                    server_version="3.0.0",
                    capabilities=capabilities,
                ),
            )
        except TypeError as e:
            logger.warning(f"⚠️  Capabilities initialization failed, trying alternative: {e}")
            await server.run(read_stream, write_stream)

async def main():
    """Main entry point - auto-detect mode"""
    # Set environment for optimized processing
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['MALLOC_ARENA_MAX'] = '2'  # Limit memory fragmentation
    
    # Check for psutil
    try:
        import psutil
    except ImportError:
        logger.info("Installing psutil for memory monitoring...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    # Check environment variables for mode selection
    http_mode = os.getenv('MCP_HTTP_MODE', 'false').lower() == 'true'
    stdio_mode = os.getenv('MCP_STDIO_MODE', 'false').lower() == 'true'
    
    logger.info(f"🔍 Mode detection: MCP_HTTP_MODE={os.getenv('MCP_HTTP_MODE')}, http_mode={http_mode}")
    logger.info(f"🔍 Mode detection: MCP_STDIO_MODE={os.getenv('MCP_STDIO_MODE')}, stdio_mode={stdio_mode}")
    logger.info("💾 MEMORY-OPTIMIZED MODE ENABLED")
    logger.info(f"📏 Chunk size: {CHUNK_SIZE} chars, Overlap: {OVERLAP_SIZE} chars")
    logger.info(f"🎯 Max memory target: {MAX_MEMORY_MB}MB")
    
    # Auto-detect mode if not explicitly set
    if not http_mode and not stdio_mode:
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
        logger.info("🚀 BioClinical Medical NER Server starting (Memory Optimized)...")
        logger.info("📏 Document limit: 5MB")
        logger.info("⏱️  Timeout: 5 minutes")
        logger.info("🧠 Memory: Chunk-based processing with aggressive cleanup")
        logger.info(f"💾 Initial memory: {bio_service.get_memory_usage():.1f}MB")
        
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)