import os
import time
import json
import re
import pandas as pd
import logging
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---- API Key ----
API_KEY = ''

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
VECTORSTORE_DIR = "vectorstore"
metadata_map = {}  # file → sheet → columns
field_mapping_cache = {}  # Cache for field mappings
query_embedding_cache = {}  # Cache for query embeddings

# Store original DataFrames for direct querying
original_dataframes = {}  # file → sheet → DataFrame

# ---- Throttled Embeddings ----
class ThrottledEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        time.sleep(5)
        return super().embed_documents(texts)

    def embed_query(self, text):
        if text in query_embedding_cache:
            logger.info("Using cached embedding for query.")
            return query_embedding_cache[text]
        time.sleep(5)
        embedding = super().embed_query(text)
        query_embedding_cache[text] = embedding
        return embedding

# ---- Smart Field Mapping Function ----
def get_smart_field_mapping(query, all_columns):
    """
    Uses AI to intelligently map user queries to actual field names
    """
    cache_key = f"{query}_{hash(str(sorted(all_columns)))}"
    if cache_key in field_mapping_cache:
        logger.info("Using cached field mapping.")
        return field_mapping_cache[cache_key]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)
    
    columns_text = ", ".join(all_columns)
    
    prompt = f"""
You are a smart field mapper. Given a user query and available database columns, identify which columns are most relevant to answer the query.

User Query: "{query}"

Available Columns: {columns_text}

Instructions:
1. Analyze the query to understand what information is being requested
2. Match query terms with column names using semantic understanding
3. Consider synonyms, abbreviations, and related terms
4. Map concepts to actual field names
5. For filtering queries, identify both the filter column and display columns

Return a JSON object with:
1. "filter_columns": Columns to filter on (e.g., "Birthday Month" for july filtering)
2. "display_columns": Columns to show in results (e.g., employee names, IDs)  
3. "value_mappings": Object mapping query values to possible database values
4. "search_strategy": Brief explanation of how to search

Example response:
{{
  "filter_columns": ["Birthday Month", "DOB"],
  "display_columns": ["Full Name", "Name", "Employee ID"],
  "value_mappings": {{
    "july": ["jul", "July", "JULY", "07", "7", "Jul"]
  }},
  "search_strategy": "Filter records where Birthday Month contains july variations"
}}

Query: "{query}"
"""
    
    try:
        response = llm.invoke(prompt).content
        logger.info(f"Field mapping response: {response}")
        
        # Parse JSON response
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            mapping_result = json.loads(match.group(0))
            field_mapping_cache[cache_key] = mapping_result
            return mapping_result
        else:
            return {"filter_columns": [], "display_columns": [], "value_mappings": {}, "search_strategy": ""}
            
    except Exception as e:
        logger.error(f"Field mapping failed: {e}")
        return {"filter_columns": [], "display_columns": [], "value_mappings": {}, "search_strategy": ""}

# ---- AI Result Filter Function ----
def ai_filter_results(query, raw_results):
    """
    Uses AI to filter out results that don't actually satisfy the query
    """
    if not raw_results:
        return []
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)
    
    # Prepare results for AI analysis - limit to first 50 for performance
    results_sample = raw_results[:50] if len(raw_results) > 50 else raw_results
    
    # Create a simplified version for AI analysis
    simplified_results = []
    for i, result in enumerate(results_sample):
        simplified = {
            "index": i,
            "key_fields": {}
        }
        
        # Include only the most relevant fields to avoid token limit
        for key, value in result.items():
            if key not in ['source_file', 'source_sheet', 'row_index']:
                # Convert to string and limit length
                str_value = str(value)[:100] if value else ""
                simplified["key_fields"][key] = str_value
        
        simplified_results.append(simplified)
    
    results_json = json.dumps(simplified_results, indent=2)
    
    prompt = f"""
You are a data quality filter. Given a user query and a list of potential results, identify which results actually satisfy the query requirements.

User Query: "{query}"

Potential Results:
{results_json}

Instructions:
1. Analyze each result to see if it truly matches the query criteria
2. Be strict - only include results that clearly satisfy the query
3. Consider semantic meaning, not just keyword matching
4. For date/time queries, ensure the values actually match the requested criteria
5. Remove any results that are clearly wrong or don't match

Return ONLY a JSON array with the indices of results that should be kept:
[0, 2, 5, 7, ...]

If no results match, return: []

CRITICAL: Return ONLY the JSON array, no other text or explanation.
"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        logger.info(f"AI filter response: {response}")
        
        # Parse the response to get indices
        match = re.search(r'\[(.*?)\]', response)
        if match:
            indices_str = match.group(1)
            if indices_str.strip():
                indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
            else:
                indices = []
        else:
            indices = []
        
        # Filter results based on AI decision
        filtered_results = []
        for idx in indices:
            if 0 <= idx < len(raw_results):
                filtered_results.append(raw_results[idx])
        
        logger.info(f"AI filtered {len(raw_results)} → {len(filtered_results)} results")
        return filtered_results
        
    except Exception as e:
        logger.error(f"AI filtering failed: {e}")
        # Return original results if AI filtering fails
        return raw_results

# ---- Direct DataFrame Query Function ----
def query_dataframes_directly(query, field_mapping):
    """
    Query original DataFrames directly for better accuracy
    """
    results = []
    
    filter_columns = field_mapping.get('filter_columns', [])
    display_columns = field_mapping.get('display_columns', [])
    value_mappings = field_mapping.get('value_mappings', {})
    
    logger.info(f"Direct query - Filter columns: {filter_columns}")
    logger.info(f"Direct query - Display columns: {display_columns}")
    logger.info(f"Direct query - Value mappings: {value_mappings}")
    
    for file_name, sheets in original_dataframes.items():
        for sheet_name, df in sheets.items():
            logger.info(f"Searching in {file_name} - {sheet_name}")
            
            # Find relevant columns in this DataFrame
            df_columns = [col.strip() for col in df.columns]
            
            # Find matching filter columns
            matching_filter_cols = []
            for filter_col in filter_columns:
                for df_col in df_columns:
                    if filter_col.lower() in df_col.lower() or df_col.lower() in filter_col.lower():
                        matching_filter_cols.append(df_col)
                        break
            
            # Find matching display columns
            matching_display_cols = []
            for display_col in display_columns:
                for df_col in df_columns:
                    if display_col.lower() in df_col.lower() or df_col.lower() in display_col.lower():
                        matching_display_cols.append(df_col)
            
            logger.info(f"Matching filter columns: {matching_filter_cols}")
            logger.info(f"Matching display columns: {matching_display_cols}")
            
            if not matching_filter_cols:
                continue
                
            # Apply filters with broader matching for AI to filter later
            filtered_df = df.copy()
            
            for filter_col in matching_filter_cols:
                if filter_col in df.columns:
                    # Apply value mappings with broader matching
                    for query_value, possible_values in value_mappings.items():
                        for possible_value in possible_values:
                            mask = filtered_df[filter_col].astype(str).str.contains(
                                possible_value, case=False, na=False
                            )
                            matching_rows = filtered_df[mask]
                            
                            logger.info(f"Filter: {filter_col} contains '{possible_value}' - Found {len(matching_rows)} rows")
                            
                            # Add matching records
                            for idx, row in matching_rows.iterrows():
                                record = {
                                    "source_file": file_name,
                                    "source_sheet": sheet_name,
                                    "row_index": int(idx)
                                }
                                
                                # Add all columns as fields
                                for col in df.columns:
                                    record[col] = str(row[col]) if pd.notna(row[col]) else ""
                                
                                results.append(record)
    
    # Remove duplicates based on row content
    unique_results = []
    seen = set()
    for result in results:
        # Create a hash of the record content (excluding metadata)
        content_hash = hash(str({k: v for k, v in result.items()
                                if k not in ['source_file', 'source_sheet', 'row_index']}))
        if content_hash not in seen:
            seen.add(content_hash)
            unique_results.append(result)
    
    logger.info(f"Direct query found {len(unique_results)} unique records before AI filtering")
    return unique_results

# ---- Load Excel files and store original DataFrames ----
def load_excel_files(data_folder="data"):
    logger.info("Loading Excel files...")
    docs = []
    global metadata_map
    metadata_map.clear()

    # Generic chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    if not os.path.exists(data_folder):
        logger.error(f"Data folder {data_folder} does not exist!")
        return docs

    excel_files = [f for f in os.listdir(data_folder) if f.endswith(('.xlsx', '.xls'))]
    logger.info(f"Found Excel files: {excel_files}")

    for file in excel_files:
        path = os.path.join(data_folder, file)
        logger.info(f"Processing file: {path}")
        
        try:
            xls = pd.ExcelFile(path)
            metadata_map[file] = {}
            logger.info(f"Sheets in {file}: {xls.sheet_names}")
            
            for sheet in xls.sheet_names:
                try:
                    logger.info(f"Processing sheet: {sheet}")
                    df = pd.read_excel(path, sheet_name=sheet)
                    
                    # Clean column names and handle NaN values
                    df.columns = df.columns.astype(str)
                    df = df.fillna('')
                    
                    metadata_map[file][sheet] = list(df.columns)
                    logger.info(f"Columns in {sheet}: {list(df.columns)}")
                    
                    # Convert to CSV with better formatting
                    csv_text = df.to_csv(index=False, encoding='utf-8')
                    
                    # Create chunks
                    chunks = splitter.split_text(csv_text)
                    
                    for i, chunk in enumerate(chunks):
                        # Generic metadata without hardcoded assumptions
                        doc_metadata = {
                            "file": file,
                            "sheet": sheet,
                            "columns": list(df.columns),
                            "chunk_id": f"{file}_{sheet}_{i}",
                            "total_rows": len(df)
                        }
                        
                        doc = Document(
                            page_content=chunk,
                            metadata=doc_metadata
                        )
                        docs.append(doc)
                        
                        # Log first few documents to verify metadata
                        if len(docs) <= 3:
                            logger.info(f"Created document {len(docs)}: metadata={doc_metadata}")
                    
                    logger.info(f"Processed {file} - {sheet}: {len(df)} rows, {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing {file} - {sheet}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error opening {file}: {e}")
            continue
    
    logger.info(f"Total documents created: {len(docs)}")
    logger.info(f"Metadata map: {metadata_map}")
    return docs

# ---- Build FAISS vector store ----
def build_vector_store(docs):
    logger.info("Building embeddings and saving FAISS index...")
    embeddings = ThrottledEmbeddings(openai_api_key=API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    logger.info(f"Vectorstore saved at {os.path.abspath(VECTORSTORE_DIR)}")
    return vectorstore

# ---- Load or create vectorstore ----
def get_vectorstore():
    if os.path.exists(VECTORSTORE_DIR):
        logger.info(f"LOADING FROM CACHE: {os.path.abspath(VECTORSTORE_DIR)}")
        embeddings = ThrottledEmbeddings(openai_api_key=API_KEY)
        vs = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Cache loaded with {len(vs.index_to_docstore_id)} documents.")
        
        # Load metadata and DataFrames if vectorstore exists but they're empty
        if not metadata_map or not original_dataframes:
            logger.info("Metadata/DataFrames empty, reloading Excel files...")
            load_excel_files("data")
        
        return vs
    else:
        docs = load_excel_files("data")
        return build_vector_store(docs)

# ---- Initialize ----
logger.info("Server starting...")
vectorstore = get_vectorstore()
logger.info(f"Vectorstore has {len(vectorstore.index_to_docstore_id)} documents.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)
logger.info("Server ready.")

# ---- Enhanced API endpoint with AI filtering ----
@app.get("/ask")
async def ask(query: str = Query(...)):
    try:
        logger.info(f"Processing query: {query}")
        
        # Get all available columns for smart mapping
        all_columns = []
        for file, sheets in metadata_map.items():
            for sheet, columns in sheets.items():
                all_columns.extend(columns)
        all_columns = list(dict.fromkeys(all_columns))
        
        # Get smart field mapping
        field_mapping = get_smart_field_mapping(query, all_columns)
        logger.info(f"Field mapping result: {field_mapping}")
        
        # Query DataFrames directly to get raw results
        raw_results = query_dataframes_directly(query, field_mapping)
        logger.info(f"Direct query found {len(raw_results)} raw results")
        
        # Apply AI filtering to remove incorrect results
        filtered_results = ai_filter_results(query, raw_results)
        logger.info(f"AI filtering: {len(raw_results)} → {len(filtered_results)} results")
        
        # Format the final results
        structured_data = []
        for result in filtered_results:
            record = {
                "source_file": result["source_file"],
                "source_sheet": result["source_sheet"],
                "all_fields": {k: v for k, v in result.items()
                             if k not in ['source_file', 'source_sheet', 'row_index']}
            }
            structured_data.append(record)
        
        return JSONResponse({
            "query": query,
            "field_mapping": field_mapping,
            "method_used": "direct_query_with_ai_filtering",
            "structured_data": structured_data,
            "total_records_found": len(structured_data),
            "success": True,
            "debug_info": {
                "raw_results_before_ai_filter": len(raw_results),
                "filtered_results_after_ai_filter": len(filtered_results),
                "ai_filter_reduction": len(raw_results) - len(filtered_results),
                "files_searched": list(original_dataframes.keys()),
                "ai_quality_check": True
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/ask_raw")
async def ask_raw(query: str = Query(...)):
    """API endpoint that returns raw results without AI filtering (for comparison)"""
    try:
        logger.info(f"Processing raw query: {query}")
        
        # Get all available columns for smart mapping
        all_columns = []
        for file, sheets in metadata_map.items():
            for sheet, columns in sheets.items():
                all_columns.extend(columns)
        all_columns = list(dict.fromkeys(all_columns))
        
        # Get smart field mapping
        field_mapping = get_smart_field_mapping(query, all_columns)
        
        # Query DataFrames directly
        raw_results = query_dataframes_directly(query, field_mapping)
        
        return JSONResponse({
            "query": query,
            "field_mapping": field_mapping,
            "method_used": "direct_query_raw_no_ai_filter",
            "data": raw_results,
            "total_records": len(raw_results),
            "success": len(raw_results) > 0,
            "debug_info": {
                "ai_filtering": False,
                "files_searched": list(original_dataframes.keys())
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing raw query: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/debug_dataframes")
async def debug_dataframes():
    """Debug endpoint to see loaded DataFrames"""
    try:
        summary = {}
        
        for file_name, sheets in original_dataframes.items():
            summary[file_name] = {}
            for sheet_name, df in sheets.items():
                summary[file_name][sheet_name] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
                }
        
        return JSONResponse({
            "original_dataframes_loaded": len(original_dataframes) > 0,
            "summary": summary,
            "total_files": len(original_dataframes),
            "total_sheets": sum(len(sheets) for sheets in original_dataframes.values())
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/metadata")
async def get_metadata():
    """Debug endpoint to see available metadata"""
    try:
        # If metadata_map is empty, try to reload it
        if not metadata_map:
            logger.info("Metadata map is empty, loading Excel files to populate metadata...")
            load_excel_files("data")
        
        # Also return column summary for smart mapping
        all_columns = []
        column_by_file = {}
        
        for file, sheets in metadata_map.items():
            column_by_file[file] = {}
            for sheet, columns in sheets.items():
                column_by_file[file][sheet] = columns
                all_columns.extend(columns)
        
        # Remove duplicates while preserving order
        unique_columns = list(dict.fromkeys(all_columns))
        
        return JSONResponse({
            "metadata_map": metadata_map,
            "summary": {
                "total_files": len(metadata_map),
                "total_sheets": sum(len(sheets) for sheets in metadata_map.values()),
                "all_unique_columns": unique_columns,
                "total_unique_columns": len(unique_columns)
            },
            "detailed_structure": column_by_file,
            "status": "success" if metadata_map else "empty",
            "dataframes_loaded": len(original_dataframes) > 0
        })
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return JSONResponse({"error": str(e), "metadata_map": metadata_map}, status_code=500)

@app.get("/rebuild_vectorstore")
async def rebuild_vectorstore():
    """Force rebuild the vectorstore with detailed logging"""
    try:
        import shutil
        global vectorstore, metadata_map, original_dataframes
        
        # Remove existing vectorstore
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)
            logger.info("Existing vectorstore deleted")
        
        # Clear caches
        query_embedding_cache.clear()
        field_mapping_cache.clear()
        
        # Rebuild with detailed logging
        logger.info("Starting vectorstore rebuild...")
        docs = load_excel_files("data")
        
        if not docs:
            return JSONResponse({"error": "No documents loaded. Check your data folder and files."})
        
        vectorstore = build_vector_store(docs)
        
        return JSONResponse({
            "message": "Vectorstore rebuilt successfully",
            "total_documents": len(vectorstore.index_to_docstore_id),
            "files_processed": list(metadata_map.keys()),
            "dataframes_loaded": len(original_dataframes),
            "metadata_map": metadata_map
        })
    except Exception as e:
        logger.error(f"Error rebuilding vectorstore: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {
        "message": "Excel Query API with AI Result Filtering",
        "features": [
            "Direct DataFrame querying for comprehensive results",
            "AI-powered result filtering for accuracy",
            "Smart field mapping with AI",
            "Post-processing quality control",
            "Removes false positives automatically"
        ],
        "endpoints": [
            "/ask - Direct query + AI filtering (recommended)",
            "/ask_raw - Direct query without AI filtering (for comparison)",
            "/metadata - View available columns",
            "/debug_dataframes - View loaded DataFrames",
            "/rebuild_vectorstore - Force rebuild"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
