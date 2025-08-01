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
API_KEY = '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'

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
    global metadata_map, original_dataframes
    metadata_map.clear()
    original_dataframes.clear()

    # Smaller chunks for better granularity
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

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
            original_dataframes[file] = {}
            logger.info(f"Sheets in {file}: {xls.sheet_names}")
            
            for sheet in xls.sheet_names:
                try:
                    logger.info(f"Processing sheet: {sheet}")
                    df = pd.read_excel(path, sheet_name=sheet)
                    
                    # Clean column names and handle NaN values
                    df.columns = df.columns.astype(str).str.strip()
                    df = df.fillna('')
                    
                    # Store original DataFrame
                    original_dataframes[file][sheet] = df.copy()
                    
                    # Store metadata properly
                    metadata_map[file][sheet] = list(df.columns)
                    logger.info(f"Columns in {sheet}: {list(df.columns)}")
                    
                    # Convert to CSV for vectorization
                    csv_text = df.to_csv(index=False, encoding='utf-8')
                    
                    # Create smaller, more focused chunks
                    chunks = splitter.split_text(csv_text)
                    
                    for i, chunk in enumerate(chunks):
                        # Enhanced metadata with column information
                        doc_metadata = {
                            "file": file,
                            "sheet": sheet,
                            "columns": list(df.columns),
                            "chunk_id": f"{file}_{sheet}_{i}",
                            "total_rows": len(df),
                            "columns_text": " | ".join(df.columns)
                        }
                        
                        # Enhanced content with column headers and better structure
                        enhanced_content = f"""
File: {file}
Sheet: {sheet}
Columns: {' | '.join(df.columns)}
Data Type: Excel Sheet Content
Searchable Fields: {' '.join(df.columns)}

{chunk}
"""
                        
                        doc = Document(
                            page_content=enhanced_content,
                            metadata=doc_metadata
                        )
                        docs.append(doc)
                    
                    logger.info(f"Processed {file} - {sheet}: {len(df)} rows, {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing {file} - {sheet}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error opening {file}: {e}")
            continue
    
    logger.info(f"Total documents created: {len(docs)}")
    logger.info(f"Original DataFrames stored: {len(original_dataframes)}")
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

# ---- Enhanced AI Consolidation Function ----
def ai_consolidate_records(filtered_results, query):
    """
    Uses AI to intelligently consolidate multiple records that belong to the same entity
    Works for any type of data - employees, products, customers, etc.
    """
    if not filtered_results or len(filtered_results) <= 1:
        return filtered_results
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)
    
    # Prepare sample data for AI analysis (limit to avoid token limits)
    sample_results = filtered_results[:20] if len(filtered_results) > 20 else filtered_results
    
    # Create simplified version for AI analysis
    simplified_results = []
    for i, result in enumerate(sample_results):
        simplified = {
            "record_index": i,
            "source_file": result.get("source_file", ""),
            "source_sheet": result.get("source_sheet", ""),
            "fields": {}
        }
        
        all_fields = result.get("all_fields", {})
        # Include all fields but limit value length
        for key, value in all_fields.items():
            str_value = str(value)[:50] if value else ""
            simplified["fields"][key] = str_value
        
        simplified_results.append(simplified)
    
    results_json = json.dumps(simplified_results, indent=2)
    
    prompt = f"""
You are a data consolidation expert. Given a user query and filtered results, identify which records belong to the same entity and should be consolidated.

User Query: "{query}"

Records to analyze:
{results_json}

Instructions:
1. Analyze the records to identify which ones represent the same entity (person, product, customer, etc.)
2. Look for common identifiers like IDs, codes, names, emails, or other unique fields
3. Group records that clearly belong to the same entity
4. For each group, suggest the best unique identifier field to use as the key
5. Consider that records from different files/sheets might have slightly different field names for the same data

Return a JSON object with:
1. "consolidation_groups": Array of groups, where each group contains record indices that should be merged
2. "identifier_field": The field name that best identifies each unique entity
3. "should_consolidate": Boolean indicating if consolidation is needed
4. "consolidation_strategy": Brief explanation of the consolidation approach

Example response:
{{
  "consolidation_groups": [
    {{
      "group_id": "entity_1",
      "record_indices": [0, 1, 3, 5],
      "primary_identifier": "Employee Code",
      "identifier_value": "DTDL24"
    }},
    {{
      "group_id": "entity_2", 
      "record_indices": [2, 4],
      "primary_identifier": "Email",
      "identifier_value": "employee25@company.com"
    }}
  ],
  "identifier_field": "Employee Code",
  "should_consolidate": true,
  "consolidation_strategy": "Group records by Employee Code as they represent the same employees from different data sources"
}}

If no consolidation is needed, return:
{{
  "consolidation_groups": [],
  "identifier_field": "",
  "should_consolidate": false,
  "consolidation_strategy": "Records appear to be for different entities"
}}

CRITICAL: Return ONLY the JSON object, no other text.
"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        logger.info(f"AI consolidation analysis: {response}")
        
        # Parse the response
        match = re.search(r'\{[\s\S]*\}', response)
        if not match:
            logger.info("No consolidation needed - returning original results")
            return filtered_results
            
        consolidation_plan = json.loads(match.group(0))
        
        if not consolidation_plan.get("should_consolidate", False):
            logger.info("AI determined no consolidation needed")
            return filtered_results
        
        # Perform the consolidation based on AI analysis
        consolidated_results = []
        processed_indices = set()
        
        for group in consolidation_plan.get("consolidation_groups", []):
            record_indices = group.get("record_indices", [])
            group_id = group.get("group_id", "unknown")
            primary_identifier = group.get("primary_identifier", "")
            identifier_value = group.get("identifier_value", "")
            
            if not record_indices:
                continue
                
            # Consolidate records in this group
            consolidated_record = {
                "entity_identifier": identifier_value or group_id,
                "primary_identifier_field": primary_identifier,
                "consolidated_from_records": len(record_indices),
                "source_files": [],
                "source_sheets": [],
                "consolidated_data": {}
            }
            
            # Merge all records in the group
            for idx in record_indices:
                if idx >= len(filtered_results) or idx in processed_indices:
                    continue
                    
                processed_indices.add(idx)
                result = filtered_results[idx]
                
                # Track sources
                source_file = result.get("source_file", "Unknown")
                source_sheet = result.get("source_sheet", "Unknown")
                
                if source_file not in consolidated_record["source_files"]:
                    consolidated_record["source_files"].append(source_file)
                if source_sheet not in consolidated_record["source_sheets"]:
                    consolidated_record["source_sheets"].append(source_sheet)
                
                # Merge fields with intelligent prefixing
                all_fields = result.get("all_fields", {})
                
                for field_name, field_value in all_fields.items():
                    if field_value is None or str(field_value).strip() == "":
                        continue
                        
                    # If field already exists with same value, don't duplicate
                    if field_name in consolidated_record["consolidated_data"]:
                        existing_value = str(consolidated_record["consolidated_data"][field_name]).strip()
                        new_value = str(field_value).strip()
                        
                        if existing_value == new_value:
                            continue  # Same value, skip
                        else:
                            # Different values, create prefixed version
                            file_prefix = source_file.replace(".xlsx", "").replace(".xls", "").replace(" ", "_")
                            prefixed_field = f"{file_prefix}_{field_name}"
                            consolidated_record["consolidated_data"][prefixed_field] = field_value
                    else:
                        # First occurrence of this field
                        consolidated_record["consolidated_data"][field_name] = field_value
            
            consolidated_results.append(consolidated_record)
        
        # Add any unprocessed records as-is
        for i, result in enumerate(filtered_results):
            if i not in processed_indices:
                consolidated_results.append({
                    "entity_identifier": f"individual_record_{i}",
                    "primary_identifier_field": "record_index",
                    "consolidated_from_records": 1,
                    "source_files": [result.get("source_file", "Unknown")],
                    "source_sheets": [result.get("source_sheet", "Unknown")],
                    "consolidated_data": result.get("all_fields", {})
                })
        
        logger.info(f"Consolidation completed: {len(filtered_results)} → {len(consolidated_results)} records")
        return consolidated_results
        
    except Exception as e:
        logger.error(f"AI consolidation failed: {e}")
        # Return original results if consolidation fails
        return filtered_results

# ---- NEW: AI Answer Generation Function ----
def generate_natural_answer(query, consolidated_results):
    """
    Generate a natural language answer based on the query and consolidated results
    """
    if not consolidated_results:
        return "No results found for your query."
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)
    
    # Prepare data for answer generation (limit to avoid token limits)
    sample_results = consolidated_results
    
    # Extract key information from consolidated results
    answer_data = []
    for result in sample_results:
        if isinstance(result, dict) and "consolidated_data" in result:
            # This is a consolidated record
            data_fields = result["consolidated_data"]
            entity_id = result.get("entity_identifier", "Unknown")
        else:
            # This is a regular record
            data_fields = {k: v for k, v in result.items() 
                          if k not in ['source_file', 'source_sheet', 'row_index']}
            entity_id = "Record"
        
        # Include the most relevant fields (limit field values to avoid token overflow)
        limited_fields = {}
        for key, value in data_fields.items():
            if value and str(value).strip():
                limited_fields[key] = str(value)[:100] if len(str(value)) > 100 else str(value)
        
        answer_data.append({
            "entity": entity_id,
            "fields": limited_fields
        })
    
    data_json = json.dumps(answer_data, indent=2)
    
    prompt = f"""
You are an intelligent data assistant. Based on the user's query and the found data, provide a direct, natural language answer.

User Query: "{query}"

Found Data:
{data_json}

Instructions for generating the answer:
1. Analyze what the user is specifically asking for
2. Extract the most relevant information from the data to answer their question
3. Provide a direct, concise answer in natural language
4. For list queries (e.g., "list all female employees"), provide comma-separated names or items
5. For specific information queries (e.g., "department of John"), provide just the requested information there can be case where deparment have multiple fileds like Department (Value Stream) and Sub Department (Sub Stream) give all in a
6. For count queries, provide the number
7. For descriptive queries, provide a brief summary
8. If there are multiple results, organize them logically (alphabetically, by relevance, etc.)
9. Keep the answer focused and avoid unnecessary details
10. If no relevant data is found, say so clearly

Examples of good answers:
- Query: "List all female employees" → Answer: "Sarah Johnson, Maria Garcia, Lisa Chen, Amanda Roberts"
- Query: "What is John's department?" → Answer: "Engineering"
- Query: "How many employees are in sales?" → Answer: "12 employees"
- Query: "Who has birthday in July?" → Answer: "Mike Wilson (July 15), Sarah Davis (July 22)"

CRITICAL: 
1. Provide ONLY the direct answer, no explanations about the data structure or methodology.
2. For specific information queries (e.g., "department of John"), provide just the requested information and there can be case where deparment have multiple fileds like Department (Value Stream) and Sub Department (Sub Stream) give all in answer.
"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        logger.info(f"Generated natural answer: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        # Fallback: create a simple answer from the data
        return create_fallback_answer(query, consolidated_results)

def create_fallback_answer(query, consolidated_results):
    """
    Create a simple fallback answer if AI generation fails
    """
    if not consolidated_results:
        return "No results found."
    
    # Simple fallback logic
    query_lower = query.lower()
    
    # For list queries, try to extract names or key identifiers
    if any(word in query_lower for word in ['list', 'all', 'who are', 'show me']):
        names = []
        for result in consolidated_results[:10]:  # Limit to first 10
            if isinstance(result, dict):
                data = result.get("consolidated_data", result)
                # Look for name-like fields
                for field, value in data.items():
                    if field and value and any(name_word in field.lower() for name_word in ['name', 'employee', 'person']):
                        if str(value).strip():
                            names.append(str(value).strip())
                        break
        
        if names:
            return ", ".join(names[:10])  # Limit to 10 names
    
    # For count queries
    if any(word in query_lower for word in ['how many', 'count', 'number of']):
        return f"{len(consolidated_results)} records found"
    
    # Default: return first relevant piece of information
    if consolidated_results:
        first_result = consolidated_results[0]
        if isinstance(first_result, dict):
            data = first_result.get("consolidated_data", first_result)
            # Return first non-empty value
            for field, value in data.items():
                if value and str(value).strip() and field not in ['source_file', 'source_sheet', 'row_index']:
                    return str(value).strip()
    
    return f"Found {len(consolidated_results)} matching records"

# Update the main /ask endpoint to include answer generation
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
        
        # Format the results for consolidation
        structured_data = []
        for result in filtered_results:
            record = {
                "source_file": result["source_file"],
                "source_sheet": result["source_sheet"],
                "all_fields": {k: v for k, v in result.items() 
                             if k not in ['source_file', 'source_sheet', 'row_index']}
            }
            structured_data.append(record)
        
        # Apply AI-powered consolidation
        consolidated_results = ai_consolidate_records(structured_data, query)
        logger.info(f"Consolidation: {len(structured_data)} → {len(consolidated_results)} records")
        
        # Generate natural language answer
        natural_answer = generate_natural_answer(query, consolidated_results)
        logger.info(f"Generated answer: {natural_answer}")
        
        return JSONResponse({
            "query": query,
            "answer": natural_answer,
            "field_mapping": field_mapping,
            "method_used": "direct_query_with_ai_filtering_and_consolidation",
            "structured_data": consolidated_results,
            "total_records_found": len(consolidated_results),
            "success": True,
            "debug_info": {
                "raw_results_before_ai_filter": len(raw_results),
                "filtered_results_after_ai_filter": len(filtered_results),
                "consolidated_results_final": len(consolidated_results),
                "ai_filter_reduction": len(raw_results) - len(filtered_results),
                "consolidation_applied": len(structured_data) != len(consolidated_results),
                "files_searched": list(original_dataframes.keys()),
                "ai_quality_check": True,
                "ai_consolidation": True,
                "answer_generated": True
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
    
@app.get("/chat")
async def chat(query: str = Query(...)):
    """
    Single ChatGPT-like endpoint that can handle any type of query:
    - Excel data queries (uses your existing data)
    - General questions
    - Math problems
    - HR analysis
    - Anything else you can think of
    """
    try:
        logger.info(f"Processing chat query: {query}")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=API_KEY)
        
        # First, determine if this is a data-related query or general query
        classification_prompt = f"""
        Analyze this user query and determine if it's asking about data from Excel files/spreadsheets or if it's a general question.
        
        Query: "{query}"
        
        Available Excel data includes employee information, HR records, and other business data.
        
        Respond with only one word:
        - "DATA" if the query is asking about specific information from Excel files/databases
        - "GENERAL" if it's a general question, math problem, analysis, or anything not requiring Excel data
        
        Examples:
        - "List all employees" → DATA
        - "Who has birthday in July?" → DATA  
        - "What is 2+2?" → GENERAL
        - "Analyze this HR scenario..." → GENERAL
        - "Tell me about remote work" → GENERAL
        """
        
        try:
            classification = llm.invoke(classification_prompt).content.strip().upper()
            logger.info(f"Query classified as: {classification}")
        except:
            classification = "GENERAL"  # Default to general if classification fails
        
        if classification == "DATA" and metadata_map:
            # Handle data queries using your existing logic
            try:
                logger.info("Processing as data query...")
                
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
                
                if raw_results:
                    # Apply AI filtering
                    filtered_results = ai_filter_results(query, raw_results)
                    
                    # Format for consolidation
                    structured_data = []
                    for result in filtered_results:
                        record = {
                            "source_file": result["source_file"],
                            "source_sheet": result["source_sheet"],
                            "all_fields": {k: v for k, v in result.items()
                                         if k not in ['source_file', 'source_sheet', 'row_index']}
                        }
                        structured_data.append(record)
                    
                    # Apply consolidation
                    consolidated_results = ai_consolidate_records(structured_data, query)
                    
                    # Generate natural answer
                    answer = generate_natural_answer(query, consolidated_results)
                    
                    return JSONResponse({
                        "query": query,
                        "answer": answer,
                        "type": "data_query",
                        "data_found": True,
                        "records_count": len(consolidated_results),
                        "model_used": "gpt-4o-mini",
                        "success": True,
                        "timestamp": time.time()
                    })
                else:
                    # No data found, fall back to general response
                    logger.info("No data found, falling back to general response")
                    classification = "GENERAL"
            except Exception as data_error:
                logger.error(f"Data query failed: {data_error}")
                # Fall back to general query
                classification = "GENERAL"
        
        # Handle as general query (or fallback from data query)
        if classification == "GENERAL" or not metadata_map:
            logger.info("Processing as general query...")
            
            general_prompt = f"""
You are an intelligent AI assistant similar to ChatGPT. You can help with a wide variety of tasks including:

- Answering general knowledge questions
- Solving mathematical problems
- Analyzing scenarios (HR, business, personal, etc.)
- Providing explanations and tutorials
- Creative writing and brainstorming
- Technical questions and coding help
- And much more

User Query: "{query}"

Instructions:
1. Understand what the user is asking for
2. Provide a comprehensive, helpful, and accurate response
3. For analytical questions, break down your thinking
4. For math problems, show your work step by step
5. For complex topics, provide structured explanations
6. Be conversational but informative
7. If you need clarification, ask follow-up questions

Please provide a detailed, helpful response to the user's query.
"""

            try:
                response = llm.invoke(general_prompt).content.strip()
                
                return JSONResponse({
                    "query": query,
                    "answer": response,
                    "type": "general_query",
                    "model_used": "gpt-4o-mini",
                    "success": True,
                    "timestamp": time.time()
                })
                
            except Exception as llm_error:
                logger.error(f"OpenAI API error: {llm_error}")
                return JSONResponse({
                    "query": query,
                    "error": f"OpenAI API error: {str(llm_error)}",
                    "success": False
                }, status_code=500)
            
    except Exception as e:
        logger.error(f"Error processing chat query: {e}", exc_info=True)
        return JSONResponse({
            "query": query,
            "error": str(e),
            "success": False
        }, status_code=500)
        
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
        "message": "Excel Query API with AI Result Filtering and Natural Language Answers",
        "features": [
            "Direct DataFrame querying for comprehensive results",
            "AI-powered result filtering for accuracy", 
            "Smart field mapping with AI",
            "Post-processing quality control",
            "Removes false positives automatically",
            "Natural language answer generation"
        ],
        "endpoints": [
            "/ask - Direct query + AI filtering + Natural Answer (recommended)",
            "/ask_raw - Direct query without AI filtering (for comparison)",
            "/metadata - View available columns",
            "/debug_dataframes - View loaded DataFrames",
            "/rebuild_vectorstore - Force rebuild"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)