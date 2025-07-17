import pandas as pd
import numpy as np
import asyncio
import logging
import io
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sqlalchemy import create_engine, text, inspect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Database and ML Processing Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DatabaseConfig(BaseModel):
    db_type: str = "sqlite"
    db_name: str = "example.db"
    table_name: str = "my_table"
    if_exists: str = "replace"

class TableInfo(BaseModel):
    table_name: str
    row_count: int

class InputPayload(BaseModel):
    file_path: str
    formulary_name: str
    criteria_name: str
    policy_date: str

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    dataframe: Dict[str, Any]
    processing_time: float
    input_data: Dict[str, Any]
    
class StoreDataPayload(BaseModel):
    filename: str
    table_name: str = "my_table"
    json_data: List[Dict]

# Database utility functions
def get_engine(db_type: str, db_name: str):
    """Create and return a database engine based on the database type."""
    if db_type == "postgresql":
        # Replace with your actual connection string
        return create_engine("postgresql://postgres:postgres@localhost:5432/pdf_data_tool")
    else:
        raise ValueError("Unsupported database type. Use 'sqlite' or 'postgresql'.")

def save_dataframe_to_db(df: pd.DataFrame, db_type: str = "sqlite", db_name: str = "example.db", 
                        table_name: str = "my_table", if_exists: str = "replace"):
    """
    Save a DataFrame to an SQL database using SQLAlchemy.
    
    Parameters:
    - df: pandas DataFrame to save
    - db_type: type of DB ('sqlite', 'postgresql', etc.)
    - db_name: database name or connection URL
    - table_name: name of the SQL table
    - if_exists: behavior if table exists ('replace', 'append', 'fail')
    """
    try:
        engine = get_engine(db_type, db_name)
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        return f"✅ Data saved to '{table_name}' in {db_type} database."
    except Exception as e:
        raise Exception(f"Error saving DataFrame to database: {str(e)}")

def save_json_to_db(json_data, db_type="sqlite", db_base_name="example", table_name="my_table", if_exists="replace"):
    """
    Convert JSON to DataFrame and save to an SQL database with a timestamped filename.

    Parameters:
    - json_data: JSON input (list of dicts or dict format)
    - db_type: type of DB ('sqlite' or 'postgresql')
    - db_base_name: base name of DB file (timestamp will be appended)
    - table_name: name of the SQL table
    - if_exists: behavior if table exists ('replace', 'append', 'fail')
    
    Returns:
    - dict: Information about the saved data
    """
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame(json_data)

        # Create timestamped DB filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = f"{db_base_name}.db" if db_type == "sqlite" else db_base_name

        # Create database engine
        if db_type == "sqlite":
            engine = create_engine(f"sqlite:///{db_name}")
        elif db_type == "postgresql":
            # Replace with your actual connection string
            engine = create_engine(f"postgresql://username:password@localhost:5432/{db_name}")
        else:
            raise ValueError("Unsupported database type. Use 'sqlite' or 'postgresql'.")

        # Save the DataFrame
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        logger.info(f"✅ Data saved to '{table_name}' in {db_type} DB: {db_name}")
        
        return {
            "database_name": db_name,
            "table_name": table_name,
            "rows_saved": len(df),
            "columns_saved": len(df.columns),
            "columns": df.columns.tolist(),
            "data_sample": df.head().to_dict(orient="records") if len(df) > 0 else []
        }
    except Exception as e:
        logger.error(f"❌ Error in save_json_to_db: {e}")
        raise

def retrieve_table_from_db(db_type: str = "sqlite", db_name: str = "demo.db", 
                          table_name: str = "my_table", limit: Optional[int] = None):
    """
    Retrieve a table from the SQL database as a DataFrame.
    
    Parameters:
    - db_type: type of DB ('sqlite', 'postgresql', etc.)
    - db_name: database name or connection URL
    - table_name: name of the SQL table
    - limit: optional limit on number of rows to retrieve
    
    Returns:
    - pandas DataFrame containing the table data
    """
    try:
        engine = get_engine(db_type, db_name)
        
        if limit:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
        else:
            query = f"SELECT * FROM {table_name}"
            
        df = pd.read_sql(query, con=engine)
        return df
    except Exception as e:
        raise Exception(f"Error retrieving table from database: {str(e)}")

def list_tables_with_count(db_type: str = "sqlite", db_name: str = "example.db"):
    """
    List all tables in the database along with their row counts.
    
    Parameters:
    - db_type: type of DB ('sqlite', 'postgresql', etc.)
    - db_name: database name or connection URL
    
    Returns:
    - List of dictionaries containing table names and row counts
    """
    try:
        engine = get_engine(db_type, db_name)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        tables_info = []
        
        with engine.connect() as conn:
            for table_name in table_names:
                # Get row count for each table
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()
                
                tables_info.append({
                    "table_name": table_name,
                    "row_count": row_count
                })
        
        return tables_info
    except Exception as e:
        raise Exception(f"Error listing tables: {str(e)}")

def generate_pharmaceutical_dataframe(input_data: dict) -> pd.DataFrame:
    """Generate pharmaceutical dataframe with drug information"""
    
    # Pharmaceutical data
    with open("formulary.json", "r") as file:
        data = json.load(file)
    
    # Create DataFrame
    df = pd.DataFrame(data["data"])
    
    # Add Document Type column
    df['Document Type'] = "Formulary"
    
    # Add contextual columns based on input
    df['formulary'] = input_data.get('formulary_name', 'default')
    df['criteria'] = input_data.get('criteria_name', 'default')
    df['policy_date'] = input_data.get('policy_date', datetime.now().strftime('%Y-%m-%d'))
    df['file_source'] = input_data.get('file_path', 'unknown')
    df['record_id'] = range(1, len(df) + 1)
    
    # Add some analytical columns
    df['has_quantity_limit'] = df['Requirements/Limit'].str.contains('QL', na=False)
    df['has_prior_auth'] = df['Requirements/Limit'].str.contains('PA', na=False)
    df['tier_level'] = df['Status'].str.extract(r'T(\d+)').astype(int)
    
    logger.info(f"Generated pharmaceutical dataframe with {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Status distribution: {df['Status'].value_counts().to_dict()}")
    
    return df

# FastAPI Endpoints

@app.get("/")
async def root():
    return {
        "message": "Database and ML Processing Service",
        "version": "1.0.0",
        "endpoints": {
            "upload_retrieval": "/upload-retrieval",
            "store_data": "/store-data",
            "save_data": "/save-data",
            "retrieve_table": "/retrieve-table",
            "list_tables": "/list-tables",
            "table_info": "/table-info/{table_name}",
            "health": "/health"
        }
    }

@app.post("/upload-retrieval", response_model=ProcessingResponse)
async def upload_retrieval(input_payload: InputPayload):
    """
    Upload retrieval process - returns pharmaceutical dataframe after 10 seconds
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing upload retrieval request with payload: {input_payload.dict()}")
        
        # Hold for 10 seconds
        logger.info("Starting 10-second upload retrieval delay...")
        await asyncio.sleep(10)
        
        # Generate pharmaceutical dataframe
        df = generate_pharmaceutical_dataframe(input_payload.dict())
        
        # Convert dataframe to dictionary for JSON response
        df_dict = {
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "status_distribution": df['Status'].value_counts().to_dict(),
                "tier_distribution": df['tier_level'].value_counts().to_dict(),
                "drugs_with_quantity_limits": int(df['has_quantity_limit'].sum()),
                "drugs_with_prior_auth": int(df['has_prior_auth'].sum()),
                "unique_formularies": int(df['formulary'].nunique()),
                "unique_criteria": int(df['criteria'].nunique())
            }
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Upload retrieval completed in {processing_time:.2f} seconds")
        
        return ProcessingResponse(
            success=True,
            message="Pharmaceutical data retrieved successfully",
            dataframe=df_dict,
            processing_time=processing_time,
            input_data=input_payload.dict()
        )
        
    except Exception as e:
        logger.error(f"Error in upload retrieval: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during upload retrieval: {str(e)}"
        )

@app.post("/store-data", response_model=ProcessingResponse)
async def store_data_endpoint(payload: StoreDataPayload):
    """Store JSON data to database with timestamped filename"""
    start_time = datetime.now()
    
    try:
        # Validate that json_data is not empty
        if not payload.json_data:
            raise HTTPException(status_code=400, detail="json_data cannot be empty")
            
        # Save data to database and get info
        db_info = save_json_to_db(
            json_data=payload.json_data,
            db_base_name=payload.filename,
            table_name=payload.table_name
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create proper dataframe dict structure
        dataframe_dict = {
            "data": payload.json_data,
            "database_info": db_info,
            "shape": [db_info["rows_saved"], db_info["columns_saved"]],
            "columns": db_info["columns"],
            "summary": {
                "total_rows": db_info["rows_saved"],
                "total_columns": db_info["columns_saved"],
                "table_name": db_info["table_name"],
                "database_name": db_info["database_name"]
            }
        }

        return ProcessingResponse(
            success=True,
            message=f"Successfully stored {db_info['rows_saved']} rows to '{payload.table_name}' table",
            dataframe=dataframe_dict,
            processing_time=processing_time,
            input_data={
                "filename": payload.filename,
                "table_name": payload.table_name,
                "data_count": len(payload.json_data)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error storing data: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/save-data")
async def save_data_to_db_endpoint(
    data: List[Dict[str, Any]],
    config: DatabaseConfig
):
    """Save JSON data to the database."""
    try:
        df = pd.DataFrame(data)
        message = save_dataframe_to_db(
            df, 
            config.db_type, 
            config.db_name, 
            config.table_name, 
            config.if_exists
        )
        
        return {
            "message": message,
            "rows_inserted": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retrieve-table")
async def retrieve_table(
    db_type: str = "sqlite",
    db_name: str = "example.db",
    table_name: str = "my_table",
    limit: Optional[int] = None
):
    """Retrieve table data from the database."""
    try:
        df = retrieve_table_from_db(db_type, db_name, table_name, limit)
        
        # Convert DataFrame to JSON-serializable format
        data = df.to_dict(orient='records')
        
        return {
            "table_name": table_name,
            "row_count": len(df),
            "columns": list(df.columns),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-tables")
async def list_tables(
    db_type: str = "sqlite",
    db_name: str = "example.db"
):
    """List all tables in the database with their row counts."""
    try:
        tables_info = list_tables_with_count(db_type, db_name)
        
        return {
            "database": db_name,
            "database_type": db_type,
            "total_tables": len(tables_info),
            "tables": tables_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/table-info/{table_name}")
async def get_table_info(
    table_name: str,
    db_type: str = "sqlite",
    db_name: str = "example.db"
):
    """Get detailed information about a specific table."""
    try:
        engine = get_engine(db_type, db_name)
        inspector = inspect(engine)
        
        # Check if table exists
        if table_name not in inspector.get_table_names():
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        # Get column information
        columns = inspector.get_columns(table_name)
        
        # Get row count
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()
        
        return {
            "table_name": table_name,
            "row_count": row_count,
            "columns": [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"]
                }
                for col in columns
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Example usage function (can be called programmatically)
def example_usage():
    """Example of how to use the functions programmatically."""
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save to database
    save_dataframe_to_db(df, db_type="sqlite", db_name="demo.db", table_name="users")
    
    # Retrieve from database
    retrieved_df = retrieve_table_from_db(db_type="sqlite", db_name="demo.db", table_name="users")
    print("Retrieved data:")
    print(retrieved_df.head())
    
    # List all tables
    tables = list_tables_with_count(db_type="sqlite", db_name="demo.db")
    print("\nTables in database:")
    for table in tables:
        print(f"- {table['table_name']}: {table['row_count']} rows")

if __name__ == "__main__":
    import uvicorn
    
    # Run example usage
    print("Running example usage...")
    example_usage()
    
    # Start the FastAPI server
    print("\nStarting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)