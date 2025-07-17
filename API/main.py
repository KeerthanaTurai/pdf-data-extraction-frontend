from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine, text
from typing import List, Dict, Any
from fastapi import UploadFile, File, Form, Request
import os
import bcrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ML Processing Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class LoginPayload(BaseModel):
    username: str
    password: str

class InputPayload(BaseModel):
    file_path: str
    formulary_name: str
    criteria_name: str
    policy_date: str

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    dataframe: Dict[str, Any]  # Changed to be more flexible
    processing_time: float
    input_data: Dict[str, Any]
    
class StoreDataPayload(BaseModel):
    filename: str
    table_name: str = "my_table"
    json_data: List[Dict]

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
        db_name = f"{db_base_name}_{timestamp}.db" if db_type == "sqlite" else db_base_name

        # Create database engine
        if db_type == "postgresql":
            # Replace with your actual connection string
            engine = create_engine("postgresql://postgres:postgres@localhost:5432/pdf_data_tool")
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

UPLOAD_FOLDER = "uploads"
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/pdf_data_tool"
engine = create_engine(DATABASE_URL)

@app.post("/login")
async def login(payload: LoginPayload):
    username = payload.username
    password = payload.password

    with engine.connect() as conn:
        user = conn.execute(
            text("SELECT id, password_hash FROM users WHERE username=:username"),
            {"username": username}
        ).fetchone()

    if user is None:
        return {"success": False, "message": "Invalid username or password."}
    user_id, password_hash = user

    # # If you used bcrypt to hash passwords (recommended!):
    # if not bcrypt.checkpw(password.encode(), password_hash.encode()):
    #     return {"success": False, "message": "Invalid username or password."}

    # If you stored plain text (not recommended), just use:
    if password != password_hash:
        return {"success": False, "message": "Invalid username or password."}

    return {"success": True, "user_id": user_id}

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: int = Form(...)
):
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Store file path and user id in the database
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO file_operations (user_id, file_path, operation, status)
                VALUES (:user_id, :file_path, :operation, :status)
            """),
            {
                "user_id": user_id,
                "file_path": file_location,
                "operation": "upload",
                "status": "pending"
            }
        )

    return {"file_path": file_location, "user_id": user_id}

@app.post("/store-data", response_model=ProcessingResponse)
async def store_data_endpoint(payload: StoreDataPayload):
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

def generate_ml_dataframe(input_data: dict) -> pd.DataFrame:
    """Generate ML-based dataframe using scikit-learn"""
    np.random.seed(42)
    
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create initial dataframe with features
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some contextual columns based on input
    df['formulary'] = input_data.get('formulary_name', 'default')
    df['criteria'] = input_data.get('criteria_name', 'default')
    df['policy_date'] = input_data.get('policy_date', datetime.now().strftime('%Y-%m-%d'))
    df['file_source'] = input_data.get('file_path', 'unknown')
    
    # Train a simple model and add predictions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Add predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    df['prediction'] = predictions
    df['probability_class_0'] = probabilities[:, 0]
    df['probability_class_1'] = probabilities[:, 1]
    
    # Add some derived metrics
    df['confidence_score'] = np.max(probabilities, axis=1)
    df['prediction_match'] = (df['target'] == df['prediction']).astype(int)
    df['record_id'] = range(1, len(df) + 1)
    
    logger.info(f"Generated ML dataframe with {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Model accuracy: {accuracy_score(y, predictions):.3f}")
    
    return df

@app.post("/process", response_model=ProcessingResponse)
async def process_data(input_payload: InputPayload):
    """
    Process input payload and return ML-generated dataframe after 10 seconds
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing request with payload: {input_payload.dict()}")
        
        # Hold for 10 seconds
        logger.info("Starting 10-second processing delay...")
        await asyncio.sleep(10)
        
        # Generate ML dataframe
        df = generate_ml_dataframe(input_payload.dict())
        
        # Convert dataframe to dictionary for JSON response
        df_dict = {
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "mean_confidence": float(df['confidence_score'].mean()),
                "accuracy": float(df['prediction_match'].mean()),
                "unique_formularies": int(df['formulary'].nunique()),
                "unique_criteria": int(df['criteria'].nunique())
            }
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return ProcessingResponse(
            success=True,
            message="Data processed successfully with ML model",
            dataframe=df_dict,
            processing_time=processing_time,
            input_data=input_payload.dict()
        )
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during data processing: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Processing Service",
        "version": "1.0.0",
        "endpoints": {
            "process": "/process",
            "store_data": "/store-data",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)