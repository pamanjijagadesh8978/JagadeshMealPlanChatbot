import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional
import json
from decimal import Decimal, InvalidOperation
from datetime import datetime, date, time
import aiosqlite
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Any, Optional, Dict, List, Literal, Annotated
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from database import get_db_connection_dynamic, DYNAMIC_DB_NAME, MASTER_TABLE_DB_NAME
import logging
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
# ── Singleton embedding model (loaded once at import time) ──────────────────
_model = SentenceTransformer("all-MiniLM-L6-v2")

DB_PATH = "memory.db"
PROFILE_PATH = "profiles.db"

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("meal_logger")


MISTRAL_ENDPOINT_1 = os.getenv("MISTRAL_ENDPOINT_1")
MISTRAL_API_KEY_1 = os.getenv("MISTRAL_API_KEY_1")

llm = ChatMistralAI(endpoint=MISTRAL_ENDPOINT_1, api_key=MISTRAL_API_KEY_1)

# ─────────────────────────────────────────────
# EMBEDDING HELPER
# ─────────────────────────────────────────────

async def get_embedding(text: str) -> np.ndarray:
    """Run sentence embedding in a thread so the event loop stays free."""
    return await asyncio.to_thread(
        lambda: _model.encode(text).astype(np.float32)
    )


# ─────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────
async def update_memory(
    user_id: str,
    text: str,
    is_mealplan: bool,
    conn: Optional[aiosqlite.Connection] = None,
    lock: Optional[asyncio.Lock] = None,
) -> None:
    embedding = await get_embedding(text)
    embedding_bytes = embedding.tobytes()
    created_at = datetime.now(timezone.utc).isoformat()

    sql = """
        INSERT INTO memories (user_id, content, is_mealplan, embedding, created_at)
        VALUES (?, ?, ?, ?, ?)
    """

    params = (
        user_id,
        text,
        int(is_mealplan),
        embedding_bytes,
        created_at,
    )

    async def _write(db: aiosqlite.Connection) -> None:
        if lock:
            async with lock:
                await db.execute(sql, params)
                await db.commit()
        else:
            await db.execute(sql, params)
            await db.commit()

    if conn is not None:
        await _write(conn)
    else:
        async with aiosqlite.connect(DB_PATH) as db:
            await _write(db)

# ─────────────────────────────────────────────
# READ – semantic search
# ─────────────────────────────────────────────
async def retrieve_memories(
    user_id: str,
    query: str,
    threshold: float = 0.3,
    max_k: int = 10,
    conn: Optional[aiosqlite.Connection] = None,
) -> list[str]:
    """
    Return the top-*max_k* memories whose cosine similarity to *query* is
    at or above *threshold*.

    Parameters
    ----------
    user_id   : str
    query     : str   – The text to match against stored memories.
    threshold : float – Minimum cosine similarity (0–1).  Default 0.6.
    max_k     : int   – Hard cap on results returned.
    conn      : optional shared connection.
    """
    print(f"retrieve_memories function is called!")
    query_embedding = await get_embedding(query)
    query_norm = np.linalg.norm(query_embedding)

    async def _fetch(db: aiosqlite.Connection) -> list[tuple]:
        cursor = await db.execute(
            "SELECT content, embedding FROM memories WHERE user_id = ?",
            (user_id,),
        )
        return await cursor.fetchall()

    if conn is not None:
        rows = await _fetch(conn)
    else:
        async with aiosqlite.connect(DB_PATH) as db:
            rows = await _fetch(db)

    results: list[tuple[str, float]] = []

    for content, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        emb_norm = np.linalg.norm(emb)

        # Guard against zero-norm vectors (corrupt / empty embeddings)
        if query_norm == 0.0 or emb_norm == 0.0:
            continue

        score = float(np.dot(query_embedding, emb) / (query_norm * emb_norm))

        if score >= threshold:
            results.append((content, score))

    results.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in results[:max_k]]

    print(f"[memory] retrieved {len(top)} relevant memories for query: {query!r}")
    print(f"Relavent memory: {top}")
    return top


# ─────────────────────────────────────────────
# READ – full dump (used by memory extractor)
# ─────────────────────────────────────────────
async def retrieve_all_memories(
    user_id: str,
    conn: Optional[aiosqlite.Connection] = None,
) -> list[str]:
    """Return every stored memory string for *user_id* (no ranking)."""

    async def _fetch(db: aiosqlite.Connection) -> list[str]:
        cursor = await db.execute(
            "SELECT content FROM memories WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    if conn is not None:
        return await _fetch(conn)

    async with aiosqlite.connect(DB_PATH) as db:
        return await _fetch(db)
    
async def retrieve_user_profile_sql_lite(
        user_id: str,
        conn: Optional[aiosqlite.Connection] = None
) -> list[str]:
    print(f"retrieve_user_profile_sql_lite is called")
    async def _fetch(db: aiosqlite.Connection) -> list[str]:
        cursor = await db.execute(
            "SELECT * FROM UserHealthProfile WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        profile = [dict(zip([col[0] for col in cursor.description], row)) for row in rows]
        # print(profile)
        return profile
    
    if conn is not None:
        return await _fetch(conn)
    
    async with aiosqlite.connect(PROFILE_PATH) as db:
        return await _fetch(db)

def clean_profile(profile: dict, user_id: str) -> dict:
    """
    Clean and normalize a raw user profile dict into a flat structure
    ready for SQLite insertion into UserHealthProfile.

    Args:
        profile: Raw profile dict returned by get_user_profile()
        user_id: The user's ID to include in the cleaned record

    Returns:
        A flat dict whose keys map 1-to-1 with UserHealthProfile columns
    """

    def join_list(value) -> str:
        """Safely join a list into a comma-separated string."""
        if not value:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v and str(v).strip().lower() != "none")
        return str(value)

    def safe_float(value) -> float | None:
        """Convert a value to float, return None if not possible."""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def safe_int(value) -> int | None:
        """Convert a value to int, return None if not possible."""
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def clean_str(value) -> str | None:
        """Strip and return a string, None if empty or null-like."""
        if not value:
            return None
        cleaned = str(value).strip()
        return None if cleaned.lower() in ("none", "null", "") else cleaned

    vitals = profile.get("vitals_numeric") or {}

    return {
        # ── Identity ───────────────────────────────────────────────────────
        "user_id":                    clean_str(user_id),

        # ── Demographics ───────────────────────────────────────────────────
        "name":                       clean_str(profile.get("name")),
        "age":                        safe_int(profile.get("age")),
        "gender":                     clean_str(profile.get("gender")),

        # ── Body Metrics ───────────────────────────────────────────────────
        # ✅ Fix 1: keys renamed to match schema columns (lbs / inches)
        "weight_kg":                 safe_float(profile.get("weight_kg")),
        "height_cm":                 safe_float(profile.get("height_cm")),
        "waist_circumference_cm":    safe_float(profile.get("waist_circumference_cm")),

        # ── Food Preferences ───────────────────────────────────────────────
        "cuisine":                    clean_str(profile.get("cuisine")),
        "activity_level":             clean_str(profile.get("activity_level")),
        # ✅ Fix 3: calories added (was missing, now included in INSERT)
        "calories":                   clean_str(profile.get("calories")),
        "dietary_preference":         join_list(profile.get("dietary_preference")),
        "restrictions":               join_list(profile.get("restrictions")),
        "digestive_issues":           join_list(profile.get("digestive_issues")),
        "allergies":                  join_list(profile.get("allergies")),
        "symptom_aggravating_foods":  join_list(profile.get("symptom_aggravating_foods")),

        # ── Vitals ─────────────────────────────────────────────────────────
        "heart_rate":                 safe_float(vitals.get("Heart Rate")),
        "blood_pressure":             clean_str(vitals.get("Blood Pressure")),
        "body_temperature":           safe_float(vitals.get("Body Temperature")),
        # ✅ Fix 2: blood_oxygen and respiratory_rate were missing
        "blood_oxygen":               safe_float(vitals.get("Blood Oxygen")),
        "respiratory_rate":           safe_float(vitals.get("Respiratory Rate")),

        # ── Medical ────────────────────────────────────────────────────────
        "medical_conditions":         join_list(profile.get("medical_conditions")),
        "goals":                      join_list(profile.get("goals")),
    }

async def fetch_user_id_sql_lite(
    token: str,
    conn: Optional[aiosqlite.Connection] = None
) -> str | None:                              # ← changed return type
    async def _fetch(db: aiosqlite.Connection) -> str | None:
        cursor = await db.execute(
            "SELECT user_id FROM users WHERE api_token = ?", (token,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row[0] if row else None        # ← return scalar, not list

    if conn is not None:
        return await _fetch(conn)
    async with aiosqlite.connect(DB_PATH) as db:
        return await _fetch(db)

def fetch_user_id(conn, token: str):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT TOP 1 UserId
            FROM UserDeviceToken
            WHERE DeviceToken = ?
        """, (token,))
        
        row = cursor.fetchone()
        return row[0] if row else None

    finally:
        cursor.close()

async def get_user_id_sql_server(token: str):
    conn = await get_db_connection_dynamic(DYNAMIC_DB_NAME)
    result = await asyncio.to_thread(fetch_user_id, conn, token)
    return result

def averagevitals(vitals_list):
    vitals_raw = vitals_list.get("Vitals", {})

    # Step 1: Decode JSON if needed
    if isinstance(vitals_raw, str):
        vitals = json.loads(vitals_raw)
    else:
        vitals = vitals_raw

    raw_values = {}

    for vital_name, vital_data in vitals.items():
        if not isinstance(vital_data, dict):
            continue

        for field in [
            "HeartRateValue",
            "BloodGlucoseValue",
            "Systolic",
            "Diastolic",
            "RespiratoryRateValue",
            "BloodOxygenValue",
            "BodyTempratureValue"
        ]:
            if field in vital_data and vital_data[field] is not None:
                try:
                    raw_values[field] = float(vital_data[field])
                except (ValueError, TypeError):
                    pass

    # Step 2: Human-readable output
    readable_vitals = {}

    if "HeartRateValue" in raw_values:
        readable_vitals["Heart Rate"] = raw_values["HeartRateValue"]

    if "BloodGlucoseValue" in raw_values:
        readable_vitals["Blood Glucose"] = raw_values["BloodGlucoseValue"]

    if "Systolic" in raw_values and "Diastolic" in raw_values:
        readable_vitals["Blood Pressure"] = {
            "systolic": raw_values["Systolic"],
            "diastolic": raw_values["Diastolic"]
        }

    if "RespiratoryRateValue" in raw_values:
        readable_vitals["Respiration Rate"] = raw_values["RespiratoryRateValue"]

    if "BloodOxygenValue" in raw_values:
        readable_vitals["Blood Oxygen Saturation"] = raw_values["BloodOxygenValue"]

    if "BodyTempratureValue" in raw_values:
        readable_vitals["Body Temperature"] = raw_values["BodyTempratureValue"]

    return readable_vitals

async def fetch_vitals(user_id: str, cursor: str, current_date: str):

    if cursor is None:
        con = await get_db_connection_dynamic(DYNAMIC_DB_NAME)
        cursor = con.cursor()

    cursor.execute("EXEC [Member].[GetUserNutritionDashboardVitals] @UserId = ?, @CurrentDate = ?, @TimeZoneOffset = ?, @OffsetMinutes = ?", user_id, current_date, 330, -330)
    vitals = dict(zip([col[0] for col in cursor.description], cursor.fetchone() or []))
    
    return vitals

async def get_patient_active_diagnoses(patient_id: str, database_name: str):
    """
    OPTIMIZED: Fetches all diagnoses using only 2 connections total, instead of N+1.
    """
    results = []
    try:
        # 1. Get Codes from Patient DB
        conn_dynamic = await get_db_connection_dynamic(database_name)
        cursor_dynamic = conn_dynamic.cursor()
        cursor_dynamic.execute("""
            SELECT ConditionCode
            FROM [dbo].[PatientDiagnosis]
            WHERE PatientId = ? AND IsActive = 1 AND Deleted = 0
        """, patient_id)
        rows = cursor_dynamic.fetchall()
        cursor_dynamic.close()
        conn_dynamic.close()

        if not rows:
            return []

        # Extract codes list
        condition_codes = [row[0] for row in rows]

        # 2. Get Names from Master DB (Single Batch Query)
        conn_master = await get_db_connection_dynamic(MASTER_TABLE_DB_NAME)
        cursor_master = conn_master.cursor()
        
        # Create parameter placeholders (?, ?, ?)
        placeholders = ','.join('?' for _ in condition_codes)
        query = f"""
            SELECT MedicalConditionName
            FROM [Master].[HealthDiagnoses]
            WHERE ConditionCode IN ({placeholders})
        """
        
        cursor_master.execute(query, condition_codes)
        master_rows = cursor_master.fetchall()
        
        results = [row[0] for row in master_rows]
        
        cursor_master.close()
        conn_master.close()
        
    except Exception as e:
        print(f"Error getting diagnoses: {e}")
        # Return empty list on error to prevent crashing the whole profile fetch
        return []

    return results

def lbs_to_kg(lbs):
        return round(float(lbs) * 0.453592, 2)

def parse_weight(weight_str):
        try:
            weight_data = json.loads(weight_str)[0]
            return lbs_to_kg(weight_data["WeightMeasurement"])
        except:
            return None

def parse_waist(waist_str):
    try:
        return round(float(waist_str) * 2.54, 2)
    except (json.JSONDecodeError, KeyError, ValueError, IndexError, TypeError):
        return None
            
def extract_descriptions(json_str):
        try:
            return [item["Description"] for item in json.loads(json_str)] if json_str else []
        except:
            return []

def parse_height(height_str):
    try:
        # Convert the height value (string) to float inches
        height_value_str = json.loads(height_str)[0]["HeightMeasurement"]
        inches = float(height_value_str)
        height_cm = round(inches * 2.54, 2)  # 1 inch = 2.54 cm
        return height_cm
    except (ValueError, TypeError):
        return None
    
def convert_json_to_profile(json_profile: dict, vitals_json: dict = None, diagnoses: list = None) -> dict:
    # ✅ Guard clause
    if (
        not isinstance(json_profile, dict)
        or json_profile.get("status") not in (None, 200)
        or json_profile.get("data") is None and "FirstName" not in json_profile
    ):
        return {
            "error": True,
            "message": json_profile.get("message", "Empty profile data"),
            "profile": None
        }
    
    name = f'{json_profile.get("FirstName", "")} {json_profile.get("LastName", "")}'.strip()
    age = json_profile.get("Age")
    gender_map = {2: "male", 3: "female", 4: "other"}
    gender = gender_map.get(json_profile.get("Gender"), "unknown")

    weight = parse_weight(json_profile.get("LatestWeight", "[]"))
    height_cm = parse_height(json_profile.get("LatestHeight", "[]"))
    waist_cm = parse_waist(json_profile.get("WaistMeasurement", "[]"))

    cuisine = json_profile.get("CuisineDes") or json_profile.get("CursineDes", "")
    activity_level = json_profile.get("PhysicalActivityDesc", "unknown")

    dietary_preference = extract_descriptions(json_profile.get("DietaryPreferences"))
    dietary_restrictions = extract_descriptions(json_profile.get("DietaryRestrictions"))
    digestive_issues = [d for d in extract_descriptions(json_profile.get("DigestiveIssues")) if d != "None"]
    food_allergies = [a for a in extract_descriptions(json_profile.get("FoodAllergies")) if a != "None"]
    symptom_aggravating_foods = [a for a in extract_descriptions(json_profile.get("SymptomAggravatingFoods")) if a != "None"]

    profile_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "weight_kg": weight,
        "height_cm": height_cm,
        "waist_circumference_cm": waist_cm,
        "cuisine": cuisine,
        "activity_level": activity_level,
        "dietary_preference": dietary_preference,
        "restrictions": dietary_restrictions,
        "digestive_issues": digestive_issues,
        "allergies": food_allergies,
        "symptom_aggravating_foods": symptom_aggravating_foods
    }

    # --- Fitness onboarding profile fields ---
    fitness_fields = [
        "primary_goal",
        "secondary_goal",
        "target_body_parts",
        "fitness_level",
        "physical_limitation",
        "specific_avoidance",
        "days_per_week",
        "session_duration",
        "available_equipment",
        "unit_system",
        "workout_location",
        "goal",
    ]

    for field in fitness_fields:
        if field in json_profile:
            if field == "session_duration":
                profile_data[field] = json_profile[field].replace("–", "-").replace(" min", " minutes")
            elif field == "days_per_week":
                value = json_profile[field]
                profile_data[field] = value if isinstance(value, list) else [value]
            else:
                profile_data[field] = json_profile[field]

    # --- Vitals ---
    if vitals_json and any(vitals_json.values()):
        profile_data["vitals_numeric"] = {}

        if "Heart Rate" in vitals_json:
            profile_data["vitals_numeric"]["Heart Rate"] = vitals_json["Heart Rate"]

        if "Blood Pressure" in vitals_json and isinstance(vitals_json["Blood Pressure"], dict):
            bp = vitals_json["Blood Pressure"]
            if "systolic" in bp and "diastolic" in bp:
                profile_data["vitals_numeric"]["Blood Pressure"] = f"{bp['systolic']}/{bp['diastolic']}"

        for key in ["Blood Glucose", "Respiration Rate", "Blood Oxygen Saturation", "Blood Ketones", "Body Temperature"]:
            if key in vitals_json:
                profile_data["vitals_numeric"][key] = vitals_json[key]

        if not profile_data["vitals_numeric"]:
            del profile_data["vitals_numeric"]

    if diagnoses:
        profile_data["medical_conditions"] = diagnoses

    return profile_data

async def get_user_profile(database_name: str, user_id: str,):
    conn_dynamic = await get_db_connection_dynamic(database_name)
    cursor_dynamic = conn_dynamic.cursor()

    # Resolve PatientId
    # cursor_dynamic.execute("SELECT TOP 1 PatientId FROM dbo.Patient WHERE PatientUserId = ?", user_id)
    # row = cursor_dynamic.fetchone()

    # --- Profile Queries ---
    cursor_dynamic.execute("EXEC [dbo].[usp_GetPatientDietaryProfileBySearch] @UserId = ?", user_id)
    profile_row1 = cursor_dynamic.fetchone()
    profile_columns1 = [col[0] for col in cursor_dynamic.description] if profile_row1 else []
    profile_dict1 = dict(zip(profile_columns1, profile_row1)) if profile_row1 else {}

    
    cursor_dynamic.execute("EXEC [dbo].[sp_GetPatinetInfoByUserId] @PatientId = ?", user_id)
    profile_row2 = cursor_dynamic.fetchone()
    profile_columns2 = [col[0] for col in cursor_dynamic.description] if profile_row2 else []
    profile_dict2 = dict(zip(profile_columns2, profile_row2)) if profile_row2 else {}
    
    
    merged_profile = {**profile_dict2, **profile_dict1}
    
    # --- Fetch Patient Goals ---
    cursor_dynamic.execute(f"""
        SELECT 
            pg.PatientGoalCode, pg.GoalCode, gm.GoalName, gm.GoalDescription, 
            gtm.GoalTypeName, gcm.CategoryName, gsm.StatusName,
            pg.TargetValue, pg.CurrentValue, pg.ProgressPercent,
            pg.StartDate, pg.TargetDate
        FROM {database_name}.[dbo].[PatientGoals] pg
        LEFT JOIN [{MASTER_TABLE_DB_NAME}].[Master].[GoalMaster] gm ON pg.GoalCode = gm.GoalCode
        LEFT JOIN [{MASTER_TABLE_DB_NAME}].[Master].[GoalTypeMaster] gtm ON gm.GoalTypeCode = gtm.GoalTypeCode
        LEFT JOIN [{MASTER_TABLE_DB_NAME}].[Master].[GoalCategoryMaster] gcm ON pg.CategoryCode = gcm.CategoryCode
        LEFT JOIN [{MASTER_TABLE_DB_NAME}].[Master].[GoalStatusMaster] gsm ON pg.StatusCode = gsm.StatusCode
        WHERE pg.PatientId = ?
        ORDER BY pg.CreatedDate DESC
    """, user_id)

    patient_goals = []
    
    now = datetime.now(timezone.utc)
    vitals_avg = averagevitals(await fetch_vitals(user_id, cursor_dynamic, (now - timedelta(days=1)).strftime("%Y-%m-%d")))
    diagnoses = []
    diagnoses = await get_patient_active_diagnoses(user_id, database_name)

    # --- Convert final profile ---
    profile = convert_json_to_profile(merged_profile, vitals_json=vitals_avg, diagnoses=diagnoses)
    profile["goals"] = patient_goals  # ✅ Add the new goals section

    return profile

# --------------------------------------------------
# BELOW: Your meal parsing / insert code (kept as-is, but with minimal fixes)
# --------------------------------------------------
Kcal = Annotated[int, Field(ge=0, description="Calories in kcal (>= 0)")]
Grams = Annotated[float, Field(ge=0, description="Grams (>= 0)")]

class MacroSchema(BaseModel):
    calories: Kcal = Field(description="Calories (kcal)")
    protein_g: Grams = Field(description="Protein in grams")
    carbs_g: Grams = Field(description="Carbohydrates in grams")
    fiber_g: Grams = Field(description="Fiber in grams (does not count toward kcal split)")
    fat_g: Grams = Field(description="Fat in grams")
    saturated_fat_g: Grams = Field(description="Saturated fat in grams (must be <10% of kcal)")

class FoodItemSchema(BaseModel):
    name: str = Field(description="Food name (e.g., 'Greek yogurt')")
    quantity: str = Field(description="Quantity/serving (e.g., '200g', '2 eggs', '1 cup')")
    portion_weight_oz: Annotated[float, Field(ge=0, description="Portion weight in ounces")]
    macros: MacroSchema = Field(description="Macros for THIS quantity")

class MealSchema(BaseModel):
    meal_name: str = Field(description="Meal name/title")
    target_macros: MacroSchema = Field(description="Target macros for the meal")
    foods: List[FoodItemSchema] = Field(description="1–4 foods for the meal with quantities and macros")
    notes: str = Field(default=None, description="Short prep notes/substitutions")

class MealPlanSchema(BaseModel):
    title: str = Field(description="Short title including total calories")
    meal_date: str = Field(description="Date of the meal plan in DD-MM-YYYY format")
    total_day_macros: MacroSchema = Field(description="Total macros for the entire day")
    breakfast: MealSchema
    morning_snack: MealSchema
    lunch: MealSchema
    evening_snack: MealSchema
    dinner: MealSchema
    hydration_tip: str = Field(description="One hydration tip")
    preparation: str = Field(
    description="Preparation instructions for the day (e.g., batch cook lunch/dinner, quick breakfast ideas, etc.)",)
    warnings: Optional[List[str]] = Field(default=None, description="Any warnings/reminders")

def _meal_macro_targets(meal_kcal: int) -> MacroSchema:
    protein_g = round((meal_kcal * 0.25) / 4.0, 1)
    fat_g = round((meal_kcal * 0.35) / 9.0, 1)
    carbs_g = round((meal_kcal * 0.40) / 4.0, 1)

    sat_fat_max = (meal_kcal * 0.10) / 9.0
    saturated_fat_g = round(max(0.0, sat_fat_max * 0.6), 1)

    fiber_g = round(max(4.0, min(14.0, (meal_kcal / 500.0) * 10.0)), 1)

    return MacroSchema(
        calories=int(meal_kcal),
        protein_g=max(0.0, protein_g),
        carbs_g=max(0.0, carbs_g),
        fiber_g=max(0.0, fiber_g),
        fat_g=max(0.0, fat_g),
        saturated_fat_g=max(0.0, saturated_fat_g),
    )

def _sum_macros(macros: List[MacroSchema]) -> MacroSchema:
    return MacroSchema(
        calories=int(round(sum(m.calories for m in macros))),
        protein_g=round(sum(m.protein_g for m in macros), 1),
        carbs_g=round(sum(m.carbs_g for m in macros), 1),
        fiber_g=round(sum(m.fiber_g for m in macros), 1),
        fat_g=round(sum(m.fat_g for m in macros), 1),
        saturated_fat_g=round(sum(m.saturated_fat_g for m in macros), 1),
    )
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class ParallelSevenDayMealPlans(TypedDict):
    day1instructions: str
    day2instructions: str
    day3instructions: str
    day4instructions: str
    day5instructions: str
    day6instructions: str
    day7instructions: str
    day1meal: str
    day2meal: str
    day3meal: str
    day4meal: str
    day5meal: str
    day6meal: str
    day7meal: str
    final7daymealplan: dict

class MealPlanInstructions(BaseModel):
    Day1Instructions: str
    Day2Instructions: str
    Day3Instructions: str
    Day4Instructions: str
    Day5Instructions: str
    Day6Instructions: str
    Day7Instructions: str

async def MealPlanInstructor(
    user_id: str,
    conn: aiosqlite.Connection,
    profile: dict,
    meal_start_date: Optional[str] = None,
    lock: Optional[asyncio.Lock] = None,
) -> MealPlanInstructions:

    async def user_meal_sentiments(conn, user_id) -> list[str]:
        async with conn.execute(
            "SELECT content FROM memories WHERE user_id = ? AND is_mealplan = 1",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    eating_sentiments = await user_meal_sentiments(conn, user_id)
    print(f"Profile :{profile}")
    print(f"Eating Sentiment: {eating_sentiments}")

    prompt = f"""
        You are an expert meal planning advisor. Your job is to generate precise,
        personalized daily meal instructions for a 7-day meal plan generator.
        based on the meal start date include the date in the instructions for each day.
        meal start date is {meal_start_date if meal_start_date else "No need to mention the meal date in the instructions."}
        Make sure date changes across days in the instructions if meal start date is provided.

        Always adapt to the user's profile and eating sentiments.
        Ensure variety — do NOT repeat the same dishes, ingredients, or meal patterns across days.

        Each day must include exactly five meals:
        Breakfast, Mid-Morning Snack, Lunch, Evening Snack, Dinner.

        USER PROFILE:
        {profile}

        USER'S PAST EATING SENTIMENTS & PREFERENCES:
        {eating_sentiments if eating_sentiments else "No previous data available."}

        STRICT RULES:
        - Respect all health conditions (e.g., low sugar for diabetes, low sodium for hypertension).
        - Respect all dietary restrictions and allergies.
        - Ensure HIGH PROTEIN distribution across the day (each main meal must include a protein source).
        - Balance macronutrients (protein, carbs, fats) — avoid carb-heavy plans.
        - Ensure vegetable diversity across the week (different vegetables each day).
        - Avoid repeating:
        - same dishes
        - same side items (e.g., chutneys, snacks)
        - same cooking styles
        - Include cuisine variety when possible (not the same cuisine every day).
        - Keep meals realistic and culturally appropriate.

        OUTPUT FORMAT:
        Return ONLY a valid JSON object in this exact structure:

        {{
        "Day1": {{
            "Date": "DD-MM-YYYY",
            "Breakfast": "...",
            "MidMorningSnack": "...",
            "Lunch": "...",
            "EveningSnack": "...",
            "Dinner": "..."
        }},
        "Day2": {{ ... }},
        ...
        "Day7": {{ ... }}
        }}

        INSTRUCTIONS STYLE:
        - Be concise and directive.
        - Describe WHAT to eat, not HOW to cook.
        - Do NOT include recipes or steps.

        Return ONLY JSON. No explanation, no markdown.
        """

    structured_llm = llm.with_structured_output(MealPlanInstructions)
    instructions: MealPlanInstructions = await structured_llm.ainvoke([HumanMessage(content=prompt)])
    return instructions

async def DayOneMealPlan(state: ParallelSevenDayMealPlans) -> dict:
    structured_llm = llm.with_structured_output(MealPlanSchema)
    result1 = await structured_llm.ainvoke(state["day1instructions"])
    return {"day1meal": result1}
async def DayTwoMealPlan(state: ParallelSevenDayMealPlans) -> dict:
    structured_llm = llm.with_structured_output(MealPlanSchema)
    result2 = await structured_llm.ainvoke(state["day2instructions"])
    return {"day2meal": result2}
async def DayThreeMealPlan(state: ParallelSevenDayMealPlans) -> dict:
    structured_llm = llm.with_structured_output(MealPlanSchema)
    result3 = await structured_llm.ainvoke(state["day3instructions"])
    return {"day3meal": result3}
async def DayFourMealPlan(state: ParallelSevenDayMealPlans) -> dict:
    structured_llm = llm.with_structured_output(MealPlanSchema)
    result4 = await structured_llm.ainvoke(state["day4instructions"])
    return {"day4meal": result4}
async def DayFiveMealPlan(state: ParallelSevenDayMealPlans) -> dict:
    structured_llm = llm.with_structured_output(MealPlanSchema)
    result5 = await structured_llm.ainvoke(state["day5instructions"])
    return {"day5meal": result5}
async def DaySixMealPlan(state: ParallelSevenDayMealPlans) -> dict:
    structured_llm = llm.with_structured_output(MealPlanSchema)
    result6 = await structured_llm.ainvoke(state["day6instructions"])
    return {"day6meal": result6}
async def DaySevenMealPlan(state: ParallelSevenDayMealPlans) -> dict:
    structured_llm = llm.with_structured_output(MealPlanSchema)
    result7 = await structured_llm.ainvoke(state["day7instructions"])
    return {"day7meal": result7}
def FinalMealPlan(state: ParallelSevenDayMealPlans) -> ParallelSevenDayMealPlans:
    return {"final7daymealplan":{
        "day1meal": state["day1meal"],
        "day2meal": state["day2meal"],
        "day3meal": state["day3meal"],
        "day4meal": state["day4meal"],
        "day5meal": state["day5meal"],
        "day6meal": state["day6meal"],
        "day7meal": state["day7meal"]
    }}

graph = StateGraph(ParallelSevenDayMealPlans)
graph.add_node('DayOneMealPlan', DayOneMealPlan)
graph.add_node('DayTwoMealPlan', DayTwoMealPlan)
graph.add_node('DayThreeMealPlan', DayThreeMealPlan)
graph.add_node('DayFourMealPlan', DayFourMealPlan)
graph.add_node('DayFiveMealPlan', DayFiveMealPlan)
graph.add_node('DaySixMealPlan', DaySixMealPlan)
graph.add_node('DaySevenMealPlan', DaySevenMealPlan)
graph.add_node('FinalMealPlan', FinalMealPlan)
#Edges
graph.add_edge(START, 'DayOneMealPlan')
graph.add_edge(START, 'DayTwoMealPlan')
graph.add_edge(START, 'DayThreeMealPlan')
graph.add_edge(START, 'DayFourMealPlan')
graph.add_edge(START, 'DayFiveMealPlan')
graph.add_edge(START, 'DaySixMealPlan')
graph.add_edge(START, 'DaySevenMealPlan')
graph.add_edge('DayOneMealPlan', 'FinalMealPlan')
graph.add_edge('DayTwoMealPlan', 'FinalMealPlan')
graph.add_edge('DayThreeMealPlan', 'FinalMealPlan')
graph.add_edge('DayFourMealPlan', 'FinalMealPlan')
graph.add_edge('DayFiveMealPlan', 'FinalMealPlan')
graph.add_edge('DaySixMealPlan', 'FinalMealPlan')
graph.add_edge('DaySevenMealPlan', 'FinalMealPlan')

graph.add_edge('FinalMealPlan', END)
workflow = graph.compile()

async def generate_meal_plan_json(
    age: int,
    gender: str,
    weight: float,
    height: float,
    activity_level: int,
    calorie_goal: int = 0,
    dietary_restrictions: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None,
    foods_to_avoid: Optional[List[str]] = None,
    chronic_conditions: Optional[List[str]] = None,
    preferred_cuisines: Optional[List[str]] = None,
    meal_plan_style: Optional[str] = None,
    meal_plan_instructions: Optional[str] = None,
    meal_start_date: Optional[str] = None,
) -> MealPlanSchema:
    dietary_restrictions = dietary_restrictions or []
    allergies = allergies or []
    foods_to_avoid = foods_to_avoid or []
    chronic_conditions = chronic_conditions or []
    preferred_cuisines = preferred_cuisines or []

    breakfast_kcal = int(round(calorie_goal * 0.25))
    morning_snack_kcal = int(round(calorie_goal * 0.10))
    lunch_kcal = int(round(calorie_goal * 0.35))
    evening_snack_kcal = int(round(calorie_goal * 0.10))
    dinner_kcal = int(round(calorie_goal * 0.20))

    rounded_total = int(round(calorie_goal))
    drift = rounded_total - (breakfast_kcal + morning_snack_kcal + lunch_kcal + evening_snack_kcal + dinner_kcal)
    dinner_kcal += drift

    targets: Dict[str, MacroSchema] = {
        "breakfast": _meal_macro_targets(breakfast_kcal),
        "morning_snack": _meal_macro_targets(morning_snack_kcal),
        "lunch": _meal_macro_targets(lunch_kcal),
        "evening_snack": _meal_macro_targets(evening_snack_kcal),
        "dinner": _meal_macro_targets(dinner_kcal),
    }
    total_day_targets = _sum_macros(list(targets.values()))

    diabetes_rule = ""
    if any("diabet" in c.lower() for c in chronic_conditions):
        diabetes_rule = "Diabetes: no added sugar/sugary drinks; prefer high-fiber/low-GI carbs; protein+fiber each meal."

    style_line = meal_plan_style or "Any"

    prompt = (
        "Return ONLY valid JSON matching MealPlanSchema.\n"
        f"Date: {meal_start_date or 'Not specified'}\n"
        f"User: age {age}, gender {gender}, {weight}kg, {height}cm, activity {activity_level}/5. "
        f"Day kcal {rounded_total}.\n"
        f"Restrictions: {dietary_restrictions or 'None'}. Allergies (avoid): {allergies or 'None'}. "
        f"Avoid foods: {foods_to_avoid or 'None'}. Conditions: {chronic_conditions or 'None'}.\n"
        f"Cuisines: {preferred_cuisines or 'Any'}. Style: {style_line}.\n"
        f"{(diabetes_rule + ' ') if diabetes_rule else ''}\n"
        f"{(meal_plan_instructions + ' ') if meal_plan_instructions else ''}\n"
        "Meals: breakfast, morning_snack, lunch, evening_snack, dinner.\n"
        "Use these meal ± 50 kcal + macro targets (grams) as target_macros; foods must sum close (±10%). "
        "Total day macros should match sum of meals (±5%). Saturated fat MUST stay under 10% of meal kcal.\n"
        f"Targets:\n"
        f"- Breakfast: kcal {targets['breakfast'].calories}, Protein {targets['breakfast'].protein_g}g, "
        f"Carbs {targets['breakfast'].carbs_g}g, Fat {targets['breakfast'].fat_g}g, "
        f"SatF {targets['breakfast'].saturated_fat_g}g, Fiber {targets['breakfast'].fiber_g}g\n"
        f"- Morning snack: kcal {targets['morning_snack'].calories}, Protein {targets['morning_snack'].protein_g}g, "
        f"Carbs {targets['morning_snack'].carbs_g}g, Fat {targets['morning_snack'].fat_g}g, "
        f"SatF {targets['morning_snack'].saturated_fat_g}g, Fiber {targets['morning_snack'].fiber_g}g\n"
        f"- Lunch: kcal {targets['lunch'].calories}, Protein {targets['lunch'].protein_g}g, "
        f"Carbs {targets['lunch'].carbs_g}g, Fat {targets['lunch'].fat_g}g, "
        f"SatF {targets['lunch'].saturated_fat_g}g, Fiber {targets['lunch'].fiber_g}g\n"
        f"- Evening snack: kcal {targets['evening_snack'].calories}, Protein {targets['evening_snack'].protein_g}g, "
        f"Carbs {targets['evening_snack'].carbs_g}g, Fat {targets['evening_snack'].fat_g}g, "
        f"SatF {targets['evening_snack'].saturated_fat_g}g, Fiber {targets['evening_snack'].fiber_g}g\n"
        f"- Dinner: kcal {targets['dinner'].calories}, Protein {targets['dinner'].protein_g}g, "
        f"Carbs {targets['dinner'].carbs_g}g, Fat {targets['dinner'].fat_g}g, "
        f"SatF {targets['dinner'].saturated_fat_g}g, Fiber {targets['dinner'].fiber_g}g\n"
        "Each meal: 1–4 foods with quantity, portion_weight_oz (number), and macros for that quantity. "
        "No allergens; respect restrictions/avoid list; easy ingredients."
    )

    logger.info("Prompt to LLM\n%s", prompt)
    structured_llm = llm.with_structured_output(MealPlanSchema)
    return await structured_llm.ainvoke(prompt)

def prompt(
        age: int,
        gender: str,
        weight: float,
        height: float,
        activity_level: int,
        calorie_goal: int = 0,
        dietary_restrictions: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        foods_to_avoid: Optional[List[str]] = None,
        chronic_conditions: Optional[List[str]] = None,
        preferred_cuisines: Optional[List[str]] = None,
        meal_plan_style: Optional[str] = None,
        meal_plan_instructions: Optional[str] = None,
) -> str:

    dietary_restrictions = dietary_restrictions or []
    allergies = allergies or []
    foods_to_avoid = foods_to_avoid or []
    chronic_conditions = chronic_conditions or []
    preferred_cuisines = preferred_cuisines or []

    breakfast_kcal = int(round(calorie_goal * 0.25))
    morning_snack_kcal = int(round(calorie_goal * 0.10))
    lunch_kcal = int(round(calorie_goal * 0.35))
    evening_snack_kcal = int(round(calorie_goal * 0.10))
    dinner_kcal = int(round(calorie_goal * 0.20))

    rounded_total = int(round(calorie_goal))
    drift = rounded_total - (breakfast_kcal + morning_snack_kcal + lunch_kcal + evening_snack_kcal + dinner_kcal)
    dinner_kcal += drift

    targets: Dict[str, MacroSchema] = {
        "breakfast": _meal_macro_targets(breakfast_kcal),
        "morning_snack": _meal_macro_targets(morning_snack_kcal),
        "lunch": _meal_macro_targets(lunch_kcal),
        "evening_snack": _meal_macro_targets(evening_snack_kcal),
        "dinner": _meal_macro_targets(dinner_kcal),
    }
    total_day_targets = _sum_macros(list(targets.values()))

    diabetes_rule = ""
    if any("diabet" in c.lower() for c in chronic_conditions):
        diabetes_rule = "Diabetes: no added sugar/sugary drinks; prefer high-fiber/low-GI carbs; protein+fiber each meal."

    style_line = meal_plan_style or "Any"

    prompt = (
        "Return ONLY valid JSON matching MealPlanSchema.\n"
        f"User: age {age}, gender {gender}, {weight}kg, {height}cm, activity {activity_level}/5. "
        f"Day kcal {rounded_total}.\n"
        f"Restrictions: {dietary_restrictions or 'None'}. Allergies (avoid): {allergies or 'None'}. "
        f"Avoid foods: {foods_to_avoid or 'None'}. Conditions: {chronic_conditions or 'None'}.\n"
        f"Cuisines: {preferred_cuisines or 'Any'}. Style: {style_line}.\n"
        f"{(diabetes_rule + ' ') if diabetes_rule else ''}\n"
        f"{(meal_plan_instructions + ' ') if meal_plan_instructions else ''}\n"
        "Meals: breakfast, morning_snack, lunch, evening_snack, dinner.\n"
        "Use these meal ± 50 kcal + macro targets (grams) as target_macros; foods must sum close (±10%). "
        "Total day macros should match sum of meals (±5%). Saturated fat MUST stay under 10% of meal kcal.\n"
        f"Targets:\n"
        f"- Breakfast: kcal {targets['breakfast'].calories}, Protein {targets['breakfast'].protein_g}g, "
        f"Carbs {targets['breakfast'].carbs_g}g, Fat {targets['breakfast'].fat_g}g, "
        f"SatF {targets['breakfast'].saturated_fat_g}g, Fiber {targets['breakfast'].fiber_g}g\n"
        f"- Morning snack: kcal {targets['morning_snack'].calories}, Protein {targets['morning_snack'].protein_g}g, "
        f"Carbs {targets['morning_snack'].carbs_g}g, Fat {targets['morning_snack'].fat_g}g, "
        f"SatF {targets['morning_snack'].saturated_fat_g}g, Fiber {targets['morning_snack'].fiber_g}g\n"
        f"- Lunch: kcal {targets['lunch'].calories}, Protein {targets['lunch'].protein_g}g, "
        f"Carbs {targets['lunch'].carbs_g}g, Fat {targets['lunch'].fat_g}g, "
        f"SatF {targets['lunch'].saturated_fat_g}g, Fiber {targets['lunch'].fiber_g}g\n"
        f"- Evening snack: kcal {targets['evening_snack'].calories}, Protein {targets['evening_snack'].protein_g}g, "
        f"Carbs {targets['evening_snack'].carbs_g}g, Fat {targets['evening_snack'].fat_g}g, "
        f"SatF {targets['evening_snack'].saturated_fat_g}g, Fiber {targets['evening_snack'].fiber_g}g\n"
        f"- Dinner: kcal {targets['dinner'].calories}, Protein {targets['dinner'].protein_g}g, "
        f"Carbs {targets['dinner'].carbs_g}g, Fat {targets['dinner'].fat_g}g, "
        f"SatF {targets['dinner'].saturated_fat_g}g, Fiber {targets['dinner'].fiber_g}g\n"
        "Each meal: 1–4 foods with quantity, portion_weight_oz (number), and macros for that quantity. "
        "No allergens; respect restrictions/avoid list; easy ingredients."
    )
    return prompt  

# -----------------------------
# Meal parsing + insert section
# (kept from your code; no changes needed for the profile JSON issue)
# -----------------------------

MealType = Literal["breakfast", "morningsnack", "lunch", "eveningsnack", "dinner"]

class MealItemLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")
    FoodName: str
    Quantity: Decimal | None = None
    Unit: str | None = None
    Calories: Decimal | None = None
    ProteinGrams: Decimal | None = None
    CarbsGrams: Decimal | None = None
    FatGrams: Decimal | None = None
    FiberGrams: Decimal | None = None
    SugarGrams: Decimal | None = None
    SodiumMg: Decimal | None = None
    Notes: str | None = None

class MealParseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[MealItemLLM] = Field(..., min_length=1)

class SampleUserMealRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    UserId: int
    MealDate: date
    MealTime: time
    MealType: MealType
    FoodName: str
    Quantity: Decimal
    Unit: str
    Calories: Decimal
    ProteinGrams: Decimal
    CarbsGrams: Decimal
    FatGrams: Decimal
    FiberGrams: Decimal
    SugarGrams: Decimal
    SodiumMg: Decimal
    Notes: str

_ALLOWED_MEAL_TYPES = {"breakfast", "morningsnack", "lunch", "eveningsnack", "dinner"}

def _safe_preview(obj, max_len: int = 3000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= max_len else s[:max_len] + "…(truncated)"

def parse_date_string(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def parse_time_string(time_str: str) -> time:
    if len(time_str.split(":")) == 2:
        return datetime.strptime(time_str, "%H:%M").time()
    return datetime.strptime(time_str, "%H:%M:%S").time()

def normalize_meal_type(meal_type: str) -> str:
    return (meal_type or "").strip().lower()

def to_decimal_or_zero(v) -> Decimal:
    try:
        if v is None:
            return Decimal("0")
        if isinstance(v, Decimal):
            return v
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return Decimal("0")
            return Decimal(s)
        return Decimal("0")
    except (InvalidOperation, ValueError):
        logger.warning("Invalid decimal value encountered: %s → defaulting to 0", v)
        return Decimal("0")

def to_str_or_empty(v) -> str:
    if v is None:
        return ""
    return str(v).strip()

def _bindable(v):
    if v is None:
        return None
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, (date, time, datetime)):
        return v.isoformat()
    return v

def _json_default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, (date, datetime, time)):
        return o.isoformat()
    return str(o)

def build_row_from_llm_item(
    item: MealItemLLM,
    user_id: int,
    meal_date: date,
    meal_time: time,
    meal_type: str,
) -> SampleUserMealRow:
    food_name = to_str_or_empty(item.FoodName) or "Unknown"
    return SampleUserMealRow(
        UserId=user_id,
        MealDate=meal_date,
        MealTime=meal_time,
        MealType=meal_type,
        FoodName=food_name,
        Quantity=to_decimal_or_zero(item.Quantity),
        Unit=to_str_or_empty(item.Unit),
        Calories=to_decimal_or_zero(item.Calories),
        ProteinGrams=to_decimal_or_zero(item.ProteinGrams),
        CarbsGrams=to_decimal_or_zero(item.CarbsGrams),
        FatGrams=to_decimal_or_zero(item.FatGrams),
        FiberGrams=to_decimal_or_zero(item.FiberGrams),
        SugarGrams=to_decimal_or_zero(item.SugarGrams),
        SodiumMg=to_decimal_or_zero(item.SodiumMg),
        Notes=to_str_or_empty(item.Notes),
    )


# ── Internal helper (not a tool) ──────────────────────────────────────────────
def _validate_date(date_str: str, field_name: str) -> str:
    """
    Validate and normalise a date string to YYYY-MM-DD.
    Accepts both YYYY-MM-DD and DD-MM-YYYY (MealPlanSchema format).
    Raises ValueError with a descriptive message on failure.
    """
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(date_str.strip(), fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(
        f"Invalid date format for '{field_name}': '{date_str}'. "
        f"Expected YYYY-MM-DD or DD-MM-YYYY (e.g. 2025-07-21 or 21-07-2025)."
    )

async def _save_meal_plan_to_db(
    user_id: str,
    meal_date_raw: str,
    meal_plan: Any,
) -> None:
    """
    Internal helper — writes one day's meal plan to the meal_plans table
    with is_confirmed = 0 (pending).
 
    Called automatically by generate_meal_plan immediately after generation.
    Never exposed to the LLM as a tool.
 
    Args:
        user_id:       The current user's ID.
        meal_date_raw: Date string in either YYYY-MM-DD or DD-MM-YYYY format.
        meal_plan:     Pydantic model, dict, list, or JSON string.
    """
    from main import graph_state
 
    conn = graph_state.get("meal_plans_conn")
    if conn is None:
        logger.error("[_save_meal_plan_to_db] meal_plans_conn not available — plan not saved.")
        return
 
    # Normalise date to YYYY-MM-DD
    try:
        normalised_date = _validate_date(meal_date_raw, "meal_date")
    except ValueError as ve:
        logger.error("[_save_meal_plan_to_db] %s", ve)
        return
 
    # Serialise plan to JSON text
    if hasattr(meal_plan, "model_dump"):          # Pydantic model
        meal_plan_json = json.dumps(meal_plan.model_dump(), ensure_ascii=False, default=str)
    elif isinstance(meal_plan, str):
        try:
            meal_plan_json = json.dumps(json.loads(meal_plan), ensure_ascii=False)
        except json.JSONDecodeError:
            meal_plan_json = meal_plan
    else:
        meal_plan_json = json.dumps(meal_plan, ensure_ascii=False, default=str)
 
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
 
    try:
        await conn.execute(
            """
            INSERT INTO meal_plans (user_id, meal_date, meal_plan, is_confirmed, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
            ON CONFLICT(user_id, meal_date)
            DO UPDATE SET
                meal_plan    = excluded.meal_plan,
                is_confirmed = 0,
                updated_at   = excluded.updated_at
            """,
            (user_id, normalised_date, meal_plan_json, now_iso, now_iso),
        )
        await conn.commit()
        logger.info(
            "[_save_meal_plan_to_db] ✅ Pending plan written for user=%s date=%s.",
            user_id, normalised_date,
        )
    except Exception as e:
        logger.error("[_save_meal_plan_to_db] DB write failed: %s", e)
 
