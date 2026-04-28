import asyncio
import logging
import os
from typing import Annotated, Any, Optional, List
from langchain_core import messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from pydantic import BaseModel, Field
from typing import TypedDict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
from utils import (
    retrieve_all_memories,
    retrieve_memories,
    retrieve_user_profile_sql_lite,
    update_memory,
    clean_profile,
    workflow,
    generate_meal_plan_json,
    prompt,
    MealPlanInstructor,
    _save_meal_plan_to_db,
    _validate_date,
)
_DATE_FORMAT = "%Y-%m-%d"


# ─────────────────────────────────────────────
# 0. LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chatbot.memory")

from langchain_openai import ChatOpenAI

base_url = os.getenv("EXPERT_LLM_BASE_URL")
api_key = os.getenv("EXPERT_LLM_API_KEY")
model = os.getenv("EXPERT_LLM_MODEL")
_expert_llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model=model
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TOKEN_LIMIT      = 15000  # trigger summarization above this
MESSAGES_TO_KEEP = 6      # how many recent messages to preserve verbatim


# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────
class State(TypedDict):
    """
    Graph state.

    - messages:           Append-only conversation history (built-in reducer).
    - user_id:            Identifies the current user across nodes.
    - retrieved_memories: Memories fetched BEFORE the chatbot runs; injected
                          into the system prompt so no extra retrieval is needed
                          inside the chatbot node.
    - summary:            A rolling prose summary of messages that were trimmed
                          to keep the context window under TOKEN_LIMIT.
                          Empty string means no summarization has occurred yet.
    """
    messages:            Annotated[list, add_messages]
    user_id:             str
    retrieved_memories:  list[str]
    summary:             str


# ─────────────────────────────────────────────
# 2. TOOLS
# ─────────────────────────────────────────────

@tool
async def get_profile(state: Annotated[dict, InjectedState]) -> str:
    """
    Call this function if and only if you any profile data is missing.
    Fetch the user profile based on user_id from the state,
    persist it to local SQLite, then return it as a string.

    Args:
        state: LangGraph injected state containing user_id

    Returns:
        User profile as a string, or an error message.
    """
    logging.info("Get profile tool is called")
    try:
        user_id = state.get("user_id")
        if not user_id:
            logger.warning("get_profile called but user_id is missing from state.")
            return "Error: user_id not found in state"

        from main import graph_state
        conn = graph_state.get("profile_conn")
        if conn is None:
            logger.error("get_profile: Database connection not available in graph_state.")
            return "Error: Database connection not available."
        profile = await retrieve_user_profile_sql_lite(user_id, conn)
        if not profile:
            logger.info("get_profile: No profile found for user_id=%s.", user_id)
            return "No profile data available."
        cleaned = clean_profile(profile)
        logger.info("get_profile: Retrieved and cleaned profile for user_id=%s.", user_id)
        return cleaned
    except Exception as e:
        logger.error("get_profile error: %s", str(e))
        return f"Error retrieving profile: {str(e)}"

@tool
async def update_user_profile(
    state: Annotated[dict, InjectedState],
    name: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    weight_kg: Optional[float] = None,
    height_cm: Optional[float] = None,
    waist_circumference_cm: Optional[float] = None,
    cuisine: Optional[str] = None,
    activity_level: Optional[str] = None,
    calories: Optional[str] = None,
    dietary_preference: Optional[str] = None,
    restrictions: Optional[str] = None,
    digestive_issues: Optional[str] = None,
    allergies: Optional[str] = None,
    symptom_aggravating_foods: Optional[str] = None,
    heart_rate: Optional[float] = None,
    blood_pressure: Optional[str] = None,
    body_temperature: Optional[float] = None,
    blood_oxygen: Optional[float] = None,
    respiratory_rate: Optional[float] = None,
    medical_conditions: Optional[str] = None,
    goals: Optional[str] = None,
) -> str:
    """
    Update one or more fields in the UserHealthProfile table for the current user.
    Only the fields that are explicitly provided will be updated.

    Args:
        state:                        LangGraph injected state containing user_id
        name:                         Full name of the user
        age:                          Age of the user
        gender:                       Gender of the user
        weight_kg:                    Weight in kilos
        height_cm:                    Height in centimeters
        waist_circumference_cm:       Waist circumference in centimeters
        cuisine:                      Preferred cuisine type
        activity_level:               Physical activity level
        calories:                     Caloric intake or target (e.g. '2000 kcal')
        dietary_preference:           Dietary preferences (comma-separated)
        restrictions:                 Dietary restrictions (comma-separated)
        digestive_issues:             Digestive issues (comma-separated)
        allergies:                    Allergies (comma-separated)
        symptom_aggravating_foods:    Foods that aggravate symptoms (comma-separated)
        heart_rate:                   Heart rate in bpm
        blood_pressure:               Blood pressure (e.g. '120/80')
        body_temperature:             Body temperature in °F
        blood_oxygen:                 Blood oxygen percentage
        respiratory_rate:             Respiratory rate in breaths/min
        medical_conditions:           Medical conditions (comma-separated)
        goals:                        Health goals (comma-separated)

    Returns:
        Success or error message as a string
    """
    FIELD_MAP = {
        "name":                       "Name",
        "age":                        "Age",
        "gender":                     "Gender",
        "weight_kg":                  "WeightKG",
        "height_cm":                  "HeightCM",
        "waist_circumference_cm":     "WaistCircumferenceCM",
        "cuisine":                    "Cuisine",
        "activity_level":             "ActivityLevel",
        "calories":                   "Calories",
        "dietary_preference":         "DietaryPreference",
        "restrictions":               "Restrictions",
        "digestive_issues":           "DigestiveIssues",
        "allergies":                  "Allergies",
        "symptom_aggravating_foods":  "SymptomAggravatingFoods",
        "heart_rate":                 "HeartRate",
        "blood_pressure":             "BloodPressure",
        "body_temperature":           "BodyTemperature",
        "blood_oxygen":               "BloodOxygen",
        "respiratory_rate":           "RespiratoryRate",
        "medical_conditions":         "MedicalConditions",
        "goals":                      "Goals",
    }

    try:
        user_id = state.get("user_id")
        if not user_id:
            return "Error: user_id not found in state"

        provided = {
            param: value
            for param, value in {
                "name":                       name,
                "age":                        age,
                "gender":                     gender,
                "weight_kg":                  weight_kg,
                "height_cm":                  height_cm,
                "waist_circumference_cm":     waist_circumference_cm,
                "cuisine":                    cuisine,
                "activity_level":             activity_level,
                "calories":                   calories,
                "dietary_preference":         dietary_preference,
                "restrictions":               restrictions,
                "digestive_issues":           digestive_issues,
                "allergies":                  allergies,
                "symptom_aggravating_foods":  symptom_aggravating_foods,
                "heart_rate":                 heart_rate,
                "blood_pressure":             blood_pressure,
                "body_temperature":           body_temperature,
                "blood_oxygen":               blood_oxygen,
                "respiratory_rate":           respiratory_rate,
                "medical_conditions":         medical_conditions,
                "goals":                      goals,
            }.items()
            if value is not None
        }

        if not provided:
            return "Error: No fields provided to update."

        # ✅ Basic input validation
        if age is not None and age <= 0:
            return "Error: age must be a positive integer."
        if weight_kg is not None and weight_kg <= 0:
            return "Error: weight_lbs must be a positive number."
        if height_cm is not None and height_cm <= 0:
            return "Error: height_inches must be a positive number."
        if waist_circumference_cm is not None and waist_circumference_cm <= 0:
            return "Error: waist_circumference_inches must be a positive number."

        set_clause = ", ".join(
            f"{FIELD_MAP[param]} = :{param}" for param in provided
        )
        params = {**provided, "user_id": user_id}
        from main import graph_state 
        conn = graph_state.get("profile_conn")
        if conn is None:
            return "Error: Database connection not available."

        await conn.execute(
            f"""
            UPDATE UserHealthProfile
            SET {set_clause},
                ModifiedDate = datetime('now')
            WHERE user_id = :user_id
            """,
            params,
        )
        await conn.commit()

        updated = ", ".join(FIELD_MAP[p] for p in provided)
        logger.info("update_user_profile: updated [%s] for user_id=%s.", updated, user_id)

        return f"Successfully updated [{updated}] for user_id={user_id}."

    except Exception as e:
        logger.error("update_user_profile error: %s", str(e))
        return f"Error updating profile field: {str(e)}"

@tool
def daily_calorie_requirement(
    age: int,
    gender: str,
    weight: float,
    height: float,
    activity_level: int,
    calorie_goal_adjustment: int,
) -> dict:
    """
    Calculate daily calorie requirement using the Mifflin-St Jeor Equation with an optional calorie adjustment for weight loss or weight gain.

    Args:
        age: Age in years
        gender: 'male' or 'female'
        weight: Weight in kg
        height: Height in cm
        activity_level: 1 (sedentary) to 5 (very active)
        calorie_goal_adjustment: Calories to add/subtract from maintenance calories.
            - Negative → calorie deficit → weight loss (e.g. -500 ≈ ~0.5 kg/week loss)
            - Positive → calorie surplus → weight gain (e.g. +500 ≈ ~0.5 kg/week gain)
            - Zero     → maintenance → stable weight

            Recommended range: 250–500 kcal/day.
            Avoid exceeding ±1000 kcal/day without medical supervision.

    Returns:
        dict
    """

    # Step 1: Calculate BMR
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender.lower() == "female":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Gender must be 'male' or 'female'")

    # Step 2: Activity multiplier
    activity_multipliers = {
        1: ("Sedentary", 1.2),
        2: ("Lightly active", 1.375),
        3: ("Moderately active", 1.55),
        4: ("Very active", 1.725),
        5: ("Super active", 1.9),
    }

    if activity_level not in activity_multipliers:
        raise ValueError("Activity level must be between 1 and 5")

    activity_name, multiplier = activity_multipliers[activity_level]

    # Step 3: Maintenance calories
    maintenance_calories = bmr * multiplier

    # Step 4: Adjustment logic
    final_calories = maintenance_calories + calorie_goal_adjustment

    if calorie_goal_adjustment > 0:
        goal_type = "Calorie Surplus (Weight Gain)"
        explanation = (
            f"You are adding {calorie_goal_adjustment} kcal to your maintenance "
            f"to support weight gain."
        )
    elif calorie_goal_adjustment < 0:
        goal_type = "Calorie Deficit (Weight Loss)"
        explanation = (
            f"You are reducing {abs(calorie_goal_adjustment)} kcal from your maintenance "
            f"to support weight loss."
        )
    else:
        goal_type = "Maintenance (Weight Stable)"
        explanation = "No calorie adjustment applied. You will maintain your current weight."

    # Step 5: Return structured result
    return {
        "bmr": round(bmr, 2),
        "activity_level": activity_name,
        "maintenance_calories": round(maintenance_calories, 2),
        "adjustment": calorie_goal_adjustment,
        "final_calories": round(final_calories, 2),
        "goal_type": goal_type,
        "explanation": explanation,
    }

@tool
async def generate_meal_plan(
    state: Annotated[dict, InjectedState],
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
    meal_plan_instructions_by_day: Optional[List[Optional[str]]] = None,
    meal_start_date: Optional[str] = None,
    days: int = 1,
) -> Any:
    """
    Generate a structured daily meal plan (or multi-day 7 day plans) tailored to the user's profile.

    Use this tool when the user asks for a meal plan, diet plan, or what to eat.

    Parameter guidance:
    - meal_plan_instructions (single-day only):
        A single string describing what the user likes to eat on that day.
        Use ONLY when days == 1.

    - meal_plan_instructions_by_day (multi-day 7 day only):
        A list of per-day instruction strings. REQUIRED when days = 7.
        Must have exactly 7 elements (index 0 = Day 1, index 6 = Day 7).

    Behavior rules (IMPORTANT):
    - If days == 1: provide meal_plan_instructions (optional), NOT by_day list.
    - If days == 7: provide meal_plan_instructions_by_day (length 7), NOT single instructions.
    """
    if not isinstance(days, int) or days < 1:
        raise ValueError("days must be an integer >= 1")

    if days == 1:
        if meal_plan_instructions_by_day:
            meal_plan_instructions = meal_plan_instructions_by_day[0]

        result = await generate_meal_plan_json(
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            activity_level=activity_level,
            calorie_goal=calorie_goal,
            dietary_restrictions=dietary_restrictions,
            allergies=allergies,
            foods_to_avoid=foods_to_avoid,
            chronic_conditions=chronic_conditions,
            preferred_cuisines=preferred_cuisines,
            meal_plan_style=meal_plan_style,
            meal_plan_instructions=meal_plan_instructions,
            meal_start_date=meal_start_date,
        )
        user_id = state["user_id"]
        if result and result.meal_date:
            await _save_meal_plan_to_db(user_id, result.meal_date, result)
            logger.info("[generate_meal_plan] Single-day plan auto-saved for user=%s", user_id)

        return result
      
    profile = {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "activity_level": activity_level,
        "calorie_goal": calorie_goal,
        "dietary_restrictions": dietary_restrictions,
        "allergies": allergies,
        "foods_to_avoid": foods_to_avoid,
        "chronic_conditions": chronic_conditions,
        "preferred_cuisines": preferred_cuisines,
        "meal_plan_style": meal_plan_style,
        "meal_plan_instructions": meal_plan_instructions
    }
    from main import graph_state
    conn = graph_state.get("memory_conn")
    user_id = state["user_id"]
    # meal_plan_instructions = await MealPlanInstructor(user_id, conn, profile)
    _instructions = await MealPlanInstructor(user_id, conn, profile, meal_start_date=meal_start_date)
    meal_plan_instructions_by_day = [
    _instructions.Day1Instructions,
    _instructions.Day2Instructions,
    _instructions.Day3Instructions,
    _instructions.Day4Instructions,
    _instructions.Day5Instructions,
    _instructions.Day6Instructions,
    _instructions.Day7Instructions,
    ]
    print(f"meal_plan_instructions: {_instructions}")

    prompts = [
        prompt(
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            activity_level=activity_level,
            calorie_goal=calorie_goal,
            dietary_restrictions=dietary_restrictions,
            allergies=allergies,
            foods_to_avoid=foods_to_avoid,
            chronic_conditions=chronic_conditions,
            preferred_cuisines=preferred_cuisines,
            meal_plan_style=meal_plan_style,
            meal_plan_instructions=meal_plan_instructions_by_day[day_idx - 1],
        )
        for day_idx in range(1, days + 1)
    ]

    initial_state = {
        f"day{i + 1}instructions": prompts[i] if i < len(prompts) else f"Plan a healthy meal for day {i + 1}"
        for i in range(7)
    }

    result = await workflow.ainvoke(initial_state)
    final_plan = result["final7daymealplan"]

    # ── Auto-save all 7 days ───────────────────────────────────────────────
    user_id = state["user_id"]
    day_keys = ["day1meal", "day2meal", "day3meal", "day4meal",
                "day5meal", "day6meal", "day7meal"]

    for key in day_keys:
        day_plan = final_plan.get(key)
        if day_plan is None:
            continue
        # MealPlanSchema object → get meal_date attribute
        meal_date_val = (
            day_plan.meal_date                        # if still a Pydantic object
            if hasattr(day_plan, "meal_date")
            else (day_plan.get("meal_date") if isinstance(day_plan, dict) else None)
        )
        if meal_date_val:
            await _save_meal_plan_to_db(user_id, meal_date_val, day_plan)

    logger.info("[generate_meal_plan] 7-day plan auto-saved for user=%s", user_id)
    return final_plan

@tool
async def nutritional_expert_analysis(query: str) -> dict:
    """
    General nutrition analysis (non-personalized) in structured JSON format.
    """
    class NutritionalAnalysisResponse(BaseModel):
        calories_estimate: Optional[float] = Field(
            default=None, description="Estimated calories if applicable"
        )
        nutrients: Optional[List[str]] = Field(
            default=None, description="Key nutrients present in the food"
        )
        health_benefits: Optional[List[str]] = Field(
            default=None, description="Health benefits of the food"
        )
        concerns: Optional[List[str]] = Field(
            default=None, description="Potential concerns or risks"
        )
        summary: str = Field(
            ..., description="Overall concise nutritional summary"
        )

    try:
        structured_llm = _expert_llm.with_structured_output(
            NutritionalAnalysisResponse
        )

        response = await structured_llm.ainvoke(query)

        return response.model_dump()

    except Exception as e:
        return {
            "error": f"Error during nutritional analysis: {str(e)}"
        }

@tool
async def get_personalised_expert_analysis_for_user_query(
    query: str,
    state: Annotated[dict, InjectedState]
) -> dict:
    """
    Personalized nutrition/health analysis using user profile in structured JSON format.
    """
    class PersonalizedExpertAnalysisResponse(BaseModel):
        is_suitable: bool = Field(
            ..., description="Whether this is suitable for the user"
        )
        reasoning: str = Field(
            ..., description="Explanation based on user's profile"
        )
        alternatives: Optional[List[str]] = Field(
            default=None, description="Healthier alternatives"
        )
        recommendations: Optional[List[str]] = Field(
            default=None, description="Additional personalized suggestions"
        )
        confidence: Optional[str] = Field(
            default=None, description="Confidence level: low, medium, high"
        )

    EXPERT_SYSTEM_PROMPT = """
    You are a professional nutrition and health expert.

    Based on the user's query and profile:
    - Decide if it is suitable (true/false)
    - Explain clearly why
    - Suggest alternatives if not suitable
    - Provide additional recommendations if useful
    - Include confidence level (low, medium, high)

    Always return structured output matching the schema.
    """

    try:
        user_id = state.get("user_id")
        if not user_id:
            return {"error": "User ID not found in state."}
        from main import graph_state
        conn = graph_state.get("profile_conn")
        if not conn:
            return {"error": "Profile DB connection not available."}

        profile = await retrieve_user_profile_sql_lite(user_id, conn)
        if not profile:
            profile = "No profile data available."

        user_message = f"User query: {query}\n\nUser profile: {profile}"

        structured_llm = _expert_llm.with_structured_output(
            PersonalizedExpertAnalysisResponse
        )

        response = await structured_llm.ainvoke([
            SystemMessage(content=EXPERT_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        return response.model_dump()

    except Exception as e:
        return {
            "error": f"Error during personalized analysis: {str(e)}"
        }

@tool
async def save_meal_plan(
    state: Annotated[dict, InjectedState],
    meal_date: str,
) -> dict:
    """
    Confirm and permanently save a meal plan that was previously generated.
 
    When a meal plan is generated it is stored as PENDING (is_confirmed = 0).
    Call this tool ONLY after the user explicitly agrees to save the plan.
    It flips is_confirmed from 0 → 1 for the given date, marking the plan
    as permanently accepted by the user.
 
    Args:
        state:     LangGraph injected state — provides user_id automatically.
        meal_date: The date of the meal plan to confirm, in YYYY-MM-DD format
                   (e.g. "2025-07-21").
 
    Returns:
        dict with keys:
            success      (bool) – True if confirmed successfully.
            message      (str)  – Human-readable status.
            user_id      (str)  – The current user.
            meal_date    (str)  – The confirmed date.
            is_confirmed (int)  – 1 if confirmed, 0 if not found / already pending.
    """
    try:
        user_id = state.get("user_id")
        if not user_id:
            return {"success": False, "message": "Error: user_id not found in state."}
 
        normalised_date = _validate_date(meal_date, "meal_date")
 
        from main import graph_state
        conn = graph_state.get("meal_plans_conn")
        if conn is None:
            return {"success": False, "message": "Error: Meal plans database connection not available."}
 
        # Check whether a pending plan actually exists for this date
        cursor = await conn.execute(
            "SELECT is_confirmed FROM meal_plans WHERE user_id = ? AND meal_date = ?",
            (user_id, normalised_date),
        )
        row = await cursor.fetchone()
 
        if row is None:
            return {
                "success":      False,
                "message":      (
                    f"No meal plan found for {normalised_date}. "
                    "Please generate a meal plan first."
                ),
                "user_id":      user_id,
                "meal_date":    normalised_date,
                "is_confirmed": 0,
            }
 
        if row[0] == 1:
            return {
                "success":      True,
                "message":      f"Meal plan for {normalised_date} is already confirmed.",
                "user_id":      user_id,
                "meal_date":    normalised_date,
                "is_confirmed": 1,
            }
 
        # Flip is_confirmed to 1 and record the confirmation timestamp
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        await conn.execute(
            """
            UPDATE meal_plans
            SET    is_confirmed = 1,
                   updated_at   = ?
            WHERE  user_id   = ?
              AND  meal_date  = ?
            """,
            (now_iso, user_id, normalised_date),
        )
        await conn.commit()
 
        logger.info(
            "[save_meal_plan] ✅ Confirmed meal plan for user_id=%s, date=%s.",
            user_id, normalised_date,
        )
        return {
            "success":      True,
            "message":      f"Meal plan for {normalised_date} has been confirmed and saved.",
            "user_id":      user_id,
            "meal_date":    normalised_date,
            "is_confirmed": 1,
        }
 
    except ValueError as ve:
        return {"success": False, "message": str(ve)}
    except Exception as e:
        logger.error("[save_meal_plan] Error: %s", str(e))
        return {"success": False, "message": f"Error confirming meal plan: {str(e)}"}
 
@tool
async def get_meal_plan(
    state: Annotated[dict, InjectedState],
    meal_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    confirmed_only: bool = False,
) -> dict:
    """
    Retrieve previously saved meal plan(s) for the current user.
 
    Supports two query modes — provide exactly one:
 
    Mode 1 — Single day:
        Pass `meal_date` (YYYY-MM-DD) to fetch the plan for that specific date.
 
    Mode 2 — Date range:
        Pass both `start_date` AND `end_date` (YYYY-MM-DD) to fetch all plans
        within the inclusive date range.
 
    Args:
        state:          LangGraph injected state — provides user_id automatically.
        meal_date:      Exact date to query (YYYY-MM-DD).  Used for single-day lookup.
        start_date:     Range start date (YYYY-MM-DD).  Used together with end_date.
        end_date:       Range end date   (YYYY-MM-DD).  Used together with start_date.
        confirmed_only: If True, return ONLY confirmed (user-accepted) plans.
                        If False (default), return ALL plans including pending ones.
                        Use True when the user asks to see their saved/confirmed plans.
                        Use False when checking whether a plan exists at all.
 
    Returns:
        dict with keys:
            success        (bool)        – True if query ran without errors.
            user_id        (str)         – The current user.
            query_mode     (str)         – "single_day" | "date_range".
            meal_plans     (list[dict])  – Zero or more plan records, each containing:
                                            meal_date    (str)
                                            meal_plan    (Any)  parsed from JSON
                                            is_confirmed (int)  1 = confirmed, 0 = pending
                                            created_at   (str)
                                            updated_at   (str)
            count          (int)         – Number of records returned.
            message        (str)         – Human-readable status.
    """
    try:
        user_id = state.get("user_id")
        if not user_id:
            return {"success": False, "message": "Error: user_id not found in state."}
 
        from main import graph_state
        conn = graph_state.get("meal_plans_conn")
        if conn is None:
            return {"success": False, "message": "Error: Meal plans database connection not available."}
 
        # ── Build confirmation filter clause ───────────────────────────────
        confirm_clause = "AND is_confirmed = 1" if confirmed_only else ""
 
        # ── Determine query mode ───────────────────────────────────────────
        has_single  = meal_date is not None
        has_range   = start_date is not None and end_date is not None
        has_partial = (start_date is None) != (end_date is None)  # XOR
 
        if has_partial:
            return {
                "success": False,
                "message": "Both start_date and end_date must be provided together for a range query.",
            }
 
        if not has_single and not has_range:
            return {
                "success": False,
                "message": (
                    "Provide either 'meal_date' for a single-day lookup, "
                    "or both 'start_date' and 'end_date' for a range query."
                ),
            }
 
        if has_single:
            # ── Mode 1: single day ─────────────────────────────────────────
            normalised = _validate_date(meal_date, "meal_date")
            cursor = await conn.execute(
                f"""
                SELECT meal_date, meal_plan, is_confirmed, created_at, updated_at
                FROM   meal_plans
                WHERE  user_id   = ?
                  AND  meal_date = ?
                  {confirm_clause}
                """,
                (user_id, normalised),
            )
            rows = await cursor.fetchall()
            query_mode = "single_day"
 
        else:
            # ── Mode 2: date range ─────────────────────────────────────────
            norm_start = _validate_date(start_date, "start_date")
            norm_end   = _validate_date(end_date,   "end_date")
 
            if norm_start > norm_end:
                return {
                    "success": False,
                    "message": f"start_date ({norm_start}) must be on or before end_date ({norm_end}).",
                }
 
            cursor = await conn.execute(
                f"""
                SELECT meal_date, meal_plan, is_confirmed, created_at, updated_at
                FROM   meal_plans
                WHERE  user_id   = ?
                  AND  meal_date >= ?
                  AND  meal_date <= ?
                  {confirm_clause}
                ORDER BY meal_date ASC
                """,
                (user_id, norm_start, norm_end),
            )
            rows = await cursor.fetchall()
            query_mode = "date_range"
 
        # ── Parse rows ─────────────────────────────────────────────────────
        records = []
        for row in rows:
            meal_date_val, meal_plan_json, is_confirmed, created_at, updated_at = row
            try:
                meal_plan_parsed = json.loads(meal_plan_json)
            except (json.JSONDecodeError, TypeError):
                meal_plan_parsed = meal_plan_json
 
            records.append({
                "meal_date":    meal_date_val,
                "meal_plan":    meal_plan_parsed,
                "is_confirmed": is_confirmed,   # 1 = confirmed, 0 = pending
                "created_at":   created_at,
                "updated_at":   updated_at,
            })
 
        count = len(records)
        filter_label = "confirmed " if confirmed_only else ""
 
        if count == 0:
            message = f"No {filter_label}meal plans found for the specified criteria."
        elif count == 1:
            status  = "✅ Confirmed" if records[0]["is_confirmed"] else "⏳ Pending"
            message = f"Found 1 {filter_label}meal plan for {records[0]['meal_date']} ({status})."
        else:
            message = (
                f"Found {count} {filter_label}meal plans "
                f"from {records[0]['meal_date']} to {records[-1]['meal_date']}."
            )
 
        logger.info(
            "[get_meal_plan] user_id=%s mode=%s confirmed_only=%s returned %d record(s).",
            user_id, query_mode, confirmed_only, count,
        )
 
        return {
            "success":    True,
            "user_id":    user_id,
            "query_mode": query_mode,
            "meal_plans": records,
            "count":      count,
            "message":    message,
        }
 
    except ValueError as ve:
        return {"success": False, "message": str(ve)}
    except Exception as e:
        logger.error("[get_meal_plan] Error: %s", str(e))
        return {"success": False, "message": f"Error retrieving meal plan: {str(e)}"}
 
@tool
async def log_meal(
    state: Annotated[dict, InjectedState],
    meal_name: str,
    meal_type: str,
    meal_occasion: str,
    consumed_date: Optional[str] = None,
    consumed_time: Optional[str] = None,
    calories: Optional[float] = None,
    protein_g: Optional[float] = None,
    carbohydrates_g: Optional[float] = None,
    fats_g: Optional[float] = None,
    fiber_g: Optional[float] = None,
    sugar_g: Optional[float] = None,
    sodium_mg: Optional[float] = None,
    notes: Optional[str] = None,
) -> dict:
    """
    Log a meal that the user has consumed.
 
    Use this tool when the user says they ate, had, or consumed something,
    or when they want to track a meal they just finished.
 
    Args:
        state:           LangGraph injected state — provides user_id automatically.
        meal_name:       Name or description of the meal (e.g. 'Grilled chicken salad').
        meal_type:       Category of the meal (e.g. 'home-cooked', 'restaurant',
                         'packaged', 'snack').
        meal_occasion:   When in the day the meal was eaten. Must be one of:
                         'breakfast', 'morning_snack', 'lunch', 'afternoon_snack',
                         'dinner', 'late_night_snack'.
        consumed_date:   Date the meal was consumed in YYYY-MM-DD format.
                         Defaults to today if not provided.
        consumed_time:   Time the meal was consumed in HH:MM (24-hour) format.
                         Defaults to current time if not provided.
        calories:        Total calories in kcal.
        protein_g:       Protein content in grams.
        carbohydrates_g: Carbohydrate content in grams.
        fats_g:          Fat content in grams.
        fiber_g:         Fiber content in grams.
        sugar_g:         Sugar content in grams.
        sodium_mg:       Sodium content in milligrams.
        notes:           Any extra notes (e.g. 'added extra dressing', 'half portion').
 
    Returns:
        dict with keys: success, message, meal_id, user_id, consumed_date, consumed_time.
    """
    VALID_OCCASIONS = {
        "breakfast", "morning_snack", "lunch",
        "afternoon_snack", "dinner", "late_night_snack",
    }
 
    try:
        user_id = state.get("user_id")
        if not user_id:
            return {"success": False, "message": "Error: user_id not found in state."}
 
        if meal_occasion.lower() not in VALID_OCCASIONS:
            return {
                "success": False,
                "message": (
                    f"Invalid meal_occasion '{meal_occasion}'. "
                    f"Must be one of: {', '.join(sorted(VALID_OCCASIONS))}."
                ),
            }
 
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
 
        if consumed_date:
            normalised_date = _validate_date(consumed_date, "consumed_date")
        else:
            normalised_date = now_ist.strftime(_DATE_FORMAT)
 
        if consumed_time:
            # Accept HH:MM or HH:MM:SS — normalise to HH:MM
            try:
                t = datetime.strptime(consumed_time.strip(), "%H:%M:%S")
            except ValueError:
                t = datetime.strptime(consumed_time.strip(), "%H:%M")
            normalised_time = t.strftime("%H:%M")
        else:
            normalised_time = now_ist.strftime("%H:%M")
 
        now_iso = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
 
        from main import graph_state
        conn = graph_state.get("consumed_meals_conn")
        if conn is None:
            return {"success": False, "message": "Error: Consumed meals database connection not available."}
 
        cursor = await conn.execute(
            """
            INSERT INTO consumed_meals (
                user_id, meal_name, meal_type, consumed_date, consumed_time,
                meal_occasion, calories, protein_g, carbohydrates_g, fats_g,
                fiber_g, sugar_g, sodium_mg, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, meal_name.strip(), meal_type.strip(),
                normalised_date, normalised_time, meal_occasion.lower(),
                calories, protein_g, carbohydrates_g, fats_g,
                fiber_g, sugar_g, sodium_mg,
                notes.strip() if notes else None,
                now_iso, now_iso,
            ),
        )
        await conn.commit()
        meal_id = cursor.lastrowid
 
        logger.info(
            "[log_meal] ✅ Logged meal id=%d for user_id=%s on %s at %s.",
            meal_id, user_id, normalised_date, normalised_time,
        )
        return {
            "success":       True,
            "message":       (
                f"'{meal_name}' logged as {meal_occasion} on "
                f"{normalised_date} at {normalised_time}."
            ),
            "meal_id":       meal_id,
            "user_id":       user_id,
            "consumed_date": normalised_date,
            "consumed_time": normalised_time,
        }
 
    except ValueError as ve:
        return {"success": False, "message": str(ve)}
    except Exception as e:
        logger.error("[log_meal] Error: %s", str(e))
        return {"success": False, "message": f"Error logging meal: {str(e)}"}

@tool
async def get_consumed_meals(
    state: Annotated[dict, InjectedState],
    consumed_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    meal_occasion: Optional[str] = None,
) -> dict:
    """
    Retrieve meals the user has previously logged as consumed.
 
    Supports two query modes — provide exactly one:
 
    Mode 1 — Single day:
        Pass `consumed_date` (YYYY-MM-DD) to fetch all logged meals for that date.
 
    Mode 2 — Date range:
        Pass both `start_date` AND `end_date` (YYYY-MM-DD) to fetch meals across
        an inclusive date range.
 
    Optionally filter by `meal_occasion` (e.g. 'breakfast', 'lunch', 'dinner').
 
    Args:
        state:          LangGraph injected state — provides user_id automatically.
        consumed_date:  Exact date (YYYY-MM-DD) for a single-day lookup.
        start_date:     Range start date (YYYY-MM-DD).
        end_date:       Range end date   (YYYY-MM-DD).
        meal_occasion:  Optional filter — one of: 'breakfast', 'morning_snack',
                        'lunch', 'afternoon_snack', 'dinner', 'late_night_snack'.
 
    Returns:
        dict with keys:
            success      (bool)
            user_id      (str)
            query_mode   (str)  – 'single_day' | 'date_range'
            meals        (list[dict]) – each record contains all logged fields
            count        (int)
            message      (str)
    """
    try:
        user_id = state.get("user_id")
        if not user_id:
            return {"success": False, "message": "Error: user_id not found in state."}
 
        from main import graph_state
        conn = graph_state.get("consumed_meals_conn")
        if conn is None:
            return {"success": False, "message": "Error: Consumed meals database connection not available."}
 
        # ── Occasion filter ────────────────────────────────────────────────
        occasion_clause = ""
        occasion_param: list = []
        if meal_occasion:
            occasion_clause = "AND meal_occasion = ?"
            occasion_param  = [meal_occasion.lower()]
 
        # ── Determine query mode ───────────────────────────────────────────
        has_single  = consumed_date is not None
        has_range   = start_date is not None and end_date is not None
        has_partial = (start_date is None) != (end_date is None)
 
        if has_partial:
            return {
                "success": False,
                "message": "Both start_date and end_date must be provided together for a range query.",
            }
 
        if not has_single and not has_range:
            return {
                "success": False,
                "message": (
                    "Provide either 'consumed_date' for a single-day lookup, "
                    "or both 'start_date' and 'end_date' for a range query."
                ),
            }
 
        COLUMNS = (
            "id", "meal_name", "meal_type", "consumed_date", "consumed_time",
            "meal_occasion", "calories", "protein_g", "carbohydrates_g", "fats_g",
            "fiber_g", "sugar_g", "sodium_mg", "notes", "created_at",
        )
        SELECT = f"SELECT {', '.join(COLUMNS)} FROM consumed_meals"
 
        if has_single:
            normalised = _validate_date(consumed_date, "consumed_date")
            cursor = await conn.execute(
                f"""
                {SELECT}
                WHERE user_id = ? AND consumed_date = ?
                {occasion_clause}
                ORDER BY consumed_time ASC
                """,
                [user_id, normalised] + occasion_param,
            )
            query_mode = "single_day"
        else:
            norm_start = _validate_date(start_date, "start_date")
            norm_end   = _validate_date(end_date,   "end_date")
            if norm_start > norm_end:
                return {
                    "success": False,
                    "message": f"start_date ({norm_start}) must be on or before end_date ({norm_end}).",
                }
            cursor = await conn.execute(
                f"""
                {SELECT}
                WHERE user_id = ?
                  AND consumed_date >= ? AND consumed_date <= ?
                  {occasion_clause}
                ORDER BY consumed_date ASC, consumed_time ASC
                """,
                [user_id, norm_start, norm_end] + occasion_param,
            )
            query_mode = "date_range"
 
        rows  = await cursor.fetchall()
        meals = [dict(zip(COLUMNS, row)) for row in rows]
        count = len(meals)
 
        if count == 0:
            message = "No consumed meals found for the specified criteria."
        elif count == 1:
            message = f"Found 1 meal logged on {meals[0]['consumed_date']} at {meals[0]['consumed_time']}."
        else:
            message = f"Found {count} meals logged."
 
        logger.info(
            "[get_consumed_meals] user_id=%s mode=%s occasion=%s returned %d record(s).",
            user_id, query_mode, meal_occasion, count,
        )
        return {
            "success":    True,
            "user_id":    user_id,
            "query_mode": query_mode,
            "meals":      meals,
            "count":      count,
            "message":    message,
        }
 
    except ValueError as ve:
        return {"success": False, "message": str(ve)}
    except Exception as e:
        logger.error("[get_consumed_meals] Error: %s", str(e))
        return {"success": False, "message": f"Error retrieving consumed meals: {str(e)}"}

TOOLS = [get_profile, update_user_profile, daily_calorie_requirement, generate_meal_plan,
         get_personalised_expert_analysis_for_user_query, nutritional_expert_analysis,
         save_meal_plan, get_meal_plan, log_meal, get_consumed_meals]

# ─────────────────────────────────────────────
# 3. LLM + TOOL BINDING
# ─────────────────────────────────────────────

def build_llm() -> ChatMistralAI:
    """
    Create the ChatMistralAI instance from environment variables.
    Set MISTRAL_ENDPOINT and MISTRAL_API_KEY before running.
    """
    MISTRAL_ENDPOINT = "https://patient-engagement.westus.models.ai.azure.com"
    MISTRAL_API_KEY  = "lowOIwacBNGCr2nTUK9eHZfdyAO7D2mE"
    # MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT", MISTRAL_ENDPOINT)
    # MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY", MISTRAL_API_KEY)

    if not MISTRAL_ENDPOINT or not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_ENDPOINT and MISTRAL_API_KEY must be set as environment variables."
        )

    return ChatMistralAI(endpoint=MISTRAL_ENDPOINT, api_key=MISTRAL_API_KEY)


# ─────────────────────────────────────────────
# 4. PYDANTIC SCHEMAS FOR STRUCTURED MEMORY OUTPUT
# ─────────────────────────────────────────────

class MemoryItem(BaseModel):
    text:   str
    is_new: bool
    is_mealplan: bool


class MemoryDecision(BaseModel):
    should_write: bool
    memories:     list[MemoryItem] = Field(default_factory=list)


_MEMORY_PROMPT = """
You are a long-term memory extraction engine for Friska, a personalised health and nutrition assistant.
 
Your sole job is to decide whether the user's latest message contains information worth storing
permanently — and if so, extract it as clean, atomic memory entries.
 
─────────────────────────────────────────────────────
CONTEXT
─────────────────────────────────────────────────────
CURRENT USER MEMORY (already stored — do NOT duplicate):
{user_details_content}
 
LATEST USER MESSAGE:
{user_message}
 
─────────────────────────────────────────────────────
THE GOLDEN RULE  (apply this first)
─────────────────────────────────────────────────────
Ask yourself:
 
  "If this user comes back in two weeks and says nothing about this topic,
   would knowing this memory make Friska's response meaningfully more personalised?"
 
If YES  → candidate for storage.
If NO   → discard immediately.
 
─────────────────────────────────────────────────────
WHAT TO STORE  (long-term, stable, reusable facts)
─────────────────────────────────────────────────────
A. FOOD PREFERENCES
   - Specific foods or ingredients the user enjoys or dislikes
   - Favourite cuisines (e.g., South Indian, Mediterranean)
   - Preferred cooking styles (e.g., grilled over fried)
   - Texture or flavour preferences (e.g., dislikes strong spices)
 
B. DIETARY RESTRICTIONS & ALLERGIES
   - Allergies stated by the user (e.g., peanuts, shellfish, dairy)
   - Intolerances (e.g., lactose intolerance, gluten sensitivity)
   - Chosen diet types (e.g., vegetarian, vegan, keto, intermittent fasting)
   - Foods the user explicitly avoids for personal/religious reasons
 
C. EATING HABITS & PATTERNS
   - Meal timing preferences (e.g., skips breakfast, eats dinner late)
   - Snacking habits (e.g., prefers light evening snacks)
   - Portion tendencies (e.g., prefers smaller, more frequent meals)
   - Cooking capacity (e.g., limited cooking time on weekdays)
 
D. HEALTH CONDITIONS  (user-stated only — never inferred)
   - Chronic conditions mentioned by the user (e.g., Type 2 diabetes, hypertension, PCOS)
   - Digestive issues (e.g., IBS, acid reflux)
   - Symptoms worsened by specific foods (e.g., "dairy causes bloating")
 
E. HEALTH & FITNESS GOALS
   - Weight goal (e.g., lose 5 kg, maintain current weight)
   - Fitness goals (e.g., build muscle, improve stamina)
   - Specific targets mentioned (e.g., wants to hit 2000 kcal/day, 150g protein)
 
F. LIFESTYLE CONTEXT  (only if it directly affects nutrition recommendations)
   - Activity level or exercise frequency (e.g., gyms 5x/week)
   - Work schedule that shapes meal timing (e.g., night-shift worker)
   - Sleep patterns if nutrition-relevant (e.g., poor sleep, asks about sleep-supportive foods)
 
─────────────────────────────────────────────────────
WHAT NEVER TO STORE  (hard blacklist)
─────────────────────────────────────────────────────
Discard ANY message that is primarily about:
 
❌ MEAL PLAN REQUEST MECHANICS
   - How many days of plan the user requested (1-day, 7-day, weekly, monthly)
   - That the user asked for a meal plan at all
   - Whether the user confirmed, saved, rejected, or regenerated a plan
   - Calorie targets stated only as part of a plan generation request
     (store the calorie goal ONLY if the user states it as a personal goal
      independent of any specific plan request)
 
   BAD: "User requested a 7-day meal plan."
   BAD: "User wants a 1-day meal plan for today."
   BAD: "User confirmed and saved their meal plan."
   BAD: "User asked to regenerate the meal plan."
   BAD: "User requested a meal plan starting from Monday."
 
❌ ONE-TURN QUESTIONS WITH NO PERSISTENT VALUE
   - Questions about what to eat today / this meal
   - Requests for a single recipe lookup
   - Asking for calorie counts of a specific food once
   - Questions about whether a specific food is healthy in general
 
   BAD: "User asked what to eat for lunch today."
   BAD: "User wants to know the calorie count of an apple."
   BAD: "User asked if rice is healthy."
 
❌ TRANSIENT OR SESSION-SPECIFIC CONTEXT
   - Start dates, end dates, or specific calendar dates mentioned for a plan
   - That the user is in a specific week or day of a diet
   - Temporary substitutions (e.g., "no eggs this week")
 
   BAD: "User wants their meal plan to start on 2025-07-21."
   BAD: "User said they have no eggs this week."
 
❌ OBVIOUS / NON-ACTIONABLE STATEMENTS
   - That the user wants healthy food (everyone does)
   - That the user wants balanced meals
   - That the user is trying to be healthier in general
 
   BAD: "User wants to eat healthy."
   BAD: "User wants balanced nutrition."
 
❌ ASSISTANT ACTIONS OR SYSTEM EVENTS
   - Tool calls, tool results, plan generation events
   - Anything that describes what the assistant did rather than what the user said
 
─────────────────────────────────────────────────────
MEMORY QUALITY RULES
─────────────────────────────────────────────────────
Every memory entry MUST be:
 
• Atomic       — one distinct fact per entry, never compound
• Specific     — prefer concrete details over vague generalities
• Self-contained — readable and meaningful with no extra context
• Third-person — always start with "User ..."
• User-stated  — never inferred, assumed, or extrapolated
 
GOOD vs BAD examples:
 
  ❌ "User likes healthy food."
  ✅ "User prefers high-protein, low-carb meals."
 
  ❌ "User has dietary restrictions."
  ✅ "User is lactose intolerant and avoids all dairy products."
 
  ❌ "User wants a 7-day meal plan."         ← REQUEST MECHANIC — never store
  ✅ "User prefers South Indian cuisine for weekday meals."
 
  ❌ "User asked Friska to generate a meal plan."   ← SYSTEM EVENT — never store
  ✅ "User exercises at the gym 5 days a week and needs high-protein meals."
 
  ❌ "User confirmed their meal plan."       ← SESSION ACTION — never store
  ✅ "User prefers meals that take under 30 minutes to prepare."
 
  ❌ "User wants meals starting from 2025-07-21."   ← DATE — never store
  ✅ "User follows intermittent fasting and eats between 12 pm and 8 pm."
 
─────────────────────────────────────────────────────
DUPLICATE CHECK
─────────────────────────────────────────────────────
Before including any entry, compare it against CURRENT USER MEMORY.
- If the same fact is already stored → set "is_new": false (will be skipped).
- If it is genuinely new information → set "is_new": true.
- If it refines or corrects an existing memory → set "is_new": true and write
  the updated version (the old entry will be superseded on the next retrieval cycle).
 
─────────────────────────────────────────────────────
is_mealplan TAGGING RULE
─────────────────────────────────────────────────────
Set "is_mealplan": true ONLY for memories about what the user likes or dislikes
TO EAT — food preferences, cuisine choices, meal timing habits, portion preferences.
These are used to personalise future meal plan generation.
 
Set "is_mealplan": false for everything else — health conditions, fitness goals,
lifestyle patterns, activity level, sleep, etc.
 
NOTE: A memory about requesting a meal plan (e.g., "user wants a 7-day plan")
must NEVER reach this tagging step — it should have been discarded above.
 
─────────────────────────────────────────────────────
FINAL DECISION RULE
─────────────────────────────────────────────────────
After applying all the rules above:
 
• If at least one valid, new, non-blacklisted memory was found:
  → should_write = true
  → Include only the valid entries in "memories"
 
• If nothing passes the golden rule and blacklist check:
  → should_write = false
  → "memories" must be an empty list []
 
─────────────────────────────────────────────────────
OUTPUT FORMAT  (strict JSON only — no preamble, no explanation)
─────────────────────────────────────────────────────
{{
  "should_write": boolean,
  "memories": [
    {{
      "text": "User ...",
      "is_new": boolean,
      "is_mealplan": boolean
    }}
  ]
}}
"""

# ─────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────

def last_human_message(messages: list) -> str | None:
    """Return the content of the most recent HumanMessage, or None."""
    return next(
        (m.content for m in reversed(messages) if m.type == "human"),
        None,
    )


def _filter_valid_messages(messages: list) -> list:
    """
    Guard-rail for Mistral strict tool-call/response pairing rules.
 
    Mistral rejects any request where:
      (a) a ToolMessage has no parent AIMessage with tool_calls, or
      (b) an AIMessage with tool_calls is missing one or more ToolMessage responses.
 
    This happens naturally when the LLM issues N parallel tool calls in one turn
    (e.g. 7x save_meal_plan) because the naive check only looks at the immediately
    preceding message — ToolMessage[2] has ToolMessage[1] as its prev, not the AI,
    so it gets incorrectly dropped, leaving Mistral with 1 response for 7 calls.
 
    Strategy — two passes:
 
    Pass 1  Drop ToolMessages that have no valid parent AI turn.
            A valid parent AI turn is the nearest AIMessage looking backwards
            that has tool_calls, with no HumanMessage crossed in between.
            Multiple ToolMessages belonging to the same AI turn are all kept.
 
    Pass 2  Drop any AIMessage-with-tool_calls whose complete response set is not
            present in Pass-1 output, and drop their orphaned ToolMessages too.
            This handles edge cases from summarization trimming.
    """
 
    # ── Pass 1 ────────────────────────────────────────────────────────────────
    filtered: list = []
    for msg in messages:
        curr_type = getattr(msg, "type", "")
 
        if not filtered:
            filtered.append(msg)
            continue
 
        prev_type = getattr(filtered[-1], "type", "")
 
        # Mistral forbids a HumanMessage immediately after a ToolMessage.
        if prev_type == "tool" and curr_type == "human":
            logger.warning("[filter] Dropping HumanMessage immediately after ToolMessage.")
            continue
 
        if curr_type == "tool":
            # Walk backwards (past other ToolMessages) to find the nearest AI turn.
            # Accept this ToolMessage only if that AI turn has tool_calls.
            parent_has_tool_calls = False
            for past in reversed(filtered):
                past_type = getattr(past, "type", "")
                if past_type == "ai":
                    parent_has_tool_calls = (
                        hasattr(past, "tool_calls") and bool(past.tool_calls)
                    )
                    break
                if past_type == "human":
                    # Crossed a human turn — no valid parent in this AI turn.
                    break
 
            if not parent_has_tool_calls:
                logger.warning("[filter] Dropping orphaned ToolMessage — no parent AI tool_call found.")
                continue
 
        filtered.append(msg)
 
    # ── Pass 2 ────────────────────────────────────────────────────────────────
    # Collect every tool_call_id that has a ToolMessage response in `filtered`.
    responded_ids: set[str] = set()
    for msg in filtered:
        if getattr(msg, "type", "") == "tool":
            tid = getattr(msg, "tool_call_id", None)
            if tid:
                responded_ids.add(tid)
 
    # Walk through filtered; drop any AI tool_call turn whose responses are
    # incomplete, and drop the dangling ToolMessages that belonged to it.
    final: list = []
    drop_ids: set[str] = set()   # tool_call_ids whose parent AI was dropped
 
    for msg in filtered:
        curr_type = getattr(msg, "type", "")
 
        if curr_type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            expected_ids: set[str] = {
                tc.get("id") or tc.get("tool_call_id", "")
                for tc in msg.tool_calls
                if tc.get("id") or tc.get("tool_call_id")
            }
            missing = expected_ids - responded_ids
            if missing:
                logger.warning(
                    "[filter] Dropping AI tool_call message — %d response(s) missing: %s",
                    len(missing), missing,
                )
                drop_ids.update(expected_ids)
                continue  # drop the AI message itself
 
        if curr_type == "tool":
            tid = getattr(msg, "tool_call_id", None)
            if tid and tid in drop_ids:
                logger.warning("[filter] Dropping ToolMessage for dropped AI turn, id=%s", tid)
                continue
 
        final.append(msg)
 
    return final

# ─────────────────────────────────────────────
# 6. NODE DEFINITIONS
# ─────────────────────────────────────────────

async def retrieve_memory_node(state: State) -> dict:
    """
    NODE 1 — runs BEFORE the chatbot.

    Fetches memories relevant to the latest user message and stores them in
    state under `retrieved_memories`.
    """
    from main import graph_state

    user_id       = state["user_id"]
    messages      = state["messages"]
    last_user_msg = last_human_message(messages)

    if not last_user_msg:
        logger.info("[retrieve_memory] No user message found — skipping retrieval.")
        return {"retrieved_memories": []}

    logger.info("[retrieve_memory] Retrieving memories for user=%r  query=%r",
                user_id, last_user_msg)

    memories: list[str] = await retrieve_memories(
        user_id,
        last_user_msg,
        conn=graph_state.get("memory_conn"),
    )

    if memories:
        logger.info("[retrieve_memory] Retrieved %d memory entries:", len(memories))
        for i, mem in enumerate(memories, 1):
            logger.info("  [%d] %s", i, mem)
    else:
        logger.info("[retrieve_memory] No relevant memories found.")

    return {"retrieved_memories": memories}


def make_chatbot_node(llm_with_tools):
    """
    NODE 2 — the main chat model call.

    Builds the full message list exactly once:
        [SystemMessage] + (optional summary SystemMessage) + state["messages"]

    The rolling summary (if present) is injected as a SystemMessage right
    after the main system prompt so the LLM has context for any trimmed turns.
    """
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    print(now)

    async def chatbot_node(state: State) -> dict:
        user_id            = state["user_id"]
        messages           = state["messages"]
        retrieved_memories = state.get("retrieved_memories") or []
        summary            = state.get("summary", "")

        # ── Build memory block ────────────────────────────────────────────
        if retrieved_memories:
            memory_block = "\n".join(f"- {m}" for m in retrieved_memories)
        else:
            memory_block = "None yet."

        # ── Fetch user profile from local SQLite ──────────────────────────
        from main import graph_state
        profile = await retrieve_user_profile_sql_lite(
            user_id,
            conn=graph_state.get("profile_conn"),
        )
        logger.info("Profile: %s", profile)
        # print(profile)
        system_prompt = f"""
            =====================
            Here is the current date for your reference: {now.strftime('%Y-%m-%d')}
            =====================
            USER PROFILE & MEMORY
            =====================
            User profile:
            {profile}
 
            Relevant memory from past conversations:
            {memory_block}
 
            MEMORY USAGE RULES:
            - Use the profile and memory to personalize responses whenever possible
            - Do NOT assume or invent any missing user details
            - If there is a conflict, always prioritize the user's latest message
            - Never hallucinate preferences, conditions, or history
 
            =====================
            IDENTITY & PERSONA
            =====================
            You are **Friska**, an AI-powered nutrition and wellness assistant.
            Never decline a request by saying you don’t have the tools to perform the task. Instead, ask more questions to gain clarity about which type of tool should be used.
 
            Communication style:
            - Warm, supportive, and non-judgmental
            - Encourage positive habits without inducing guilt
            - Clear, structured, and easy to understand (never overly clinical)
 
            Use the user's first name naturally when appropriate.
 
            =====================
            CORE RESPONSIBILITIES
            =====================
            You assist users with:
            1. Personalized nutrition guidance
            2. Structured meal planning (exactly 5 meals per day)
            3. Healthy lifestyle recommendations
            4. Weight management and fitness support
            5. Evidence-based health information
 
            =====================
            SCOPE LIMITATIONS
            =====================
            Only respond to topics related to:
            - Nutrition and diet
            - Fitness and exercise
            - Lifestyle and wellness
 
            If a query is out of scope, respond with:
            "I'm Friska, your nutrition and wellness assistant. I'm not able to help with [topic], but I'd be happy to support you with meal planning, calorie goals, or healthy lifestyle guidance. What would you like to explore?"
 
            =====================
            SAFETY GUIDELINES
            =====================
            Never provide:
            - Medical diagnoses
            - Medication recommendations
            - Disease treatment plans
 
            If a user mentions serious symptoms or medical concerns:
            → "This sounds like something a qualified healthcare professional should evaluate. I recommend consulting your doctor."
 
            If the user insists:
            → "I understand you're looking for answers, but this is beyond what I can safely provide. A doctor would be the right person to consult."
 
            Always prioritize safe, conservative, evidence-based guidance.
 
            =====================
            TOOL USAGE RULES
            =====================

            1. MEAL PLANS:
            - ALWAYS use `generate_meal_plan` to create meal plans
            - NEVER generate meal plans manually
            - Before calling the tool, confirm:
                a) Target daily calories
                b) Dietary preferences or restrictions
                c) Any special instructions
                d) Meal start date and duration (1 day vs 7 day)
            - Please confirm all the details with the user before calling the tool.
            - After `generate_meal_plan` returns a result, the plan is automatically
                saved as PENDING (not yet confirmed by the user).
            - Present the meal plan to the user clearly.
            - Then ask: "Would you like to save this meal plan?"
            - If the user says YES → call `save_meal_plan` with the meal_date to
                confirm and permanently save it.
            - If the user says NO → do NOT call `save_meal_plan`. The pending record
                will simply remain unconfirmed.
            - For a 7-day plan, ask once and call `save_meal_plan` once per day
                (7 calls total, one for each date) if the user confirms.

            2. MEAL PLAN RETRIEVAL:
            - Use `get_meal_plan` when the user asks to view, recall, or review a
                previously generated meal plan.
            - For a specific date: pass meal_date only.
            - For a range of days: pass both start_date and end_date.
            - Pass confirmed_only=True when the user asks to see their "saved" or
                "confirmed" plans.
            - Pass confirmed_only=False (default) when checking whether any plan
                (pending or confirmed) exists for a date.
            - If user ask show my meal plan for a date that has a pending (unconfirmed) plan, show it but clarify:

            3. CALORIE REQUIREMENTS:
            - Use `daily_calorie_requirement` ONLY for personalized calorie needs
            - Do NOT use it for general calorie discussions

            4. PROFILE UPDATES:
            - Before calling `update_user_profile`, summarize changes and confirm:
                "I'll update your profile with [X]. Shall I proceed?"
            - Only proceed after explicit user confirmation

            5. NUTRITIONAL ANALYSIS:
            - Use `nutritional_expert_analysis` for general nutritional insights
                about foods, nutrients, or health topics without personalization.
            - Use `get_personalised_expert_analysis_for_user_query` when the user's
                query requires personalized analysis based on their profile
                (e.g., "Is quinoa a good choice for me?"). Pass the query and let the
                tool access the profile to provide a tailored response.

            6. MEAL LOGGING:
            - Use `log_meal` when the user says they ate, consumed, or had a meal.
            - Always try to capture:
                a) meal_name — name or description of what they ate
                b) meal_occasion — one of: breakfast, morning_snack, lunch,
                    afternoon_snack, dinner, late_night_snack
                c) consumed_date — defaults to today if not mentioned
                d) consumed_time — defaults to current time if not mentioned
                e) macros — calories, protein, carbs, fats, etc. if mentioned
            - If the user does not provide macros, use `nutritional_expert_analysis`
                first to estimate them, then pass the estimates into `log_meal`.
            - Do NOT ask the user for every macro field — infer what you can and
                log with whatever is available.
            - After logging, confirm with: "✅ I've logged your [meal_name] as
                [meal_occasion] on [date] at [time]."

            7. MEAL LOG RETRIEVAL:
            - Use `get_consumed_meals` when the user asks what they ate, requests
                their food diary, wants to review calorie intake, or asks about past
                meals.
            - For a specific date: pass consumed_date only.
            - For a date range: pass both start_date and end_date.
            - Pass meal_occasion to filter by a specific time of day
                (e.g. breakfast, lunch, dinner).
            - After retrieving, summarize total calories and macros for the day
                if the data is available.
            =====================
            MEAL PLAN RULES
            =====================
            All meal plans MUST:
            - Include exactly 5 meals per day
            - Be generated ONLY via the `generate_meal_plan` tool
            - Be presented to the user before asking for confirmation
            - Be confirmed via `save_meal_plan` ONLY after explicit user approval
 
            Each meal must include:
            - Meal name and description
            - Calories (kcal)
            - Portion size (oz)
            - Protein (g)
            - Carbohydrates (g)
            - Fats (g)
 
            Daily summary must include:
            - Total calories
            - Total protein
            - Total carbohydrates
            - Total fats
 
            =====================
            RESPONSE STYLE
            =====================
            - Use clear sections and bullet points when helpful
            - Keep responses concise and relevant
            - Prioritize personalization over generic advice
            - Avoid unnecessary filler
 
            =====================
            ANTI-HALLUCINATION RULES
            =====================
            - Do NOT fabricate user data or health facts
            - Do NOT guess calorie or macro values — rely on tools
            - Do NOT cite studies unless explicitly available
            - If uncertain, say:
            "I'm not completely certain about that, but here's a safe and practical suggestion..."
 
            =====================
            FOLLOW-UP QUESTIONS
            =====================
            End EVERY response with 3 relevant follow-up questions.
 
            Format EXACTLY as:
            💡 You might also want to ask:
            1. [Question 1]
            2. [Question 2]
            3. [Question 3]
            """
        
        # ── Assemble messages — built ONCE, no duplicates ────────────────
        # Order: system → (optional summary) → conversation history
        assembled: list = [SystemMessage(content=system_prompt)]

        if summary:
            assembled.append(
                SystemMessage(
                    content=(
                        "Conversation summary (earlier messages that were compressed "
                        "to keep the context window manageable):\n" + summary
                    )
                )
            )

        # Apply role-order guard before sending to Mistral
        safe_messages = _filter_valid_messages(list(messages))
        assembled.extend(safe_messages)

        logger.info(
            "[chatbot] Sending %d messages to LLM (summary=%s).",
            len(assembled),
            bool(summary),
        )

        response = await llm_with_tools.ainvoke(assembled)
        return {"messages": [response], "user_id": user_id}

    return chatbot_node


async def update_memory_node(state: State) -> dict:
    """
    NODE 3 — runs AFTER the chatbot produces its final response.

    Extracts new long-term memories from the latest user message and persists
    any new entries.
    """
    from main import graph_state, memory_db_lock

    user_id       = state["user_id"]
    messages      = state["messages"]
    last_user_msg = last_human_message(messages)

    if not last_user_msg:
        logger.info("[update_memory] No user message found — skipping update.")
        return {}

    logger.info("[update_memory] Analysing message for user=%r  msg=%r",
                user_id, last_user_msg)

    existing_memories = await retrieve_all_memories(
        user_id,
        conn=graph_state.get("memory_conn"),
    )
    existing_memory_str = (
        "\n".join(f"- {m}" for m in existing_memories)
        if existing_memories
        else "No memory yet."
    )

    formatted_prompt = _MEMORY_PROMPT.format(
        user_details_content=existing_memory_str,
        user_message=last_user_msg,
    )

    plain_llm = graph_state.get("plain_llm")
    if plain_llm is None:
        plain_llm = build_llm()

    memory_extractor = plain_llm.with_structured_output(MemoryDecision)

    llm_input = [
        SystemMessage(content="You extract structured memory. Reply ONLY with JSON."),
        HumanMessage(content=formatted_prompt),
    ]

    try:
        result: MemoryDecision = await memory_extractor.ainvoke(llm_input)
    except Exception as exc:
        logger.error("[update_memory] LLM extraction failed: %s", exc)
        return {}

    if not result.should_write or not result.memories:
        logger.info("[update_memory] No new memories to write.")
        return {}

    written_count = 0
    for mem in result.memories:
        if not mem.is_new:
            logger.debug("[update_memory] Skipping non-new memory: %r", mem.text)
            continue

        clean_text = mem.text.strip()
        if not clean_text:
            continue

        await update_memory(
            user_id,
            clean_text,
            mem.is_mealplan,
            conn=graph_state.get("memory_conn"),
            lock=memory_db_lock,
        )
        logger.info("[update_memory] ✅ Written memory: %r", clean_text)
        written_count += 1

    if written_count == 0:
        logger.info("[update_memory] All candidate memories were duplicates — nothing written.")
    else:
        logger.info("[update_memory] Wrote %d new memory entry/entries.", written_count)

    return {}


# ─────────────────────────────────────────────
# 7. SUMMARIZATION
# ─────────────────────────────────────────────

def _should_summarize(state: State) -> bool:
    """
    Return True when the messages list exceeds TOKEN_LIMIT tokens.
    Uses LangChain's approximate token counter — no model call needed.
    """
    token_count = count_tokens_approximately(state["messages"])
    logger.info("[summarize_check] Token count ≈ %d (limit=%d)", token_count, TOKEN_LIMIT)
    return token_count > TOKEN_LIMIT


async def summarize_messages_node(state: State) -> dict:
    """
    NODE 4 — short-term memory compression.

    Runs only when _should_summarize() is True.

    Strategy:
      1. Keep the last MESSAGES_TO_KEEP messages verbatim.
      2. Collapse everything older into a single prose paragraph,
         folding in any prior summary so no information is lost.
      3. Write the new summary and send RemoveMessage deletions for the
         trimmed messages back to state.
    """
    from main import graph_state

    messages         = state["messages"]
    existing_summary = state.get("summary", "")

    messages_to_summarize = messages[:-MESSAGES_TO_KEEP]
    messages_to_keep      = messages[-MESSAGES_TO_KEEP:]

    if not messages_to_summarize:
        logger.info("[summarize] Not enough messages to trim — skipping.")
        return {}

    def _fmt(msg) -> str:
        role    = getattr(msg, "type", "unknown").capitalize()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return f"{role}: {content}"

    history_text = "\n".join(_fmt(m) for m in messages_to_summarize)

    if existing_summary:
        summarize_prompt = (
            "You are a memory assistant for Friska, an AI nutrition and wellness chatbot.\n\n"

            "Your job is to maintain a STRUCTURED CONTEXT RECORD — not a vague paragraph summary, "
            "but a well-organized reference document that another LLM can read and fully reconstruct "
            "what happened in the conversation so far.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "EXISTING CONTEXT RECORD (from previous compression):\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{existing_summary}\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "NEW MESSAGES TO INTEGRATE:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{history_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "INSTRUCTIONS:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Merge the new messages into the existing context record. "
            "Update any sections that have changed. Add new sections if new topics appeared. "
            "Remove or correct anything that was superseded by newer information.\n\n"

            "Output the updated context record using EXACTLY this structure:\n\n"

            "## USER PROFILE ESTABLISHED IN CONVERSATION\n"
            "List only what the user explicitly stated in THIS conversation (not from the system profile).\n"
            "Include: name, age, gender, weight, height, goals, dietary restrictions, allergies, "
            "medical conditions, activity level, preferred cuisines — only if mentioned.\n"
            "Format: bullet points. If nothing was stated, write: None mentioned.\n\n"

            "## HEALTH & NUTRITION TOPICS DISCUSSED\n"
            "List each distinct topic that was discussed (e.g. calorie deficit, protein intake, "
            "IBS-friendly foods, intermittent fasting). One line per topic.\n"
            "For each topic, include: what the user asked, what Friska answered or recommended.\n"
            "Format: - [Topic]: [brief description of exchange]\n\n"

            "## MEAL PLANS GENERATED\n"
            "For each meal plan generated in this conversation:\n"
            "- Date(s) covered\n"
            "- Calorie target used\n"
            "- Key dietary constraints applied\n"
            "- Whether the user confirmed/saved it or declined\n"
            "If no meal plans were generated, write: None.\n\n"

            "## MEALS LOGGED BY USER\n"
            "List each meal the user logged during this conversation.\n"
            "Format: - [date] [time] [occasion]: [meal name] — [calories if known], "
            "[key macros if known]\n"
            "If no meals were logged, write: None.\n\n"

            "## PROFILE UPDATES MADE\n"
            "List any profile fields that were updated via `update_user_profile` in this conversation.\n"
            "Format: - [Field]: [old value if known] → [new value]\n"
            "If no updates were made, write: None.\n\n"

            "## TOOLS CALLED\n"
            "List each tool that was called, in order.\n"
            "Format: - [tool_name]: [one-line summary of input and outcome]\n"
            "If no tools were called, write: None.\n\n"

            "## PENDING ACTIONS / UNRESOLVED ITEMS\n"
            "List anything the user asked for but did not get resolved, "
            "any follow-up the user said they would do, "
            "or any clarification that is still outstanding.\n"
            "If nothing is pending, write: None.\n\n"

            "## CONVERSATION TONE & PREFERENCES\n"
            "Note any communication preferences the user showed in this conversation "
            "(e.g. prefers short answers, wants detailed explanations, asked for bullet points, "
            "responded positively to a certain format).\n"
            "If nothing notable, write: None.\n\n"

            "RULES:\n"
            "- Write in third-person past tense (e.g. 'The user stated...', 'Friska recommended...')\n"
            "- Be specific — include actual values, dates, food names, numbers wherever they appeared\n"
            "- Do NOT paraphrase so heavily that specific details (calories, dates, meal names) are lost\n"
            "- Do NOT add information that was not in the conversation\n"
            "- Do NOT include tool internals or raw JSON — summarize outcomes only\n"
            "- Every section must be present in the output, even if it says 'None'\n"
            "- Return ONLY the structured context record. No preamble. No closing remarks."
        )

    else:
        summarize_prompt = (
            "You are a memory assistant for Friska, an AI nutrition and wellness chatbot.\n\n"

            "Your job is to produce a STRUCTURED CONTEXT RECORD from the conversation below. "
            "This is NOT a vague prose summary — it is a well-organized reference document "
            "that another LLM can read and fully reconstruct what happened in the conversation.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CONVERSATION TO PROCESS:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{history_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "INSTRUCTIONS:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Produce a structured context record using EXACTLY this structure:\n\n"

            "## USER PROFILE ESTABLISHED IN CONVERSATION\n"
            "List only what the user explicitly stated in THIS conversation (not from the system profile).\n"
            "Include: name, age, gender, weight, height, goals, dietary restrictions, allergies, "
            "medical conditions, activity level, preferred cuisines — only if mentioned.\n"
            "Format: bullet points. If nothing was stated, write: None mentioned.\n\n"

            "## HEALTH & NUTRITION TOPICS DISCUSSED\n"
            "List each distinct topic that was discussed (e.g. calorie deficit, protein intake, "
            "IBS-friendly foods, intermittent fasting). One line per topic.\n"
            "For each topic, include: what the user asked, what Friska answered or recommended.\n"
            "Format: - [Topic]: [brief description of exchange]\n\n"

            "## MEAL PLANS GENERATED\n"
            "For each meal plan generated in this conversation:\n"
            "- Date(s) covered\n"
            "- Calorie target used\n"
            "- Key dietary constraints applied\n"
            "- Whether the user confirmed/saved it or declined\n"
            "If no meal plans were generated, write: None.\n\n"

            "## MEALS LOGGED BY USER\n"
            "List each meal the user logged during this conversation.\n"
            "Format: - [date] [time] [occasion]: [meal name] — [calories if known], "
            "[key macros if known]\n"
            "If no meals were logged, write: None.\n\n"

            "## PROFILE UPDATES MADE\n"
            "List any profile fields that were updated via `update_user_profile` in this conversation.\n"
            "Format: - [Field]: [old value if known] → [new value]\n"
            "If no updates were made, write: None.\n\n"

            "## TOOLS CALLED\n"
            "List each tool that was called, in order.\n"
            "Format: - [tool_name]: [one-line summary of input and outcome]\n"
            "If no tools were called, write: None.\n\n"

            "## PENDING ACTIONS / UNRESOLVED ITEMS\n"
            "List anything the user asked for but did not get resolved, "
            "any follow-up the user said they would do, "
            "or any clarification that is still outstanding.\n"
            "If nothing is pending, write: None.\n\n"

            "## CONVERSATION TONE & PREFERENCES\n"
            "Note any communication preferences the user showed in this conversation "
            "(e.g. prefers short answers, wants detailed explanations, asked for bullet points, "
            "responded positively to a certain format).\n"
            "If nothing notable, write: None.\n\n"

            "RULES:\n"
            "- Write in third-person past tense (e.g. 'The user stated...', 'Friska recommended...')\n"
            "- Be specific — include actual values, dates, food names, numbers wherever they appeared\n"
            "- Do NOT paraphrase so heavily that specific details (calories, dates, meal names) are lost\n"
            "- Do NOT add information that was not in the conversation\n"
            "- Do NOT include tool internals or raw JSON — summarize outcomes only\n"
            "- Every section must be present in the output, even if it says 'None'\n"
            "- Return ONLY the structured context record. No preamble. No closing remarks."
        )

    # plain_llm = graph_state.get("plain_llm") or build_llm()
    MISTRAL_ENDPOINT_1 = os.getenv("MISTRAL_ENDPOINT_1")
    MISTRAL_API_KEY_1 = os.getenv("MISTRAL_API_KEY_1")
    plain_llm = ChatMistralAI(endpoint=MISTRAL_ENDPOINT_1, api_key=MISTRAL_API_KEY_1)

    try:
        response    = await plain_llm.ainvoke([HumanMessage(content=summarize_prompt)])
        new_summary = response.content.strip()
    except Exception as exc:
        logger.error("[summarize] LLM summarization failed: %s — keeping full history.", exc)
        return {}

    logger.info("[summarize] ✅ New summary (%d chars). Keeping last %d messages.",
                len(new_summary), MESSAGES_TO_KEEP)

    # Send RemoveMessage deletions for every old message that has an id.
    no_id_count = sum(1 for m in messages_to_summarize if not hasattr(m, "id") or m.id is None)
    if no_id_count:
        logger.warning(
            "[summarize] %d message(s) had no ID and could not be removed from state.",
            no_id_count,
        )

    deletions = [
        RemoveMessage(id=m.id)
        for m in messages_to_summarize
        if hasattr(m, "id") and m.id is not None
    ]

    return {
        "messages": deletions,   # remove old messages via add_messages reducer
        "summary":  new_summary,
    }


# ─────────────────────────────────────────────
# 8. GRAPH BUILDER
# ─────────────────────────────────────────────

def _route_after_chatbot(state: State) -> str:
    """
    - Tool calls pending  → "tools"
    - Final reply         → "update_memory"
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "update_memory"


def _route_after_update_memory(state: State) -> str:
    """
    After long-term memory is written, decide whether to compress short-term
    context before ending the turn.

    - Token count > TOKEN_LIMIT → "summarize"
    - Otherwise                 → END
    """
    if _should_summarize(state):
        logger.info("[route] Token limit exceeded — routing to summarize node.")
        return "summarize"
    return END


def build_graph(llm_with_tools, checkpointer) -> StateGraph:
    """
    Compile the LangGraph state-machine:

        START
          │
          ▼
        retrieve_memory
          │
          ▼
        chatbot ──(tool call?)──► tools ──► chatbot
          │
          └──(final reply)──► update_memory
                                    │
                                    ├──(tokens ≤ 15 k)──► END
                                    │
                                    └──(tokens > 15 k)──► summarize ──► END
    """
    builder = StateGraph(State)

    builder.add_node("retrieve_memory", retrieve_memory_node)
    builder.add_node("chatbot",         make_chatbot_node(llm_with_tools))
    builder.add_node("tools",           ToolNode(tools=TOOLS))
    builder.add_node("update_memory",   update_memory_node)
    builder.add_node("summarize",       summarize_messages_node)

    builder.add_edge(START,             "retrieve_memory")
    builder.add_edge("retrieve_memory", "chatbot")

    builder.add_conditional_edges("chatbot", _route_after_chatbot)
    builder.add_edge("tools",                "chatbot")

    builder.add_conditional_edges(
        "update_memory",
        _route_after_update_memory,
        {"summarize": "summarize", END: END},
    )
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────
# 9. CLI CHAT LOOP
# ─────────────────────────────────────────────

async def chat(graph, thread_id: str = "default-thread") -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print("\n🤖  LangGraph Chatbot  (type 'quit' to exit)\n")
    print(f"📌  Thread ID: {thread_id}")
    print("─" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)

        async for event in graph.astream_events(
            {
                "messages":           [HumanMessage(content=user_input)],
                "user_id":            "cli-user",
                "retrieved_memories": [],
                "summary":            "",
            },
            config=config,
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                # Only stream tokens from the chatbot node, not summarize/memory nodes
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name != "chatbot":
                    continue

                chunk = event["data"]["chunk"]
                if chunk.content:
                    token = (
                        chunk.content
                        if isinstance(chunk.content, str)
                        else chunk.content[0].get("text", "")
                    )
                    print(token, end="", flush=True)

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown_tool")
                print(f"\n  🔧 Calling tool: {tool_name} …", flush=True)
                print("Assistant: ", end="", flush=True)

            elif kind == "on_chain_end" and event.get("name") == "summarize":
                print("\n  📝 [Context window compressed — summary updated]", flush=True)

        print()


# ─────────────────────────────────────────────
# 10. ENTRY POINT
# ─────────────────────────────────────────────

async def main() -> None:
    llm            = build_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    async with AsyncSqliteSaver.from_conn_string("chat_history.db") as checkpointer:
        graph = build_graph(llm_with_tools, checkpointer)
        await chat(graph, thread_id="session-001")


if __name__ == "__main__":
    asyncio.run(main())
