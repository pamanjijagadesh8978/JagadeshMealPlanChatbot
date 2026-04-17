import asyncio
import logging
import os
from typing import Annotated, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from pydantic import BaseModel, Field
from typing import TypedDict

from utils import (
    retrieve_all_memories,
    retrieve_memories,
    update_memory,
    get_user_profile,
    retrive_user_profile_sql_lite,
    clean_profile,
    workflow,
    generate_meal_plan_json,
    prompt,
    update_weight,
    update_height,
    update_activity_level,
    get_activity_level,
    update_calories,
    get_calories,
    MealPlanInstructor
)


# ─────────────────────────────────────────────
# 0. LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chatbot.memory")


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

        profile = await get_user_profile("FriskaAiCCM_HFWL", user_id)
        print(f"Profile: {profile}")
        if not profile:
            return "Error: profile not found"

        logger.info("get_profile: profile fetched for user_id=%s", user_id)

        # ── Enrich profile with computed fields BEFORE saving ────────────────
        # ✅ Fix 5: fetch these first so they are included in the INSERT
        activity_level = await get_activity_level(user_id)
        calories       = await get_calories(user_id)
        profile["activity_level"] = activity_level
        profile["calories"]       = calories

        # ── Clean and persist to local SQLite ────────────────────────────────
        cleaned = clean_profile(profile, user_id)  # must return keys below
        from main import graph_state 
        conn = graph_state.get("profile_conn")
        if conn is None:
            return "Error: Database connection not available."

        await conn.execute(
            """
            INSERT OR REPLACE INTO UserHealthProfile (
                user_id,
                Name, Age, Gender,
                WeightKG, HeightCM, WaistCircumferenceCM,
                Cuisine, ActivityLevel, Calories,
                DietaryPreference, Restrictions, DigestiveIssues,
                Allergies, SymptomAggravatingFoods,
                HeartRate, BloodPressure, BodyTemperature,
                BloodOxygen, RespiratoryRate,
                MedicalConditions, Goals,
                ModifiedDate
            ) VALUES (
                :user_id,
                :name, :age, :gender,
                :weight_kg, :height_cm, :waist_circumference_cm,
                :cuisine, :activity_level, :calories,
                :dietary_preference, :restrictions, :digestive_issues,
                :allergies, :symptom_aggravating_foods,
                :heart_rate, :blood_pressure, :body_temperature,
                :blood_oxygen, :respiratory_rate,
                :medical_conditions, :goals,
                datetime('now')
            )
            """,
            cleaned,
        )
        await conn.commit()
        logger.info("get_profile: profile for user_id=%s saved to local SQLite.", user_id)
        logger.info("Profile Type: %s, Profile: %s", type(profile), profile)
        return str(profile)

    except Exception as e:
        logger.error("get_profile error: %s", str(e))
        return f"Error fetching profile: {str(e)}"

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
        height_cm:                    Height in inches
        waist_circumference_cm:       Waist circumference in inches
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

        # ✅ Fixed: use `is not None` to correctly handle 0.0 values
        if weight_kg is not None:
            await update_weight(weight_kg, user_id)
        if height_cm is not None:
            await update_height(height_cm, user_id)
        if activity_level is not None:
            await update_activity_level(activity_level, user_id)
        if calories is not None:
            await update_calories(calories, user_id)

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

        return generate_meal_plan_json(
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
        )

    # if days == 7 and meal_plan_instructions_by_day is None:
    #     return {
    #         "error": True,
    #         "message": (
    #             "days = 7 requires meal_plan_instructions_by_day (list length == 7). "
    #             "Provide it, or provide meal_plan_instructions to apply to all days."
    #         ),
    #         "expected": {"meal_plan_instructions_by_day_length": days},
    #         "received": {"meal_plan_instructions_by_day": None},
    #     }

    # if not isinstance(meal_plan_instructions_by_day, list):
    #     raise TypeError("meal_plan_instructions_by_day must be a list when provided.")

    # if len(meal_plan_instructions_by_day) != days:
    #     raise ValueError(
    #         f"meal_plan_instructions_by_day must have exactly {days} items, "
    #         f"but got {len(meal_plan_instructions_by_day)}."
    #     )
    
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
    _instructions = await MealPlanInstructor(user_id, conn, profile)
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
    return result["final7daymealplan"]


TOOLS = [get_profile, update_user_profile, daily_calorie_requirement, generate_meal_plan]


# ─────────────────────────────────────────────
# 3. LLM + TOOL BINDING
# ─────────────────────────────────────────────

def build_llm() -> ChatMistralAI:
    """
    Create the ChatMistralAI instance from environment variables.
    Set MISTRAL_ENDPOINT and MISTRAL_API_KEY before running.
    """
    MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT")
    MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")

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
    You are a long-term memory extraction engine for a health and nutrition assistant.

    Your role is to extract meaningful, persistent user information that can improve personalized nutrition, meal planning, and lifestyle guidance.

    -------------------------
    CURRENT USER MEMORY:
    {user_details_content}

    LATEST USER MESSAGE:
    {user_message}
    -------------------------

    HEALTH-FOCUSED EXTRACTION RULES:

    1. Extract ONLY long-term, stable, and reusable information related to:

    A. DIET & FOOD PREFERENCES
    - Foods the user likes or dislikes
    - Favorite cuisines or meals
    - Eating habits (e.g., "skips breakfast", "eats late at night")

    B. DIETARY RESTRICTIONS
    - Allergies (e.g., nuts, dairy)
    - Intolerances (e.g., lactose intolerance)
    - Diet types (e.g., vegetarian, vegan, keto)

    C. HEALTH & FITNESS GOALS
    - Weight loss, muscle gain, maintenance
    - Fitness goals or routines
    - Lifestyle improvements

    D. HEALTH CONDITIONS (NON-DIAGNOSTIC ONLY)
    - User-mentioned conditions (e.g., "has diabetes", "high BP")
    - Only store if explicitly stated by the user
    - DO NOT infer or assume

    E. LIFESTYLE PATTERNS
    - Sleep habits
    - Activity levels
    - Work schedule affecting meals

    F. SKILL / KNOWLEDGE LEVEL
    - Beginner/intermediate/advanced in fitness or nutrition

    -------------------------

    DO NOT STORE:
    - One-time questions
    - Temporary context (e.g., "What should I eat today?")
    - Generic or obvious statements
    - Assistant responses
    - Medical advice or inferred conditions

    -------------------------

    MEMORY QUALITY RULES:

    - Each memory must be:
    • Atomic (one idea per entry)
    • Clear and self-contained
    • Written in third person
    • Start with "User ..."

    - Prefer SPECIFIC over generic:
    ❌ "User likes healthy food"
    ✅ "User prefers high-protein vegetarian meals"

    - Avoid duplicates or near-duplicates from CURRENT USER MEMORY
    - Only store if it adds NEW value

    -------------------------

    DECISION RULE:
    - If NO useful long-term memory is found:
    → should_write = false
    → Tag: Is this memory related to meal plan or not

    -------------------------

    ADDITIONAL TAGGING RULE:

    - For each memory, include a boolean field "is_mealplan"
    - Set "is_mealplan": true if the memory is directly related to:
    • Specific meals or meal plans
    • Daily/weekly food plans
    • Structured eating schedules

    - Otherwise set "is_mealplan": false

    -------------------------

    OUTPUT FORMAT (STRICT JSON ONLY):

    {{
    "should_write": boolean,
    "memories": [
        {{
        "text": "User ...",
        "is_new": true,
        "is_mealplan": boolean
        }}
    ]
    }}

    -------------------------

    IMPORTANT:
    - Do NOT include explanations
    - Do NOT include extra text
    - Return ONLY valid JSON
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
    Ensure the message list conforms to Mistral's strict role-ordering rules:
      - Must start with a human message (after system).
      - tool result messages must be immediately preceded by an AI message
        that contains the corresponding tool call.
      - No 'user' message immediately after a 'tool' message.

    Strategy: walk the list and drop any message that would violate ordering.
    This is a safety net; the root cause (duplicate messages) is fixed separately.
    """
    filtered = []
    for msg in messages:
        if not filtered:
            filtered.append(msg)
            continue

        prev = filtered[-1]
        prev_type = getattr(prev, "type", "")
        curr_type = getattr(msg, "type", "")

        # Mistral: after a tool message the next must be ai/assistant
        if prev_type == "tool" and curr_type == "human":
            logger.warning(
                "_filter_valid_messages: dropping human message immediately after tool message "
                "to preserve Mistral role ordering. This should not happen — check for duplicates."
            )
            continue

        filtered.append(msg)

    return filtered


# ─────────────────────────────────────────────
# 6. NODE FUNCTIONS
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
        profile = await retrive_user_profile_sql_lite(
            user_id,
            conn=graph_state.get("profile_conn"),
        )
        logger.info("Profile: %s", profile)
        # print(profile)
        system_prompt = f"""
            =====================
            USER PROFILE & MEMORY
            =====================
            User profile:
            {profile}

            Relevant memory from past conversations:
            {memory_block}

            RULES FOR USING MEMORY:
            - Prioritize memory and profile when personalizing responses
            - Do NOT assume or invent details not present in the above
            - If memory conflicts with the current user message, prefer the current message
            - Do NOT hallucinate user preferences, conditions, or history

            =====================
            IDENTITY & PERSONA
            =====================
            You are **Friska**, an AI-powered nutrition and health assistant.

            Tone: Warm, encouraging, and non-judgmental. Motivate users without
            making them feel guilty about past habits or choices. Be clear and
            structured, but never cold or clinical.

            Address the user by their first name when appropriate.

            =====================
            CORE ROLE
            =====================
            You help users with:
            1. Personalized nutrition guidance
            2. Structured meal planning (always 5 meals/day)
            3. Healthy lifestyle recommendations
            4. Weight, fitness, and wellness support
            5. Evidence-based health information

            =====================
            SCOPE LIMITATION
            =====================
            ONLY respond to topics related to:
            - Nutrition and diet
            - Fitness and exercise
            - Lifestyle and wellness

            If a query is unrelated, respond with:
            "I'm Friska, your nutrition and wellness assistant! I'm not able to
            help with [topic], but I'd love to help you with something like meal
            planning, calorie goals, or healthy lifestyle tips. What would you
            like to explore?"

            =====================
            SAFETY GUARDRAILS
            =====================
            DO NOT provide:
            - Medical diagnoses
            - Medication prescriptions
            - Treatment plans for diseases or conditions

            For serious symptoms or medical concerns:
            → Say: "This sounds like something a qualified healthcare professional
            should evaluate. I'd encourage you to consult your doctor."

            If a user insists you diagnose or prescribe:
            → Firmly but kindly decline every time:
            "I understand you're looking for answers, but this is outside what
            I'm able to safely help with. A doctor would be the right person to
            consult here."

            Always prioritize safety and conservative, evidence-based guidance.

            =====================
            TOOL USAGE RULES
            =====================
            1. MEAL PLANS:
            - NEVER generate a meal plan without calling `generate_meal_plan`
            - Before generating, always confirm:
                a) Target daily calories
                b) Dietary preferences or restrictions
                c) Any special instructions from the user

            2. CALORIE REQUIREMENTS:
            - Call `daily_calorie_requirement` when a user asks for THEIR
                personalized daily calorie target or needs a calorie estimate
            - Do NOT call this tool for casual mentions of calories
                (e.g., "that meal has a lot of calories")

            3. PROFILE UPDATES:
            - Before calling `update_profile`, summarize the changes and confirm:
                "I'll update your profile with [X]. Shall I go ahead?"
            - Only call the tool after the user explicitly confirms

            =====================
            MEAL PLAN REQUIREMENTS
            =====================
            All meal plans must:
            - Contain exactly 5 meals per day
            - Be generated using the `generate_meal_plan` tool only

            For EACH meal, include:
            - Meal name and description
            - Calories (kcal)
            - Portion size (oz)
            - Protein (g)
            - Carbohydrates (g)
            - Fats (g)

            For the FULL DAY, include a summary of:
            - Total calories
            - Total protein (g)
            - Total carbohydrates (g)
            - Total fats (g)

            =====================
            RESPONSE STYLE
            =====================
            - Use the user's first name when it feels natural
            - Prefer bullet points and clear sections for structured content
            - Avoid generic advice when personalization is possible
            - Be concise — don't pad responses with filler

            =====================
            ANTI-HALLUCINATION RULES
            =====================
            - Do NOT invent user data, preferences, or medical facts
            - Do NOT fabricate calorie or macro values — always use tool output
            - Do NOT cite specific studies unless retrieved from memory or tools
            - If unsure: say "I'm not certain about that" and offer a safe, 
            conservative alternative

            =====================
            FOLLOW-UP QUESTIONS
            =====================
            At the end of EVERY response, suggest 3 relevant follow-up questions
            tailored to the user's current context and goals.

            Format exactly as:
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
            "You are a support assistant for Friska, a healthcare chatbot. "
            "Your goal is to reduce conversation load by maintaining a clear and concise summary "
            "of the user's interaction history.\n\n"

            "You are maintaining a rolling summary of a conversation.\n\n"

            f"EXISTING SUMMARY (previous context):\n{existing_summary}\n\n"
            f"NEW MESSAGES TO ADD:\n{history_text}\n\n"

            "Write an updated, concise summary that combines the existing summary with the new messages. "
            "Focus only on key facts, user concerns, symptoms, decisions, and relevant context. "
            "Omit filler, repetition, and unnecessary details. "

            "Write in third-person past tense. "
            "Ensure the summary is clear, structured, and easy for another assistant to understand. "

            "Return ONLY the updated summary paragraph. Do not include any extra text."
        )
        logger.info("[summarize] Extending existing summary with %d older messages.",
                    len(messages_to_summarize))
    else:
        summarize_prompt = (
            "You are a support assistant for Friska, a healthcare chatbot. "
            "Your goal is to reduce conversation load by maintaining a clear and concise summary "
            "of the user's interaction history.\n\n"
            f"CONVERSATION TO SUMMARIZE:\n{history_text}\n\n"
            "Write a concise prose summary of the conversation above. Focus on the key "
            "topics, questions, tool calls, and outcomes. Write in third-person past tense. "
            "Return ONLY the summary paragraph, no preamble."
        )
        logger.info(
            "[summarize] No prior summary found. Generating fresh summary from %d messages.",
            len(messages_to_summarize),
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