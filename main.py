import os
import requests
import re
import logging
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import json
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scholarship_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Database Model for User Details
class UserDetails(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    query = db.Column(db.String(1000), nullable=False)
    caste = db.Column(db.String(50), nullable=True)
    income = db.Column(db.String(50), nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    course_level = db.Column(db.String(50), nullable=True)
    is_hostel = db.Column(db.Boolean, nullable=True)
    cgpa = db.Column(db.Float, nullable=True)
    is_minority = db.Column(db.Boolean, nullable=True)
    has_disability = db.Column(db.Boolean, nullable=True)
    ex_serviceman_parent = db.Column(db.Boolean, nullable=True)
    scholarship_type = db.Column(db.String(50), nullable=False, default='unspecified')
    intent = db.Column(db.String(50), nullable=True)  # Store detected intent
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<UserDetails {self.id}>"

# Create database tables
with app.app_context():
    db.create_all()

# List of fallback questions
FALLBACK_QUESTIONS = [
    "Would you like to know about the application process for these scholarships?",
    "Can you share your caste category (SC/ST/OBC/General/Minority) for more tailored results?",
    "What is your annual family income range? (e.g., below ₹1L, ₹1-2.5L, ₹2.5-8L, above ₹8L)",
    "Are you a girl student? This helps identify gender-specific scholarships.",
    "Are you currently staying in a hostel?",
    "What is your current CGPA or percentage?",
    "Do you belong to any minority community? (Muslim/Christian/Sikh/etc.)",
    "Do you have any disabilities?",
    "Are your parents ex-servicemen or freedom fighters?",
    "Would you like information about government, private, NGO, or college-specific scholarships?"
]

def lemmatize_text(text):
    words = text.split()
    return ' '.join(lemmatizer.lemmatize(word) for word in words)

def extract_user_details_with_gemini(user_input):
    """
    Use Gemini API to extract user details from the input.
    Returns a dictionary with extracted fields.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        return {}
    
    prompt = f"""
Analyze the following user query about scholarships and extract relevant details in JSON format.
Return ONLY the JSON object with the extracted fields, nothing else.

Fields to extract:
- caste: SC, ST, OBC, General, Minority (or null if not mentioned)
- income: Annual family income if mentioned (e.g., "1.5 lakh", "below 2.5L")
- gender: male, female (or null if not mentioned)
- course_level: ug (undergraduate), pg (postgraduate) (or null if not mentioned)
- is_hostel: true if staying in hostel, false if not, null if not mentioned
- cgpa: Numeric CGPA or percentage if mentioned
- is_minority: true if from minority community, false or null otherwise
- has_disability: true if has disability, false or null otherwise
- ex_serviceman_parent: true if parent is ex-serviceman, false or null otherwise
- scholarship_type: government, private, ngo, college (or null if not mentioned)

Example output for "I'm an SC girl with 1.5L income looking for government scholarships":
{{
  "caste": "sc",
  "income": "1.5 lakh",
  "gender": "female",
  "scholarship_type": "government"
}}

Now analyze this query:
"{user_input}"
"""

    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 1,
            "topP": 1.0,
            "maxOutputTokens": 500,
        }
    }
    params = {'key': GEMINI_API_KEY}

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            params=params,
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            content = response.json()
            if 'candidates' in content and len(content['candidates']) > 0:
                response_text = content['candidates'][0]['content']['parts'][0]['text'].strip()
                try:
                    # Extract JSON from the response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_str = response_text[json_start:json_end]
                    extracted_data = json.loads(json_str)
                    logger.debug(f"Extracted details from Gemini: {extracted_data}")
                    return extracted_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Gemini response as JSON: {response_text}")
                    return {}
            else:
                logger.error("No response from Gemini API for entity extraction")
                return {}
        else:
            error_message = f'Gemini API error: {response.status_code}'
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    error_message += f" - {error_detail['error'].get('message', 'Unknown error')}"
            except:
                error_message += f" - {response.text[:200]}"
            logger.error(error_message)
            return {}
    except requests.exceptions.Timeout:
        logger.error("Gemini API request timeout for entity extraction")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during entity extraction: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error during entity extraction: {str(e)}")
        return {}

def analyze_query_type(user_input):
    """
    Use Gemini API to classify the user input into one of the predefined intents.
    """
    user_input_lower = user_input.lower().strip()
    logger.debug(f"Analyzing query type for input: {user_input_lower} using Gemini API")

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        return 'general_conversation'  # Fallback intent

    # Define the possible intents
    possible_intents = [
        'greeting',
        'bot_info',
        'bot_functionality',
        'casual',
        'scholarship_query',
        'scholarship_personalized',
        'scholarship_types',
        'scholarship_type_selection',
        'general_conversation'
    ]

    # Create prompt for Gemini API to classify intent
    prompt = f"""
Classify the following user input into one of these intents: {', '.join(possible_intents)}.
Return only the intent name, nothing else.

Input: "{user_input_lower}"

Intent descriptions for reference:
- greeting: User says hi, hello, or similar greetings.
- bot_info: User asks about what the bot can do or its capabilities.
- bot_functionality: User asks how the bot works or how to use it.
- casual: User makes casual remarks like "thanks", "okay", or small talk.
- scholarship_query: User asks general questions about scholarships or financial aid.
- scholarship_personalized: User provides personal details (e.g., caste, income, course) to find specific scholarships.
- scholarship_types: User asks about types or categories of scholarships.
- scholarship_type_selection: User explicitly selects a scholarship type (e.g., government, private, ngo, college).
- general_conversation: Any other input that doesn't fit the above categories.
"""

    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 1,
            "topP": 1.0,
            "maxOutputTokens": 50,
        }
    }
    params = {'key': GEMINI_API_KEY}

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            params=params,
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            content = response.json()
            if 'candidates' in content and len(content['candidates']) > 0:
                intent = content['candidates'][0]['content']['parts'][0]['text'].strip()
                if intent in possible_intents:
                    logger.debug(f"Gemini API detected intent: {intent}")
                    return intent
                else:
                    logger.warning(f"Gemini API returned invalid intent: {intent}. Falling back to general_conversation")
                    return 'general_conversation'
            else:
                logger.error("No response from Gemini API for intent detection")
                return 'general_conversation'
        else:
            error_message = f'Gemini API error: {response.status_code}'
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    error_message += f" - {error_detail['error'].get('message', 'Unknown error')}"
            except:
                error_message += f" - {response.text[:200]}"
            logger.error(error_message)
            return 'general_conversation'

    except requests.exceptions.Timeout:
        logger.error("Gemini API request timeout for intent detection")
        return 'general_conversation'
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during intent detection: {str(e)}")
        return 'general_conversation'
    except Exception as e:
        logger.error(f"Unexpected error during intent detection: {str(e)}")
        return 'general_conversation'

def get_missing_details(user_details):
    """Determine which important details are missing from user profile"""
    important_fields = [
        'caste', 'income', 'gender', 'course_level',
        'is_hostel', 'cgpa', 'is_minority',
        'has_disability', 'ex_serviceman_parent'
    ]
    
    missing = []
    for field in important_fields:
        if not user_details.get(field):
            missing.append(field)
    return missing

def create_fallback_question(missing_fields):
    """Create appropriate follow-up question based on missing fields"""
    if not missing_fields:
        return FALLBACK_QUESTIONS[0]  # Default question about application process
    
    # Prioritize certain questions
    if 'caste' in missing_fields:
        return FALLBACK_QUESTIONS[1]
    if 'income' in missing_fields:
        return FALLBACK_QUESTIONS[2]
    if 'gender' in missing_fields:
        return FALLBACK_QUESTIONS[3]
    if 'is_hostel' in missing_fields:
        return FALLBACK_QUESTIONS[4]
    if 'cgpa' in missing_fields:
        return FALLBACK_QUESTIONS[5]
    if 'is_minority' in missing_fields:
        return FALLBACK_QUESTIONS[6]
    if 'has_disability' in missing_fields:
        return FALLBACK_QUESTIONS[7]
    if 'ex_serviceman_parent' in missing_fields:
        return FALLBACK_QUESTIONS[8]
    
    return FALLBACK_QUESTIONS[-1]  # Generic scholarship type question

def create_scholarship_prompt(user_input, user_details=None):
    query_type = analyze_query_type(user_input)
    logger.debug(f"Creating prompt for query type: {query_type}")
    
    recent_user = db.session.query(UserDetails).order_by(UserDetails.timestamp.desc()).first()
    stored_scholarship_type = recent_user.scholarship_type if recent_user and recent_user.scholarship_type != 'unspecified' else None
    stored_details = {}
    if recent_user:
        stored_details = {
            'caste': recent_user.caste,
            'income': recent_user.income,
            'gender': recent_user.gender,
            'course_level': recent_user.course_level,
            'is_hostel': recent_user.is_hostel,
            'cgpa': recent_user.cgpa,
            'is_minority': recent_user.is_minority,
            'has_disability': recent_user.has_disability,
            'ex_serviceman_parent': recent_user.ex_serviceman_parent,
            'scholarship_type': stored_scholarship_type
        }
    
    # Determine missing details for fallback question
    current_details = user_details or {}
    combined_details = {**stored_details, **current_details}
    missing_details = get_missing_details(combined_details)
    fallback_question = create_fallback_question(missing_details)
    
    common_mistakes = """
💡 **Avoid these mistakes:**
📄 Missing/blurry/expired documents
📅 Applying late, wrong year
📚 Wrong marks/course code
🏷️ Wrong category, multiple applications
🏦 Inactive/wrong bank details
📱 Wrong mobile/email, lost ID
"""
    
    if query_type == 'greeting':
        return f"""
User greeted: "{user_input}". Respond warmly, introduce as Maharashtra scholarship assistant.
Example: "Hi! I'm here to help with Maharashtra scholarships. What's up?"
"""
    
    elif query_type == 'bot_info':
        return f"""
User asked: "{user_input}". Explain capabilities:
- Find government, private, NGO, college scholarships
- Guide on applications, eligibility
- Share tips
Ask: "What scholarship info do you need?"
"""
    
    elif query_type == 'bot_functionality':
        return f"""
User asked: "{user_input}". Explain:
- Match details to government, private, NGO, college scholarships
- Use portals like MahaDBT (https://mahadbtmahait.gov.in/), NSP (https://scholarships.gov.in/)
- Guide applications
Ask: "Which scholarship type interests you?"
"""
    
    elif query_type == 'casual':
        return f"""
User said: "{user_input}". Respond naturally, guide to scholarships if relevant.
Example: "Thanks! Need scholarship help?"
"""
    
    elif query_type == 'general_conversation':
        return f"""
User said: "{user_input}". Redirect to scholarship types.
Example: "I focus on Maharashtra scholarships (government, private, NGO, college). Which type are you looking for?"
{common_mistakes}
"""
    
    elif query_type == 'scholarship_query':
        if stored_scholarship_type:
            details_str = "\n🤖 What the chatbot knows about the user:\n"
            for key, value in stored_details.items():
                if value is not None and key != 'scholarship_type':
                    details_str += f"- {key.replace('_', ' ').title()}: {value}\n"
            details_str += f"- Scholarship Type: {stored_scholarship_type.capitalize()}\n"
            return f"""
The user said: "{user_input}"  
Scholarship type stored: **{stored_scholarship_type.capitalize()}**  
📌 Mention all details which you know about user:
{details_str}

🎯 Your task:

1. List 4–5 relevant **scholarships in Maharashtra** for the `{stored_scholarship_type}` category in this compact format:

🎓 **[Scholarship Name]**  
**Level:** [UG / PG / Both]  
**Category:** [SC / ST / OBC / General / Minority / Girls]  
**Description:** [Short 1-line purpose of scholarship]  
**Portal:** [Application or Info URL]

2. Match scholarships to the user profile (caste, income, level, etc. if available).
3. End with: _"Would you like more details on any of these scholarships or help with the application process?"_

🔖 Example types:
- Government: Post Matric Scholarship for SC, Rajarshi Shahu Maharaj Scholarship  
- Private: TATA Capital Pankh, Colgate Keep India Smiling  
- NGO: ONGC Foundation, KC Mahindra Scholarship  
- College: Mumbai University Merit Scholarship, Pune University Endowment

{common_mistakes}

{fallback_question}
"""
        return f"""
The user said: "{user_input}"  
❗ No scholarship type is currently stored.

🎯 Your task:

1. List the main types of scholarships in Maharashtra with 1-line descriptions:  
- **Government:** State/central funded (e.g., [MahaDBT](https://mahadbtmahait.gov.in/))  
- **Private:** Corporate-funded (e.g., [Buddy4Study](https://buddy4study.com/))  
- **NGO:** Non-profit based  
- **College/University:** Institution-specific (e.g., [Mumbai University](https://mu.ac.in/))

2. Ask: _"Which type of scholarship are you looking for? (e.g., Government, Private, NGO, College)"_

{common_mistakes}

{fallback_question}
"""
    
    elif query_type == 'scholarship_personalized':
        details_str = ""
        if user_details:
            details_str += "\n📌 User profile based on current input:\n"
            for key, value in user_details.items():
                if value is not None:
                    details_str += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        if stored_scholarship_type:
            if not user_details or 'scholarship_type' not in user_details or user_details['scholarship_type'] is None:
                user_details = user_details or {}
                user_details['scholarship_type'] = stored_scholarship_type
            details_str += f"\n📦 Scholarship type from memory: **{stored_scholarship_type.capitalize()}**\n"
            return f"""
The user said: "{user_input}"  
📌 Mention all details which you know about user:
{details_str}

🎯 Your task:

1. Suggest 4–5 relevant **scholarships in Maharashtra** matching the user's `{stored_scholarship_type}` category and profile:

🎓 **[Scholarship Name]**  
**Level:** [UG / PG / Both]  
**Category:** [SC / ST / OBC / General / Minority / Girls]  
**Description:** [Short 1-line purpose of scholarship]  
**Portal:** [Application or Info URL]

2. Ensure matches based on caste, income, academic level, etc.
3. End with: _"Would you like more details on any of these scholarships or help with the application process?"_

🔖 Example sources:
- Government: Post Matric for SC, Rajarshi Shahu  
- Private: Tata Capital, Colgate  
- NGO: ONGC, KC Mahindra  
- College: University-based

{common_mistakes}

{fallback_question}
"""
        return f"""
The user said: "{user_input}"  
Here is what the chatbot knows based on this query:  
{details_str}  
❗ However, no scholarship type is stored yet.

🎯 Your task:

1. List the main types of scholarships in Maharashtra with 1-line descriptions:  
- **Government:** e.g., [MahaDBT](https://mahadbtmahait.gov.in/)  
- **Private:** e.g., [Buddy4Study](https://buddy4study.com/)  
- **NGO:** Often listed on private platforms  
- **College:** e.g., [Mumbai University](https://mu.ac.in/)

2. Ask: _"Which type of scholarship are you looking for? (e.g., Government, Private, NGO, College)"_

{common_mistakes}

{fallback_question}
"""
    
    elif query_type == 'scholarship_types':
        return f"""
User asked: "{user_input}" about types of scholarships.
Your task is to:
1. List the main types of scholarships available in Maharashtra with brief descriptions:
   - Government: Funded by state or central government, e.g., via MahaDBT (https://mahadbtmahait.gov.in/)
   - Private: Offered by private organizations, e.g., via Buddy4Study (https://buddy4study.com/)
   - NGO: Provided by non-profits, often listed on Buddy4Study
   - College/University: Institution-specific, e.g., Mumbai University (https://mu.ac.in/)
2. Ask: "Which type of scholarship are you looking for? (e.g., Government, Private, NGO, College)"

End with:
{common_mistakes}

{fallback_question}
"""
    
    elif query_type == 'scholarship_type_selection':
        scholarship_type = user_details.get('scholarship_type', 'unspecified') if user_details else 'unspecified'
        summary = "Based on our conversation, here's what I know about you: "
        known_details = []
        
        if recent_user:
            if recent_user.course_level:
                course = 'BSc Computer Science' if recent_user.course_level == 'ug' and 'bsc' in user_input.lower() else 'Postgraduate'
                known_details.append(f"you are a {course} student")
            if recent_user.caste:
                known_details.append(f"your category is {recent_user.caste.upper()}")
            if recent_user.income:
                known_details.append(f"your family income is {recent_user.income}")
            if recent_user.gender:
                known_details.append(f"you are a {recent_user.gender} student")
        
        if scholarship_type != 'unspecified':
            known_details.append(f"you are looking for {scholarship_type} scholarships")
        
        if known_details:
            summary += ", ".join(known_details) + "."
        else:
            summary = f"I know you're looking for {scholarship_type} scholarships."
        
        details_str = ""
        if user_details:
            details_str = "\nUser profile from current query:\n"
            for key, value in user_details.items():
                if value is not None:
                    details_str += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        return f"""
The user said: "{user_input}" and selected the scholarship type: {scholarship_type.capitalize() if scholarship_type != 'unspecified' else 'Unspecified'}.
{summary}
{details_str}
Your task is to:
1. List 2–3 relevant scholarships in Maharashtra for the {scholarship_type} type in this compact format:

🎓 **[Scholarship Name]**  
**Level:** [UG / PG / Both]  
**Category:** [SC / ST / OBC / General / Minority / Girls]  
**Description:** [Short 1-line purpose of scholarship]  
**Portal:** [Application or Info URL]

2. Ensure scholarships match the user's profile (if details provided) and are specific to Maharashtra.
3. End with: "Would you like more details on any of these scholarships or help with the application process?"

Example scholarships (customize based on type):
- Government: Post Matric Scholarship for SC Students, Rajarshi Shahu Maharaj Scholarship
- Private: TATA Capital Pankh Scholarship, Colgate Keep India Smiling Scholarship
- NGO: ONGC Foundation Scholarship, KC Mahindra Scholarship
- College: Mumbai University Merit Scholarship, Pune University Endowment Scholarship

{common_mistakes}

{fallback_question}
"""
    
    return f"""
Unhandled intent: {query_type}. Redirect to scholarship help.
Example: "I'm here for Maharashtra scholarships (government, private, NGO, college). Which type are you looking for?"
{common_mistakes}

{fallback_question}
"""

def format_response(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(#{1,6})\s*([^\n]+)', r'\1 \2', text)
    text = re.sub(r'^\s*[•·]\s*', '• ', text, flags=re.MULTILINE)
    
    # Ensure the fallback question is properly formatted
    text = re.sub(r'(Would you like|Can you share|Are you|Do you|What is).*\?', 
                 lambda m: "\n\n" + m.group(0), text)
    
    lines = text.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    # Ensure the response ends with a question
    if not any(punct in text[-1] for punct in ['?', '!']):
        text += "\n\nWould you like more information about any of these scholarships?"
    
    return text.strip()

def validate_input(user_input):
    if not user_input or not user_input.strip():
        return False, "Please provide a valid query"
    
    if len(user_input.strip()) < 3:
        return False, "Query too short. Please provide more details"
    
    if len(user_input) > 1000:
        return False, "Query too long. Please keep it under 1000 characters"
    
    return True, user_input.strip()

@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_input = data.get('query', '')
        
        is_valid, result = validate_input(user_input)
        if not is_valid:
            return jsonify({'error': result}), 400
        
        user_input = result
        logger.debug(f"Processing chat query: {user_input}")
        
        # Use Gemini for both intent detection and entity extraction
        query_type = analyze_query_type(user_input)
        user_details = extract_user_details_with_gemini(user_input)
        
        with app.app_context():
            # Validate extracted details
            valid_details = {k: v for k, v in user_details.items() if v is not None}
            valid_details['intent'] = query_type  # Store the detected intent
            
            logger.debug(f"Valid details to store: {valid_details}")
            
            if valid_details:
                existing_user = db.session.query(UserDetails).order_by(UserDetails.timestamp.desc()).first()
                
                if existing_user:
                    # Update only non-None fields, preserving scholarship_type unless explicitly changed
                    existing_user.query = user_input
                    existing_user.intent = query_type
                    existing_user.timestamp = datetime.utcnow()
                    
                    for key, value in valid_details.items():
                        if key != 'scholarship_type' or (key == 'scholarship_type' and value is not None):
                            setattr(existing_user, key, value)
                    
                    logger.debug(f"Updated user details for record ID {existing_user.id}: {valid_details}")
                else:
                    new_user = UserDetails(
                        query=user_input,
                        caste=user_details.get('caste'),
                        income=user_details.get('income'),
                        gender=user_details.get('gender'),
                        course_level=user_details.get('course_level'),
                        is_hostel=user_details.get('is_hostel'),
                        cgpa=user_details.get('cgpa'),
                        is_minority=user_details.get('is_minority'),
                        has_disability=user_details.get('has_disability'),
                        ex_serviceman_parent=user_details.get('ex_serviceman_parent'),
                        scholarship_type=user_details.get('scholarship_type', 'unspecified'),
                        intent=query_type
                    )
                    db.session.add(new_user)
                    logger.debug(f"Created new user details record: {valid_details}")
                
                try:
                    db.session.commit()
                    logger.info(f"Successfully stored/updated user details: {valid_details}")
                except Exception as e:
                    db.session.rollback()
                    logger.error(f"Database storage error: {str(e)}")
                    return jsonify({'error': f'Failed to store user details: {str(e)}'}), 500
            else:
                logger.warning(f"No valid details extracted from input: {user_input}")
            
            # Log current database state for debugging
            current_user = db.session.query(UserDetails).order_by(UserDetails.timestamp.desc()).first()
            if current_user:
                current_state = {c.name: getattr(current_user, c.name) for c in current_user.__table__.columns}
                logger.debug(f"Current database state: {current_state}")
        
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not configured")
            return jsonify({'error': 'GEMINI_API_KEY not configured'}), 500
        
        full_prompt = create_scholarship_prompt(user_input, user_details)
        logger.debug(f"Generated prompt: {full_prompt[:200]}...")

        # Add instructions to ensure chatbot knows answers to fallback questions
        full_prompt += """
IMPORTANT INSTRUCTIONS FOR THE CHATBOT:
1. You MUST maintain the response structure: 
   - First provide scholarship information based on known user details
   - Then ask ONE relevant follow-up question at the end

2. For follow-up questions:
   - If user hasn't provided key details (caste, income, etc.), ask for those
   - If all details are provided, ask if they want application process info
   - Keep questions simple and one at a time

3. You MUST know the answers to your own follow-up questions. For example:
   - If asking about caste, be ready to explain different categories
   - If asking about application process, be ready to guide them
   - If asking about income, know the typical ranges

4. Always end with a question to keep the conversation flowing.
"""

        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }
        params = {'key': GEMINI_API_KEY}

        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            params=params,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            content = response.json()
            if 'candidates' in content and len(content['candidates']) > 0:
                generated_text = content['candidates'][0]['content']['parts'][0]['text']
                formatted_text = format_response(generated_text)
                logger.info(f"Generated response: {formatted_text[:100]}...")
                return jsonify({
                    'success': True,
                    'response': formatted_text,
                    'formatted_markdown': formatted_text,
                    'metadata': {
                        'query': user_input,
                        'response_type': 'scholarship_info',
                        'source': 'gemini_knowledge',
                        'timestamp': str(datetime.now().isoformat()),
                        'includes_links': True,
                        'includes_common_mistakes': True,
                        'scholarship_type_stored': user_details.get('scholarship_type', 'unspecified'),
                        'intent_detected': query_type
                    }
                })
            else:
                logger.error("No response generated from Gemini")
                return jsonify({'error': 'No response generated from Gemini'}), 500
        else:
            error_message = f'Gemini API error: {response.status_code}'
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    error_message += f" - {error_detail['error'].get('message', 'Unknown error')}"
            except:
                error_message += f" - {response.text[:200]}"
            logger.error(error_message)
            return jsonify({'error': error_message}), response.status_code
            
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        return jsonify({'error': 'Request timeout. Please try again.'}), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': str(datetime.now()),
        'database': 'connected' if db.engine else 'disconnected',
        'gemini_api': 'configured' if GEMINI_API_KEY else 'not_configured'
    })

if __name__ == '__main__':
    logger.info("Starting Enhanced Maharashtra Scholarship Assistant with Gemini-powered entity extraction...")
    logger.info(f"GEMINI_API_KEY configured: {'Yes' if GEMINI_API_KEY else 'No'}")
    logger.info(f"Database configured: {'Yes' if app.config['SQLALCHEMY_DATABASE_URI'] else 'No'}")
    logger.info("\nAvailable endpoints:")
    logger.info("  POST /chat - Main chat endpoint for scholarship queries")
    logger.info("  GET  /health - Health check and status")
    logger.info("\nKey Features:")
    logger.info("  - Gemini-powered intent detection and entity extraction")
    logger.info("  - Persistent user profile tracking")
    logger.info("  - Smart fallback questions to gather more details")
    logger.info("  - Structured responses with scholarship info first, questions last")
    
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured!")
        logger.warning("Set your Gemini API key: export GEMINI_API_KEY='your_api_key_here'")
        logger.warning("Get API key from: https://aistudio.google.com/app/apikey")
    
    if not app.config['SECRET_KEY'] or app.config['SECRET_KEY'] == 'default-secret-key':
        logger.warning("FLASK_SECRET_KEY not configured or using default!")
        logger.warning("Set a secure key: export FLASK_SECRET_KEY='your_secure_key_here'")
    
    logger.info("\nReady to assist with Maharashtra scholarship queries with Gemini-powered entity extraction!")
    logger.info("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)