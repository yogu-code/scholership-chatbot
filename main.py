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
    scholarship_type = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<UserDetails {self.id}>"

# Create database tables
with app.app_context():
    db.create_all()

def lemmatize_text(text):
    words = text.split()
    return ' '.join(lemmatizer.lemmatize(word) for word in words)

def extract_user_details(user_input):
    user_input_lower = user_input.lower().strip()
    logger.debug(f"Extracting details from input: {user_input_lower}")
    
    details = {
        'caste': None,
        'income': None,
        'gender': None,
        'course_level': None,
        'is_hostel': None,
        'cgpa': None,
        'is_minority': None,
        'has_disability': None,
        'ex_serviceman_parent': None,
        'scholarship_type': None
    }

    caste_patterns = {
        'sc': r'\b(sc|scheduled caste)\b',
        'st': r'\b(st|scheduled tribe)\b',
        'obc': r'\b(obc|other backward class)\b',
        'general': r'\b(general|open)\b',
        'minority': r'\b(minority|muslim|christian|sikh|jain|parsi|buddhist)\b'
    }
    for caste, pattern in caste_patterns.items():
        if re.search(pattern, user_input_lower):
            details['caste'] = caste
            if caste == 'minority':
                details['is_minority'] = True
            break

    income_pattern = r'\b(?:income|family income)\s*(?:is|of)?\s*(?:below|under|\d+\.?\d*\s*(?:lakh|lacs?|k|thousand)?|\d+\s*to\s*\d+\.?\d*\s*(?:lakh|lacs?))\b'
    income_match = re.search(income_pattern, user_input_lower)
    if income_match:
        details['income'] = income_match.group(0)

    if re.search(r'\b(girl|female|woman)\b', user_input_lower):
        details['gender'] = 'female'
    elif re.search(r'\b(boy|male|man)\b', user_input_lower):
        details['gender'] = 'male'

    course_patterns = {
        'ug': r'\b(bsc|ba|be|btech|bcom|undergraduate|1st year|2nd year|3rd year|b\.?sc\.?\s*(?:cs|computer science)?)\b',
        'pg': r'\b(msc|ma|mtech|mcom|postgraduate|master)\b'
    }
    for level, pattern in course_patterns.items():
        if re.search(pattern, user_input_lower):
            details['course_level'] = level
            break

    if re.search(r'\b(hostel|staying in hostel|hosteller)\b', user_input_lower):
        details['is_hostel'] = True
    elif re.search(r'\b(not in hostel|day scholar)\b', user_input_lower):
        details['is_hostel'] = False

    cgpa_pattern = r'\b(\d+\.?\d*)\s*(?:cgpa|percentage|percent|marks)\b'
    cgpa_match = re.search(cgpa_pattern, user_input_lower)
    if cgpa_match:
        details['cgpa'] = float(cgpa_match.group(1))

    if re.search(r'\b(disability|disabled|handicap)\b', user_input_lower):
        details['has_disability'] = True

    if re.search(r'\b(ex-serviceman|freedom fighter|parent is ex-serviceman)\b', user_input_lower):
        details['ex_serviceman_parent'] = True

    scholarship_types = {
        'government': r'\b(government|govt|mahadbt|nsp|national scholarship|option\s*1|choose\s*1|select\s*1|1)\b',
        'private': r'\b(private|buddy4study|vidyasaarathi|option\s*2|choose\s*2|select\s*2|2)\b',
        'ngo': r'\b(ngo|non-profit|charity|option\s*3|choose\s*3|select\s*3|3)\b',
        'college': r'\b(college|university|mumbai university|pune university|option\s*4|choose\s*4|select\s*4|4)\b'
    }
    for sch_type, pattern in scholarship_types.items():
        match = re.search(pattern, user_input_lower)
        if match:
            details['scholarship_type'] = sch_type
            logger.debug(f"Scholarship type extracted: {sch_type} from match: {match.group(0)}")
            break
        else:
            logger.debug(f"No match for scholarship type pattern: {pattern}")

    logger.debug(f"Extracted details: {details}")
    return details

def analyze_query_type(user_input):
    user_input_lower = user_input.lower().strip()
    user_input_lemmatized = lemmatize_text(user_input_lower)
    logger.debug(f"Analyzing query type for input: {user_input_lower}")
    
    intent_keywords = {
        'greeting': {
            'keywords': ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening'],
            'weight': 1.0
        },
        'bot_info': {
            'keywords': [
                'what can you do', 'what do you do', 'how can you help', 'tell me about yourself',
                'who are you', 'what are you', 'your capabilities', 'help me', 'about this chatbot'
            ],
            'weight': 0.9
        },
        'bot_functionality': {
            'keywords': [
                'how does this work', 'how do you work', 'how to use this chatbot', 'how you find scholarships',
                'how does it work', 'explain how you work', 'how to get scholarship through you'
            ],
            'weight': 0.9
        },
        'casual': {
            'keywords': [
                'how are you', 'what\'s up', 'how is your day', 'tell me a joke',
                'thank you', 'thanks', 'okay', 'ok', 'yes', 'no', 'maybe',
                'i am fine', 'good', 'nice', 'cool', 'awesome', 'great'
            ],
            'weight': 0.7
        },
        'scholarship_query': {
            'keywords': [
                'scholarship', 'financial aid', 'education loan', 'study help', 'fees', 'grant',
                'mahadbt', 'nsp', 'government scheme', 'tuition fee', 'hostel allowance',
                'eligibility', 'application', 'deadline', 'scholarship list', 'available scholarships'
            ],
            'weight': 0.95
        },
        'scholarship_personalized': {
            'keywords': [
                'i am', 'i\'m', 'my category', 'my income', 'student', 'bsc', 'msc', 'engineering', 'medical',
                'college', 'university', 'sc', 'st', 'obc', 'general', 'low income', 'merit',
                'find scholarship', 'scholarship for me', 'looking for scholarship', 'need scholarship'
            ],
            'weight': 1.1
        },
        'scholarship_types': {
            'keywords': [
                'scholarship types', 'types of scholarships', 'government scholarship', 'private scholarship',
                'ngo scholarship', 'college scholarship', 'university scholarship', 'mahadbt', 'buddy4study',
                'mumbai university', 'scholarship providers', 'kinds of scholarships', 'scholarship categories'
            ],
            'weight': 1.0
        },
        'scholarship_type_selection': {
            'keywords': [
                'government', 'govt', 'private', 'ngo', 'college', 'university', 'mahadbt', 'buddy4study',
                'national scholarship', 'vidyasaarathi', 'mumbai university', 'pune university',
                'option 1', 'choose 1', 'select 1', '1',
                'option 2', 'choose 2', 'select 2', '2',
                'option 3', 'choose 3', 'select 3', '3',
                'option 4', 'choose 4', 'select 4', '4'
            ],
            'weight': 1.5
        },
        'general_conversation': {
            'keywords': [],
            'weight': 0.5
        }
    }
    
    lemmatized_keywords = {
        intent: {
            'keywords': [lemmatize_text(keyword) for keyword in data['keywords']],
            'weight': data['weight']
        }
        for intent, data in intent_keywords.items()
    }
    
    question_patterns = [
        r'^(what|how|where|when|why|who|can you|tell me)\b',
        r'.*\?$'
    ]
    
    personalized_patterns = [
        r'\bi(?:\'m| am)\b.*(student|category|income|scholarship)',
        r'\bmy (category|income|course|degree)\b',
        r'\b(sc|st|obc|general)\b.*scholarship'
    ]
    
    scholarship_types_patterns = [
        r'\b(type|kind|category|categories|source|provider).*(scholarship|grants)\b',
        r'\b(government|private|ngo|college|university|mahadbt|buddy4study|mumbai university)\b.*scholarship',
        r'\bwhat (type|kind|category|categories).*(scholarship|grants)\b',
        r'\bscholarship.*(type|kind|category|categories|source|provider)\b'
    ]
    
    scholarship_type_selection_patterns = [
        r'\b(government|govt|private|ngo|college|university|mahadbt|buddy4study|nsp|vidyasaarathi|mumbai university|pune university)\b',
        r'^(option|choose|select)\s*[1-4]$',
        r'^\d$'
    ]
    
    negation_keywords = ['not', 'no', 'don\'t', 'doesn\'t']
    
    intent_scores = {intent: 0.0 for intent in intent_keywords}
    
    for intent, data in lemmatized_keywords.items():
        for phrase in data['keywords']:
            if phrase in user_input_lemmatized:
                intent_scores[intent] += data['weight']
    
    for intent, data in lemmatized_keywords.items():
        for keyword in data['keywords']:
            if len(keyword.split()) == 1 and keyword in user_input_lemmatized.split():
                intent_scores[intent] += data['weight'] * 0.5
    
    is_question = any(re.search(pattern, user_input_lower) for pattern in question_patterns)
    if is_question:
        intent_scores['scholarship_query'] *= 1.2
        intent_scores['bot_info'] *= 1.2
        intent_scores['bot_functionality'] *= 1.2
        intent_scores['scholarship_types'] *= 1.2
    
    is_personalized = any(re.search(pattern, user_input_lower) for pattern in personalized_patterns)
    if is_personalized:
        intent_scores['scholarship_personalized'] *= 1.5
    
    is_scholarship_types = any(re.search(pattern, user_input_lower) for pattern in scholarship_types_patterns)
    if is_scholarship_types:
        intent_scores['scholarship_types'] *= 1.5
    
    is_type_selection = any(re.search(pattern, user_input_lower) for pattern in scholarship_type_selection_patterns)
    if is_type_selection:
        intent_scores['scholarship_type_selection'] *= 2.5
    
    if any(neg in user_input_lemmatized for neg in negation_keywords):
        intent_scores['scholarship_query'] *= 0.5
        intent_scores['scholarship_personalized'] *= 0.5
        intent_scores['scholarship_types'] *= 0.5
        intent_scores['scholarship_type_selection'] *= 0.5
    
    if len(user_input.split()) <= 2:
        intent_scores['casual'] += 1.0
        intent_scores['general_conversation'] *= 0.8
    
    personal_keyword_count = sum(1 for keyword in lemmatized_keywords['scholarship_personalized']['keywords']
                                if keyword in user_input_lemmatized)
    if personal_keyword_count >= 3:
        intent_scores['scholarship_personalized'] *= 1.3
    
    types_keyword_count = sum(1 for keyword in lemmatized_keywords['scholarship_types']['keywords']
                             if keyword in user_input_lemmatized)
    if types_keyword_count >= 2:
        intent_scores['scholarship_types'] *= 1.3
    
    type_selection_count = sum(1 for keyword in lemmatized_keywords['scholarship_type_selection']['keywords']
                              if keyword in user_input_lemmatized)
    if type_selection_count >= 1:
        intent_scores['scholarship_type_selection'] *= 1.8
    
    max_intent = max(intent_scores, key=intent_scores.get)
    max_score = intent_scores[max_intent]
    
    if max_score < 0.5:
        max_intent = 'general_conversation'
    
    logger.debug(f"Detected intent: {max_intent} with score: {max_score}")
    return max_intent

def create_scholarship_prompt(user_input, user_details=None):
    query_type = analyze_query_type(user_input)
    logger.debug(f"Creating prompt for query type: {query_type}")
    
    recent_user = db.session.query(UserDetails).first()
    stored_scholarship_type = recent_user.scholarship_type if recent_user else None
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
            'ex_serviceman_parent': recent_user.ex_serviceman_parent
        }
    
    follow_up_questions = """
📋 **Please share:**
1. Category? (SC/ST/OBC/General/Minority)
2. Income? (Below ₹1L/₹1-2.5L/₹2.5-8L/Above ₹8L)
3. Girl student?
4. Hostel stay?
5. CGPA/percentage?
6. Minority community? (Muslim/Christian/Sikh/etc.)
7. Disabilities?
8. Parents ex-servicemen/freedom fighters?
💡 More details help me find the best scholarships!
"""
    
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
{follow_up_questions}
"""
    
    elif query_type == 'scholarship_query':
        if stored_scholarship_type:
            details_str = "\nUser profile from database:\n"
            for key, value in stored_details.items():
                if value is not None:
                    details_str += f"- {key.replace('_', ' ').title()}: {value}\n"
            return f"""
The user said: "{user_input}" and has a stored scholarship type: {stored_scholarship_type.capitalize()}.
{details_str}
Your task is to:
1. List 2–3 relevant scholarships in Maharashtra for the {stored_scholarship_type} type in this compact format:

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
{follow_up_questions}
"""
        else:
            return f"""
The user said: "{user_input}" and has no stored scholarship type.
Your task is to:
1. List the main types of scholarships available in Maharashtra with brief descriptions:
   - Government: Funded by state or central government, e.g., via MahaDBT (https://mahadbtmahait.gov.in/)
   - Private: Offered by private organizations, e.g., via Buddy4Study (https://buddy4study.com/)
   - NGO: Provided by non-profits, often listed on Buddy4Study
   - College/University: Institution-specific, e.g., Mumbai University (https://mu.ac.in/)
2. Ask: "Which type of scholarship are you looking for? (e.g., Government, Private, NGO, College)"

End with:
{common_mistakes}
{follow_up_questions}
"""
    
    elif query_type == 'scholarship_personalized':
        details_str = ""
        if user_details:
            details_str = "\nUser profile from current query:\n"
            for key, value in user_details.items():
                if value is not None:
                    details_str += f"- {key.replace('_', ' ').title()}: {value}\n"
        if stored_scholarship_type:
            details_str += f"\nStored scholarship type: {stored_scholarship_type.capitalize()}\n"
            return f"""
The user said: "{user_input}" and has a stored scholarship type: {stored_scholarship_type.capitalize()}.
{details_str}
Your task is to:
1. List 2–3 relevant scholarships in Maharashtra for the {stored_scholarship_type} type in this compact format:

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
{follow_up_questions}
"""
        else:
            return f"""
The user said: "{user_input}" and shared some personal details but no scholarship type is stored.
{details_str}
Your task is to:
1. List the main types of scholarships available in Maharashtra with brief descriptions:
   - Government: Funded by state or central government, e.g., via MahaDBT (https://mahadbtmahait.gov.in/)
   - Private: Offered by private organizations, e.g., via Buddy4Study (https://buddy4study.com/)
   - NGO: Provided by non-profits, often listed on Buddy4Study
   - College/University: Institution-specific, e.g., Mumbai University (https://mu.ac.in/)
2. Ask: "Which type of scholarship are you looking for? (e.g., Government, Private, NGO, College)"

End with:
{common_mistakes}
{follow_up_questions}
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
{follow_up_questions}
"""
    
    elif query_type == 'scholarship_type_selection':
        scholarship_type = user_details.get('scholarship_type', 'unknown') if user_details else 'unknown'
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
The user said: "{user_input}" and selected the scholarship type: {scholarship_type.capitalize()}.
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
{follow_up_questions}
"""
    
    return f"""
Unhandled intent: {query_type}. Redirect to scholarship help.
Example: "I'm here for Maharashtra scholarships (government, private, NGO, college). Which type are you looking for?"
{common_mistakes}
{follow_up_questions}
"""

def format_response(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(#{1,6})\s*([^\n]+)', r'\1 \2', text)
    text = re.sub(r'^\s*[•·]\s*', '• ', text, flags=re.MULTILINE)
    lines = text.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    text = '\n'.join(cleaned_lines)
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
        
        user_details = extract_user_details(user_input)
        query_type = analyze_query_type(user_input)
        
        with app.app_context():
            existing_user = db.session.query(UserDetails).first()
            
            if query_type in ['scholarship_personalized', 'scholarship_type_selection', 'scholarship_query']:
                if existing_user:
                    existing_user.query = user_input
                    existing_user.timestamp = datetime.utcnow()
                    
                    for key, value in user_details.items():
                        if value is not None:
                            setattr(existing_user, key, value)
                    
                    logger.debug(f"Updated user details for single record")
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
                        scholarship_type=user_details.get('scholarship_type')
                    )
                    db.session.add(new_user)
                    logger.debug(f"Created single user details record")
                
                try:
                    db.session.commit()
                    logger.info(f"Stored/Updated user details for single record, Scholarship Type={user_details.get('scholarship_type')}")
                except Exception as e:
                    db.session.rollback()
                    logger.error(f"Database storage error: {str(e)}")
                    return jsonify({'error': f'Failed to store user details: {str(e)}'}), 500
        
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not configured")
            return jsonify({'error': 'GEMINI_API_KEY not configured'}), 500
        
        full_prompt = create_scholarship_prompt(user_input, user_details)
        logger.debug(f"Generated prompt: {full_prompt[:200]}...")

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
                        'scholarship_type_stored': user_details.get('scholarship_type', None)
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
        'message': 'Maharashtra Scholarship Assistant is running',
        'gemini_configured': bool(GEMINI_API_KEY),
        'database_configured': bool(app.config['SQLALCHEMY_DATABASE_URI']),
        'version': '2.6',
        'features': [
            'Gemini AI powered responses',
            'Maharashtra scholarship information',
            'Direct scholarship website links',
            'Comprehensive database with URLs',
            'Common application mistakes guidance',
            'Pro tips for successful applications',
            'Helpline numbers included',
            'Single record user details storage'
        ],
        'key_links': {
            'mahadbt': 'https://mahadbtmahait.gov.in/',
            'nsp': 'https://scholarships.gov.in/',
            'buddy4study': 'https://buddy4study.com/'
        }
    })

@app.route('/links', methods=['GET'])
def get_scholarship_links():
    return jsonify({
        'government_portals': {
            'Maharashtra DBT Portal': 'https://mahadbtmahait.gov.in/',
            'National Scholarship Portal': 'https://scholarships.gov.in/',
            'PM Scholarship Portal': 'https://ksb.gov.in/',
            'UGC Scholarships': 'https://www.ugc.ac.in/page/Scholarships-and-Fellowships.aspx'
        },
        'search_platforms': {
            'Buddy4Study': 'https://buddy4study.com/',
            'Vidyasaarathi': 'https://www.vidyasaarathi.co.in/',
            'Scholarships.com': 'https://www.scholarships.com/'
        },
        'technical_education': {
            'Maharashtra DTE': 'https://dtemaharashtra.gov.in/',
            'AICTE Scholarships': 'https://www.aicte-india.org/schemes/students-development-schemes'
        },
        'universities': {
            'Pune University': 'http://www.unipune.ac.in/student_welfare/scholarships.htm',
            'Mumbai University': 'https://mu.ac.in/student-services',
            'Shivaji University': 'http://www.unishivaji.ac.in/scholarships'
        },
        'helplines': {
            'Maharashtra DBT Helpline': '18002102131',
            'NSP Helpline': '0120-6619540',
            'Student Helpline': '8448440632'
        }
    })

@app.route('/mistakes', methods=['GET'])
def get_common_mistakes():
    return jsonify({
        'documentation_errors': {
            'missing_documents': 'Not submitting all required documents',
            'expired_certificates': 'Using outdated certificates (must be within 1 year)',
            'wrong_format': 'Uploading documents in incorrect format',
            'poor_quality_scans': 'Blurry or unclear document copies',
            'missing_signatures': 'Forgetting required signatures'
        },
        'income_declaration_mistakes': {
            'wrong_income_figures': 'Not matching income with official certificates',
            'wrong_sources': 'Including/excluding income sources incorrectly',
            'outdated_income_proof': 'Using old salary certificates',
            'agricultural_income_errors': 'Incorrect agricultural/business income calculation'
        },
        'timing_issues': {
            'last_minute_rush': 'Applying on deadline day',
            'missing_renewal_dates': 'Forgetting annual scholarship renewal',
            'academic_year_confusion': 'Applying for wrong academic year',
            'document_expiry': 'Certificates expiring before deadline'
        },
        'academic_errors': {
            'wrong_cgpa': 'Entering incorrect academic scores',
            'course_code_mistakes': 'Wrong course/branch selection',
            'institution_details': 'Incorrect college AISHE codes',
            'semester_confusion': 'Wrong academic year/semester'
        },
        'category_mistakes': {
            'wrong_category': 'Incorrect caste/reservation category',
            'duplicate_applications': 'Multiple applications for same scholarship',
            'ineligible_applications': 'Applying for non-qualifying scholarships',
            'age_limit_ignorance': 'Not checking age eligibility'
        },
        'bank_account_issues': {
            'inactive_accounts': 'Using dormant bank accounts',
            'wrong_details': 'Incorrect account/IFSC details',
            'non_dbt_accounts': 'Accounts not linked to DBT',
            'joint_account_problems': 'Using parent\'s instead of student\'s account'
        },
        'technical_mistakes': {
            'incomplete_applications': 'Not completing all form sections',
            'wrong_contact_info': 'Inactive mobile/email addresses',
            'password_problems': 'Forgetting login credentials',
            'browser_issues': 'Incompatible browser or cache issues'
        },
        'pro_tips': {
            'start_early': 'Begin applications 15-20 days before deadline',
            'create_checklist': 'Verify all documents before submission',
            'keep_records': 'Maintain copies of all documents',
            'regular_followup': 'Check application status weekly',
            'seek_help': 'Contact helplines for guidance'
        },
        'helplines': {
            'Maharashtra DBT': '18002102131',
            'NSP Helpline': '0120-6619540',
            'Student Helpline': '8448440632'
        }
    })

@app.route('/test', methods=['POST'])
def test_chat():
    try:
        sample_query = "Government"
        
        test_request = {'query': sample_query}
        
        with app.test_request_context('/chat', method='POST', json=test_request):
            return chat_with_gemini()
            
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        return jsonify({'error': f'Test error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    logger.error("404 Endpoint not found")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    logger.error("405 Method not allowed")
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error("500 Internal server error")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Enhanced Maharashtra Scholarship Assistant...")
    logger.info(f"GEMINI_API_KEY configured: {'Yes' if GEMINI_API_KEY else 'No'}")
    logger.info(f"Database configured: {'Yes' if app.config['SQLALCHEMY_DATABASE_URI'] else 'No'}")
    logger.info("\nAvailable endpoints:")
    logger.info("  POST /chat - Main chat endpoint for scholarship queries")
    logger.info("  GET  /health - Health check and status")
    logger.info("  GET  /links - Get all scholarship website links")
    logger.info("  GET  /mistakes - Get common scholarship application mistakes")
    logger.info("  POST /test - Test endpoint with sample query")
    logger.info("\nUsage:")
    logger.info("  curl -X POST http://localhost:5000/chat \\")
    logger.info("       -H 'Content-Type: application/json' \\")
    logger.info("       -d '{\"query\": \"Government\"}'")
    logger.info("\nKey Features:")
    logger.info("  Direct scholarship website links included")
    logger.info("  Government portals (MahaDBT, NSP)")
    logger.info("  Private scholarship platforms")
    logger.info("  University-specific links")
    logger.info("  Helpline numbers included")
    logger.info("  Single record user details storage")
    
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured!")
        logger.warning("Set your Gemini API key: export GEMINI_API_KEY='your_api_key_here'")
        logger.warning("Get API key from: https://aistudio.google.com/app/apikey")
    
    if not app.config['SECRET_KEY'] or app.config['SECRET_KEY'] == 'default-secret-key':
        logger.warning("FLASK_SECRET_KEY not configured or using default!")
        logger.warning("Set a secure key: export FLASK_SECRET_KEY='your_secure_key_here'")
    
    logger.info("\nReady to assist with Maharashtra scholarship queries with single record persistence!")
    logger.info("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)