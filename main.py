import os
import requests
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent'

def analyze_query_type(user_input):
    """
    Better analysis of query type for more natural responses
    """
    user_input_lower = user_input.lower().strip()
    
    # Simple greetings
    greetings = ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening']
    
    # General questions about the bot
    bot_questions = [
        'what can you do', 'what do you do', 'how can you help', 'tell me about yourself',
        'who are you', 'what are you', 'your capabilities', 'help me'
    ]
    
    # Personal/casual conversation
    personal_casual = [
        'how are you', 'what\'s up', 'how is your day', 'tell me a joke',
        'thank you', 'thanks', 'okay', 'ok', 'yes', 'no', 'maybe',
        'i am fine', 'good', 'nice', 'cool', 'awesome', 'great'
    ]
    
    # Scholarship-related keywords
    scholarship_keywords = [
        'scholarship', 'financial aid', 'education loan', 'study help', 'fees', 'grant',
        'bsc', 'msc', 'engineering', 'medical', 'college', 'university', 'student',
        'sc', 'st', 'obc', 'general', 'category', 'income', 'eligibility', 'application',
        'mahadbt', 'nsp', 'government scheme', 'tuition fee', 'hostel allowance'
    ]
    
    # Study/education but not necessarily scholarship
    education_general = [
        'course', 'exam', 'admission', 'career', 'job', 'placement', 'internship',
        'semester', 'marks', 'cgpa', 'percentage', 'result'
    ]
    
    # Check for exact matches first
    if user_input_lower in greetings:
        return 'greeting'
    
    if any(phrase in user_input_lower for phrase in bot_questions):
        return 'bot_info'
    
    if user_input_lower in personal_casual or len(user_input.split()) <= 2:
        return 'casual'
    
    # Check for scholarship-specific queries
    if any(keyword in user_input_lower for keyword in scholarship_keywords):
        return 'scholarship_query'
    
    # Education-related but not scholarship-specific
    if any(keyword in user_input_lower for keyword in education_general):
        return 'education_general'
    
    # Default to general conversation
    return 'general_conversation'

def create_scholarship_prompt(user_input):
    """
    Create context-appropriate prompts for different query types
    """
    query_type = analyze_query_type(user_input)
    
    if query_type == 'greeting':
        return f"""
You are a friendly, helpful assistant specializing in Maharashtra scholarships. The user greeted you with "{user_input}".

Respond warmly and naturally. Keep it brief and conversational. Introduce yourself as someone who can help with scholarship information when they need it, but don't immediately list scholarships.

Example responses:
- "Hello! Nice to meet you. I'm here to help students in Maharashtra find scholarships and financial aid. How's your day going?"
- "Hi there! I'm your scholarship assistant. I can help you navigate the world of educational funding in Maharashtra. What brings you here today?"

Keep it natural and human-like. Don't overwhelm with information unless asked.
"""

    elif query_type == 'bot_info':
        return f"""
The user asked "{user_input}" - they want to know about your capabilities.

You are a Maharashtra Scholarship Assistant. Explain what you can do in a conversational way:

You can help with:
- Finding suitable scholarships based on their profile
- Information about government scholarships (MahaDBT, NSP)
- Private and corporate scholarships
- Application procedures and deadlines
- Common mistakes to avoid
- Document requirements
- Eligibility criteria

But also mention you're happy to chat and help them understand their options step by step.

Keep it conversational, not like a boring list. End by asking what specific help they need.
"""

    elif query_type == 'casual':
        return f"""
The user said "{user_input}" which is casual conversation.

Respond naturally and conversationally. Acknowledge what they said and gently guide toward how you can help if appropriate, but don't force scholarship information.

Examples:
- If they say "thanks": "You're welcome! Is there anything else I can help you with?"
- If they say "how are you": "I'm doing well, thank you for asking! How about you? Are you looking for any information today?"
- If they say "ok" or "yes": "Great! What would you like to know more about?"

Keep it natural and human-like. Only mention scholarships if it flows naturally in the conversation.
"""

    elif query_type == 'education_general':
        return f"""
The user asked "{user_input}" which is education-related but not specifically about scholarships.

Respond to their question naturally first, then gently offer scholarship help if it's relevant to their situation.

For example:
- If asking about courses: Answer about courses, then mention "By the way, if you're concerned about funding your education, I can help you find relevant scholarships too."
- If asking about admissions: Help with admission info, then ask if they need financial assistance information.

Don't force scholarship information, but make it available as an option.
"""

    elif query_type == 'general_conversation':
        return f"""
The user said "{user_input}" which seems like general conversation.

Respond naturally and helpfully to their query. If it's completely unrelated to education or scholarships, politely let them know your specialty while still being helpful.

Example: "I wish I could help with that, but I specialize in helping Maharashtra students find scholarships and educational funding. Is there anything related to your studies or educational expenses I can assist you with?"

Keep it friendly and conversational, not robotic.
"""

    else:  # scholarship_query
        return f"""
You are an expert Maharashtra Government Scholarship Assistant with comprehensive knowledge of all Maharashtra state scholarships and central scholarships available to Maharashtra students.

IMPORTANT INSTRUCTIONS:
1. Be conversational and human-like in your tone
2. Provide accurate, detailed information about Maharashtra scholarships when asked
3. Include specific amounts, eligibility criteria, and deadlines when known
4. Format your response in a clear, structured manner with proper headings
5. Always provide practical guidance and next steps
6. If you don't know specific current details, clearly state that and provide general guidance
7. Focus on scholarships relevant to the user's profile
8. Include relevant website links when mentioning specific scholarships
9. ALWAYS end with follow-up questions to gather more user details for better recommendations
10. ALWAYS include the common mistakes section to help students avoid application errors

RESPONSE FORMAT:

**For Scholarship Lists (when user asks for general/multiple scholarships):**
Use this compact format:

🎓 **[Scholarship Name]**
**Level:** [School / Diploma / UG / PG / Research]
**Type:** [Government / Central / State / Private / NGO]
**Category:** [SC / ST / OBC / EWS / Minority / Girls / General / All]
**Description:** [1-2 lines about what the scholarship offers or who it helps]
**Application:**
• Portal: [URL]
• Deadline: [Optional]

---

**For Detailed Scholarship Information (when user asks for specific details):**
Use this detailed format:

### 🎓 [Scholarship Name]
**Level:** [Undergraduate/Postgraduate/Both]
**Type:** [State/Central/Private]
**Category:** [SC/ST/OBC/General/Merit-based]

**Eligibility:**
- Academic: [Requirements]
- Income: [Limit if applicable]
- Other: [Additional criteria]

**Benefits:**
- Amount: ₹[Specific amount or range]
- Duration: [How long]
- Coverage: [What it covers]

**Application:**
- Portal: [Application portal]
- Documents: [Key documents needed]
- Deadline: [Typical deadline period]

**Contact:** [Official website or contact info]

---

COMPREHENSIVE MAHARASHTRA SCHOLARSHIPS DATABASE WITH LINKS:

## Government Scholarships (via MahaDBT):
**Main Portal:** https://mahadbtmahait.gov.in/

1. **Government of India Post Matric Scholarship**
   - Link: https://mahadbtmahait.gov.in/SchemeData/SchemeData?str=E9DDFA703C38E51A8F0FC3ACA06E90087F5900FF
   
2. **Post Matric Tuition Fee and Examination Fee (Freeship)**
   - Link: https://mahadbtmahait.gov.in/SchemeData/SchemeData?str=E9DDFA703C38E51A8F0FC3ACA06E90087F5900FF

3. **Post Matric Scholarship to VJNT Students**
   - Link: https://mahadbtmahait.gov.in/

4. **Post Matric Scholarship to OBC Students**
   - Link: https://mahadbtmahait.gov.in/

5. **Post Matric Scholarship to SBC Students**
   - Link: https://mahadbtmahait.gov.in/

6. **Pre-Matric Scholarship for SC Students**
   - Link: https://scholarships.gov.in/

7. **Rajarshi Chhatrapati Shahu Maharaj Merit Scholarship**
   - Link: https://mahadbtmahait.gov.in/

8. **Eklavya Scholarship**
   - Link: https://mahadbtmahait.gov.in/

9. **Dr. Panjabrao Deshmukh Vasatigruh Nirvah Bhatta Yojna (DTE)**
   - Link: https://dtemaharashtra.gov.in/

10. **Economically Backward Class (EBC) Fee Reimbursement Scheme**
    - Link: https://mahadbtmahait.gov.in/

## Central Government Scholarships:
**Main Portal:** https://scholarships.gov.in/

1. **National Scholarship Portal (NSP)**
   - Pre-Matric Scholarships
   - Post-Matric Scholarships
   - Merit-cum-Means Scholarships
   - Link: https://scholarships.gov.in/

2. **UGC Scholarships**
   - Link: https://www.ugc.ac.in/page/Scholarships-and-Fellowships.aspx

3. **AICTE Scholarships**
   - Pragati Scholarship (for Girls): https://www.aicte-india.org/schemes/students-development-schemes/pragati-scholarship-scheme
   - Saksham Scholarship (Differently Abled): https://www.aicte-india.org/schemes/students-development-schemes/saksham-scholarship-scheme

## University Specific Scholarships:

1. **Savitribai Phule Pune University**
   - Link: http://www.unipune.ac.in/
   - Student Welfare: http://www.unipune.ac.in/student_welfare/scholarships.htm

2. **University of Mumbai**
   - Link: https://mu.ac.in/
   - Student Services: https://mu.ac.in/student-services

3. **Shivaji University, Kolhapur**
   - Link: http://www.unishivaji.ac.in/
   - Scholarships: http://www.unishivaji.ac.in/scholarships

4. **Dr. Babasaheb Ambedkar Marathwada University**
   - Link: https://www.bamu.ac.in/
   - Student Welfare: https://www.bamu.ac.in/student-welfare

## Private/Corporate Scholarships:

1. **Tata Trusts Scholarships**
   - Link: https://www.tatatrusts.org/our-work/individual-grants-programme/education-grants

2. **Aditya Birla Scholarship**
   - Link: https://www.adityabirlascholars.com/

3. **Narotam Sekhsaria Foundation**
   - Link: https://nsfoundation.co.in/

4. **Lila Poonawalla Foundation (Girls Only)**
   - Link: https://www.lilapoonawallafoundation.org/

5. **HDFC Educational Crisis Scholarship**
   - Link: https://www.hdfcbank.com/personal/about-us/corporate-social-responsibility/holistic-rural-development-programme

6. **Indian Oil Academic Scholarships**
   - Link: https://iocl.com/pages/corporate-social-responsibility

7. **ONGC Scholarship**
   - Link: https://www.ongcindia.com/web/eng/csr-and-sd/education/scholarship-scheme

8. **Reliance Foundation Scholarships**
   - Link: https://reliancefoundation.org/

9. **Kotak Kanya Scholarship (Girls)**
   - Link: https://www.kotak.com/en/about-kotak/csr.html

10. **Glow & Lovely Foundation Scholarship (Girls)**
    - Application through: https://buddy4study.com/

## Specialized Platforms & Resources:

1. **Buddy4Study (Scholarship Search Platform)**
   - Link: https://buddy4study.com/
   - Maharashtra Scholarships: https://buddy4study.com/scholarships/maharashtra

2. **Vidyasaarathi (Corporate Scholarships)**
   - Link: https://www.vidyasaarathi.co.in/

3. **PMRF (Prime Minister's Research Fellowship)**
   - Link: https://pmrf.in/

4. **WomenTech Network Scholarships**
   - Link: https://www.womentech.net/scholarships

## Technical Education Scholarships:

1. **Maharashtra Technical Education (DTE)**
   - Link: https://dtemaharashtra.gov.in/
   - CAP Portal: https://cap.dtemaharashtra.gov.in/

2. **All India Council for Technical Education (AICTE)**
   - Link: https://www.aicte-india.org/
   - Student Scholarships: https://www.aicte-india.org/schemes/students-development-schemes

## Medical Education Scholarships:

1. **Directorate of Medical Education & Research (DMER)**
   - Link: https://dmer.maharashtra.gov.in/

2. **National Medical Commission**
   - Link: https://www.nmc.org.in/

## Banking/Financial Institution Scholarships:

1. **SBI Foundation Scholarships**
   - Link: https://www.sbifoundation.in/

2. **Canara Bank Scholarships**
   - Link: https://canarabank.com/

3. **IDFC First Bank Scholarships**
   - Link: https://www.idfcfirstbank.com/

## Government Department Links:

1. **Ministry of Social Justice & Empowerment**
   - Link: https://socialjustice.gov.in/

2. **Ministry of Tribal Affairs**
   - Link: https://tribal.gov.in/

3. **Ministry of Minority Affairs**
   - Link: https://minorityaffairs.gov.in/

4. **Department of Higher Education**
   - Link: https://www.education.gov.in/

MANDATORY WEBSITE LINKS SECTION:
When providing detailed scholarship information, include relevant links from this database:

**Key Scholarship Portals:**
• Maharashtra DBT Portal: https://mahadbtmahait.gov.in/
• National Scholarship Portal: https://scholarships.gov.in/
• Buddy4Study (Search Platform): https://buddy4study.com/
• DTE Maharashtra: https://dtemaharashtra.gov.in/
• AICTE Scholarships: https://www.aicte-india.org/schemes/students-development-schemes
• UGC Scholarships: https://www.ugc.ac.in/page/Scholarships-and-Fellowships.aspx
• Tata Trusts: https://www.tatatrusts.org/our-work/individual-grants-programme/education-grants
• Aditya Birla Scholarship: https://www.adityabirlascholars.com/

Include these links naturally in your response when mentioning relevant scholarships, not as a separate section.

MANDATORY COMMON MISTAKES SECTION:
Always include this section at the end of your response (before follow-up questions) to help students avoid application errors:
💡 While applying for scholarships, make sure to avoid these common mistakes:

📄 Document Mistakes
-Missing documents like caste, income, or marksheets
-Uploading blurry scans or wrong file formats (only PDF usually allowed)
-Using expired certificates (older than 1 year)

📅 Timing Mistakes
-Applying on the last day (site/server may crash)
-Forgetting to renew scholarship yearly
-Choosing the wrong academic year/semester

📚 Academic Info Errors
-Entering wrong marks or percentage
-Selecting incorrect course or college code (AISHE)

🏷️ Eligibility Mistakes
-Choosing the wrong category (SC/ST/OBC/Minority)
-Applying even when not eligible
-Submitting multiple/double applications

🏦 Bank Details Issues
-Using inactive or joint accounts
-Giving wrong IFSC or account number
-Not linking account with DBT

📱 Other Common Errors
-Using someone else’s mobile/email
-Forgetting login password or losing application ID
-Leaving the form incomplete and submitting

MANDATORY FOLLOW-UP QUESTIONS:
Always end your response with these personalized questions to help narrow down the best scholarships:

📋 **To help me suggest the most suitable scholarships for you, please share:**

1. **What's your category?** (SC/ST/OBC/VJNT/SBC/General/Minority)
2. **What's your family's annual income range?** (Below ₹1L, ₹1-2.5L, ₹2.5-8L, Above ₹8L)
3. **Are you a girl student?** (For gender-specific scholarships)
4. **Do you stay in a hostel?** (For hostel allowance schemes)
5. **What's your current CGPA/percentage?** (For merit-based scholarships)
6. **Are you from a minority community?** (Muslim/Christian/Sikh/Buddhist/Jain/Parsi)
7. **Do you have any disabilities?** (For specific disability scholarships)
8. **Are your parents ex-servicemen or freedom fighters?** (For special category scholarships)

💡 **The more details you share, the better I can match you with scholarships that fit your profile perfectly!**

USER QUERY: {user_input}

Please provide relevant information based on the user's query, select the most relevant scholarships from the comprehensive list above, include relevant website links naturally in your response, include the mandatory common mistakes section to help students avoid application errors, and ALWAYS end with the follow-up questions to gather more user details for personalized recommendations.
"""
def format_response(text):
    """
    Clean and format the response text
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Ensure proper heading formatting
    text = re.sub(r'(#{1,6})\s*([^\n]+)', r'\1 \2', text)
    
    # Clean up bullet points
    text = re.sub(r'^\s*[•·]\s*', '• ', text, flags=re.MULTILINE)
    
    # Remove trailing spaces
    lines = text.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    return text.strip()

def validate_input(user_input):
    """
    Validate and clean user input
    """
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
        # Get user input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_input = data.get('query', '')
        
        # Validate input
        is_valid, result = validate_input(user_input)
        if not is_valid:
            return jsonify({'error': result}), 400
        
        user_input = result
        
        # Check if Gemini API key is configured
        if not GEMINI_API_KEY:
            return jsonify({'error': 'GEMINI_API_KEY not configured'}), 500
        
        # Create prompt for Gemini
        full_prompt = create_scholarship_prompt(user_input)

        # Prepare request to Gemini API
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": full_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }

        params = {
            'key': GEMINI_API_KEY
        }

        # Make request to Gemini API
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            params=params,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            content = response.json()
            
            # Extract the generated text
            if 'candidates' in content and len(content['candidates']) > 0:
                generated_text = content['candidates'][0]['content']['parts'][0]['text']
                
                # Format the response
                formatted_text = format_response(generated_text)
                
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
                        'includes_common_mistakes': True
                    }
                })
            else:
                return jsonify({'error': 'No response generated from Gemini'}), 500
                
        else:
            error_message = f'Gemini API error: {response.status_code}'
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    error_message += f" - {error_detail['error'].get('message', 'Unknown error')}"
            except:
                error_message += f" - {response.text[:200]}"
            
            return jsonify({'error': error_message}), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout. Please try again.'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Maharashtra Scholarship Assistant is running',
        'gemini_configured': bool(GEMINI_API_KEY),
        'version': '2.2',
        'features': [
            'Gemini AI powered responses',
            'Maharashtra scholarship information',
            'Direct scholarship website links',
            'Comprehensive database with URLs',
            'Common application mistakes guidance',
            'Pro tips for successful applications',
            'Helpline numbers included',
            'No external search dependencies'
        ],
        'key_links': {
            'mahadbt': 'https://mahadbtmahait.gov.in/',
            'nsp': 'https://scholarships.gov.in/',
            'buddy4study': 'https://buddy4study.com/'
        }
    })

@app.route('/links', methods=['GET'])
def get_scholarship_links():
    """
    Endpoint to get all scholarship links
    """
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
    """
    New endpoint to get common scholarship application mistakes
    """
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
    """
    Test endpoint with a sample query
    """
    try:
        sample_query = "I am a BSc Computer Science 3rd year student. What scholarships are available for me?"
        
        # Create a test request
        test_request = {'query': sample_query}
        
        # Simulate the chat request
        with app.test_request_context('/chat', method='POST', json=test_request):
            return chat_with_gemini()
            
    except Exception as e:
        return jsonify({'error': f'Test error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Maharashtra Scholarship Assistant...")
    print(f"📊 GEMINI_API_KEY configured: {'✅ Yes' if GEMINI_API_KEY else '❌ No'}")
    print("\n📡 Available endpoints:")
    print("  POST /chat - Main chat endpoint for scholarship queries")
    print("  GET  /health - Health check and status")
    print("  GET  /links - Get all scholarship website links")
    print("  POST /test - Test endpoint with sample query")
    print("\n💡 Usage:")
    print("  curl -X POST http://localhost:5000/chat \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"query\": \"What scholarships are available for BSc students?\"}'")
    print("\n🔗 Key Features:")
    print("  ✅ Direct scholarship website links included")
    print("  ✅ Government portals (MahaDBT, NSP)")
    print("  ✅ Private scholarship platforms")
    print("  ✅ University-specific links")
    print("  ✅ Helpline numbers included")
    
    if not GEMINI_API_KEY:
        print("\n⚠️  WARNING: GEMINI_API_KEY not configured!")
        print("   Set your Gemini API key: export GEMINI_API_KEY='your_api_key_here'")
        print("   Get API key from: https://aistudio.google.com/app/apikey")
    else:
        print("\n✅ Ready to assist with Maharashtra scholarship queries with direct links!")
    
    print("\n" + "="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)