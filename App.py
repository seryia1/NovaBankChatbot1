import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page configuration
st.set_page_config(
    page_title="NovaBank - AI-Powered Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
        /* Main styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Header styling */
        .bank-header {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .bank-logo {
            font-size: 2rem;
            font-weight: bold;
            color: #1E88E5;
        }
        
        .bank-logo span {
            color: #333;
        }
        
        /* Card styling */
        .bank-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            transition: transform 0.3s;
        }
        
        .bank-card:hover {
            transform: translateY(-5px);
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #1E88E5;
        }
        
        /* Feature box styling */
        .feature-box {
            background-color: #f0f7ff;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1E88E5;
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        
        .stButton>button:hover {
            background-color: #1565C0;
        }
        
        /* Chat container */
        .chat-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
        }
        
        .chat-header {
            background-color: #1E88E5;
            color: white;
            padding: 10px 15px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .chat-body {
            height: 300px;
            overflow-y: auto;
            padding: 15px;
        }
        
        .chat-footer {
            padding: 10px 15px;
            border-top: 1px solid #eee;
        }
        
        /* Chat toggle button */
        .chat-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background-color: #1E88E5;
            color: white;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            cursor: pointer;
            z-index: 1001;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob {display: none;}
        
        /* Testimonial styling */
        .testimonial {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            margin-bottom: 1rem;
            font-style: italic;
        }
        
        .testimonial-author {
            font-weight: bold;
            text-align: right;
            margin-top: 0.5rem;
        }
        
        /* Hero section */
        .hero-section {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        
        .hero-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        .hero-subtitle {
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
            opacity: 0.9;
        }
        
        /* Chat message styling */
        .user-message {
            background-color: #1E88E5;
            color: white;
            padding: 10px 15px;
            border-radius: 18px 18px 0 18px;
            margin-bottom: 10px;
            max-width: 80%;
            margin-left: auto;
            word-wrap: break-word;
        }
        
        .bot-message {
            background-color: #f0f0f0;
            color: #333;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 0;
            margin-bottom: 10px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        /* Navigation styling */
        .nav-button {
            background-color: transparent;
            border: none;
            color: #1E88E5;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .nav-button:hover {
            background-color: #f0f7ff;
        }
        
        .nav-button.active {
            background-color: #1E88E5;
            color: white;
        }
        
        .nav-container {
            display: flex;
            justify-content: space-between;
            background-color: white;
            padding: 0.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess dataset
@st.cache_data
def load_dataset():
    """Load the complete NovaBank Q&A dataset"""
    
    # Your complete dataset embedded directly
    qa_data = """
🏦 GENERAL BANK OVERVIEW
Q: What is NovaBank?
A: NovaBank is a modern digital bank offering checking, savings, credit, loans, and investment services designed to make banking simple, secure, and accessible from your phone.

Q: What types of accounts does NovaBank offer?
A: NovaBank offers Checking Accounts, Savings Accounts, Student Accounts, Business Accounts, and Joint Accounts.

Q: Where is NovaBank located?
A: NovaBank is a fully digital bank, so you can access your account online or through our mobile app anytime, anywhere.

Q: What are NovaBank's business hours?
A: Our digital services are available 24/7. For live support, our agents are available Monday–Friday, 8 AM–8 PM, and Saturday, 9 AM–5 PM.

💸 ACCOUNT & BALANCE
Q: How do I check my balance?
A: You can check your balance by logging into the NovaBank app or texting 'BAL' to 29292.

Q: How do I open a NovaBank account?
A: You can open an account through our website or mobile app. You'll need a valid ID, proof of address, and a Social Security Number or Tax ID.

Q: How can I close my NovaBank account?
A: To close your account, please contact customer service through the app or call our hotline. Make sure your balance is $0 and all pending transactions are cleared.

Q: Can I get a joint account?
A: Yes! NovaBank offers joint accounts. You and your co-holder can apply through the app under 'Open Joint Account.'

💳 DEPOSITS, WITHDRAWALS, TRANSFERS
Q: How do I make a deposit?
A: You can deposit money through the app, at an ATM, or by linking an external account.

Q: How long does it take for a deposit to show up?
A: Most deposits reflect immediately, but some external transfers may take up to 2 business days.

Q: Can I withdraw cash?
A: Yes, you can withdraw cash at any NovaBank ATM or partner ATM. Use your debit card and PIN.

Q: How do I transfer money?
A: You can transfer money between NovaBank accounts, to external accounts, or using services like Zelle, all through the app.

Q: Can I set up recurring transfers or payments?
A: Yes, go to the 'Bill Pay' section or 'Transfers' in the app and select the recurring option.

📄 STATEMENTS, DOCUMENTS, REPORTS
Q: I need a PDF bank statement.
A: Sure. Please tell me which month or date range you need the statement for.

Q: How do I report a fraud?
A: Tap "Report an Issue" in the app or contact support immediately. You can also reply here with the suspicious transaction details.

Q: How can I dispute a transaction?
A: Go to the 'Transactions' section in the app, select the transaction, and tap 'Dispute.' You can also call customer service for urgent cases.

🏠 LOANS, MORTGAGES, CREDIT
Q: What's the interest rate on personal loans?
A: Our personal loans start at 6.5% APR, depending on your credit profile.

Q: I want to apply for a mortgage.
A: Great! Please provide your income, employment status, and the home price, and we'll guide you through the next steps.

Q: How can I increase my credit limit?
A: We review your account history and credit score. You can apply under the 'Credit' section in the app.

Q: What documents are needed for a loan?
A: You'll typically need proof of income, valid ID, recent bank statements, and employment verification.

Q: Does NovaBank offer student loans?
A: Yes, NovaBank offers competitive student loan rates. You can apply online or through the app.

🛡 SECURITY & SETTINGS
Q: How do I reset my password?
A: Tap 'Forgot Password' on the login screen of the app or website, and follow the reset instructions.

Q: How do I update my contact information?
A: Go to the 'Settings' section in the app, then 'Profile,' and update your phone, email, or address.

Q: Is NovaBank safe?
A: NovaBank uses advanced encryption, multi-factor authentication, and FDIC insurance up to $250,000 per account.

📊 CUSTOMER-SPECIFIC REQUESTS
Q: What are Alice's latest transactions?
A: Here are Alice's recent transactions:
- Purchase of $22.98 at Uber
- Transfer of $462.8
- Deposit of $322.94
- Purchase of $554.51 at Netflix
- Transfer of $518.55
- Withdrawal of $273.64
- Transfer of $963.48
- Transfer of $314.71
- Loan Payment of $51.0
- Withdrawal of $890.61

Q: What are Bob's latest transactions?
A: Here are Bob's recent transactions:
- Purchase of $782.81 at Walmart
- Deposit of $122.39
- Transfer of $709.95
- Transfer of $272.5
- Deposit of $539.54
- Loan Payment of $558.47
- Withdrawal of $544.82
- Loan Payment of $336.38
- Purchase of $672.08 at Starbucks
- Loan Payment of $299.74

Q: What are Carlos's latest transactions?
A: Here are Carlos's recent transactions:
- Deposit of $461.86
- Purchase of $599.47 at Apple Store
- Withdrawal of $581.31
- Transfer of $462.37
- Purchase of $704.01 at Uber
- Transfer of $643.87
- Deposit of $642.63
- Purchase of $362.57 at Walmart
- Purchase of $73.84 at Uber
- Deposit of $507.74

Q: What are Diana's latest transactions?
A: Here are Diana's recent transactions:
- Transfer of $285.66
- Transfer of $993.74
- Purchase of $105.7 at Walmart
- Withdrawal of $807.42
- Transfer of $610.36
- Transfer of $837.34
- Transfer of $262.9
- Loan Payment of $446.4
- Purchase of $91.74 at Netflix
- Transfer of $883.69

Q: What are Evelyn's latest transactions?
A: Here are Evelyn's recent transactions:
- Transfer of $773.25
- Withdrawal of $389.06
- Deposit of $102.07
- Loan Payment of $538.51
- Transfer of $408.87
- Withdrawal of $308.7
- Deposit of $598.49
- Withdrawal of $923.72
- Purchase of $852.39 at Uber
- Deposit of $435.56
"""
    
    # Parse the dataset
    questions = []
    answers = []
    
    # Split by double newlines to get Q&A pairs
    pairs = qa_data.strip().split('\n\n')
    
    for pair in pairs:
        lines = pair.strip().split('\n')
        question = None
        answer_lines = []
        
        for line in lines:
            if line.startswith('Q: '):
                question = line[3:].strip()
            elif line.startswith('A: '):
                answer_lines.append(line[3:].strip())
            elif question and line.strip() and not line.startswith('🏦') and not line.startswith('💸') and not line.startswith('💳') and not line.startswith('📄') and not line.startswith('🏠') and not line.startswith('🛡') and not line.startswith('📊'):
                # This is a continuation of the answer
                answer_lines.append(line.strip())
        
        if question and answer_lines:
            questions.append(question)
            answers.append(' '.join(answer_lines))
    
    return questions, answers
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def get_most_relevant_answer(user_query, questions, answers):
    all_text = questions + [user_query]
    all_text = [preprocess(t) for t in all_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_text)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_index = np.argmax(similarity_scores)
    return answers[best_match_index]

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi there! I'm NovaBot, your AI banking assistant. How can I help you today?"}
    ]

if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Load dataset
questions, answers = load_dataset("novabank_dataset.txt")

# Apply custom CSS
local_css()

# Header
st.markdown("""
<div class="bank-header">
    <div class="bank-logo">Nova<span>Bank</span></div>
</div>
""", unsafe_allow_html=True)

# Navigation using standard Streamlit components
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    home_btn = st.button("🏠 Home")
    if home_btn:
        st.session_state.current_page = "Home"
        st.rerun()  # FIXED: Changed from st.experimental_rerun()

with col2:
    accounts_btn = st.button("💳 Accounts")
    if accounts_btn:
        st.session_state.current_page = "Accounts"
        st.rerun()  # FIXED: Changed from st.experimental_rerun()

with col3:
    loans_btn = st.button("💰 Loans")
    if loans_btn:
        st.session_state.current_page = "Loans"
        st.rerun()  # FIXED: Changed from st.experimental_rerun()

with col4:
    investments_btn = st.button("📈 Investments")
    if investments_btn:
        st.session_state.current_page = "Investments"
        st.rerun()  # FIXED: Changed from st.experimental_rerun()

with col5:
    services_btn = st.button("⚙️ Services")
    if services_btn:
        st.session_state.current_page = "Services"
        st.rerun()  # FIXED: Changed from st.experimental_rerun()

with col6:
    about_btn = st.button("ℹ️ About")
    if about_btn:
        st.session_state.current_page = "About"
        st.rerun()  # FIXED: Changed from st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Display current page
selected = st.session_state.current_page

# Home Page
if selected == "Home":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Banking Reimagined</div>
        <div class="hero-subtitle">Experience the future of banking with NovaBank's AI-powered solutions.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Smart Banking</div>
            <p>Experience banking powered by artificial intelligence that learns and adapts to your financial habits.</p>
            <div class="feature-box">
                <strong>✓ AI-Powered Insights</strong><br>
                <strong>✓ Personalized Recommendations</strong><br>
                <strong>✓ Automated Savings</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Secure Transactions</div>
            <p>Your security is our priority with state-of-the-art encryption and multi-factor authentication.</p>
            <div class="feature-box">
                <strong>✓ Biometric Authentication</strong><br>
                <strong>✓ Real-time Fraud Detection</strong><br>
                <strong>✓ End-to-End Encryption</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">24/7 Support</div>
            <p>Get help anytime with our AI assistant and dedicated support team available around the clock.</p>
            <div class="feature-box">
                <strong>✓ AI Chatbot Assistant</strong><br>
                <strong>✓ Live Video Banking</strong><br>
                <strong>✓ Dedicated Advisors</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Testimonials
    st.subheader("What Our Customers Say")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="testimonial">
            "NovaBank has completely transformed my banking experience. The AI assistant is incredibly helpful and the app is so intuitive!"
            <div class="testimonial-author">- Alice Johnson</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="testimonial">
            "The loan application process was so smooth. I got approved in minutes and the funds were in my account the same day!"
            <div class="testimonial-author">- Bob Smith</div>
        </div>
        """, unsafe_allow_html=True)

# Accounts Page
elif selected == "Accounts":
    st.header("Banking Accounts")
    
    tab1, tab2, tab3 = st.tabs(["Checking", "Savings", "Business"])
    
    with tab1:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">NovaCheck Premium</div>
            <p>Our flagship checking account with premium benefits and AI-powered financial insights.</p>
            <div class="feature-box">
                <strong>✓ No monthly fees</strong><br>
                <strong>✓ Free ATM withdrawals worldwide</strong><br>
                <strong>✓ AI-powered spending insights</strong><br>
                <strong>✓ Cashback on everyday purchases</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Open Checking Account", key="open_checking")
        
    with tab2:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">NovaGrow Savings</div>
            <p>Watch your money grow with our high-yield savings account featuring AI-optimized interest rates.</p>
            <div class="feature-box">
                <strong>✓ 3.5% APY</strong><br>
                <strong>✓ No minimum balance</strong><br>
                <strong>✓ Automated savings goals</strong><br>
                <strong>✓ Smart round-up feature</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Open Savings Account", key="open_savings")
        
    with tab3:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">NovaBiz Account</div>
            <p>Designed for businesses of all sizes with powerful tools to manage your company finances.</p>
            <div class="feature-box">
                <strong>✓ Free business transactions</strong><br>
                <strong>✓ Integrated invoicing</strong><br>
                <strong>✓ Employee expense cards</strong><br>
                <strong>✓ Business financial insights</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Open Business Account", key="open_business")

# Loans Page
elif selected == "Loans":
    st.header("Loan Products")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Personal Loans</div>
            <p>Flexible personal loans with competitive rates and quick approval.</p>
            <div class="feature-box">
                <strong>✓ Borrow up to $50,000</strong><br>
                <strong>✓ Rates from 4.99% APR</strong><br>
                <strong>✓ Terms from 12-60 months</strong><br>
                <strong>✓ No prepayment penalties</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Auto Loans</div>
            <p>Drive away in your dream car with our competitive auto financing.</p>
            <div class="feature-box">
                <strong>✓ New and used vehicles</strong><br>
                <strong>✓ Rates from 3.49% APR</strong><br>
                <strong>✓ Up to 84-month terms</strong><br>
                <strong>✓ Quick online approval</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Home Mortgages</div>
            <p>Find your dream home with our flexible mortgage options.</p>
            <div class="feature-box">
                <strong>✓ Fixed and adjustable rates</strong><br>
                <strong>✓ First-time homebuyer programs</strong><br>
                <strong>✓ Refinancing options</strong><br>
                <strong>✓ Digital application process</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Business Loans</div>
            <p>Fuel your business growth with our flexible financing solutions.</p>
            <div class="feature-box">
                <strong>✓ Working capital loans</strong><br>
                <strong>✓ Equipment financing</strong><br>
                <strong>✓ Commercial real estate</strong><br>
                <strong>✓ SBA loan options</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.button("Apply for a Loan", key="apply_loan")

# Investments Page
elif selected == "Investments":
    st.header("Investment Solutions")
    
    tab1, tab2, tab3 = st.tabs(["Retirement", "Brokerage", "Wealth Management"])
    
    with tab1:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">NovaRetire IRA</div>
            <p>Plan for your future with our tax-advantaged retirement accounts.</p>
            <div class="feature-box">
                <strong>✓ Traditional and Roth IRAs</strong><br>
                <strong>✓ AI-powered retirement planning</strong><br>
                <strong>✓ Automatic contributions</strong><br>
                <strong>✓ Low-fee investment options</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Open Retirement Account", key="open_retirement")
        
    with tab2:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">NovaTrade</div>
            <p>Invest in stocks, ETFs, and more with our intuitive trading platform.</p>
            <div class="feature-box">
                <strong>✓ Commission-free trades</strong><br>
                <strong>✓ Fractional shares</strong><br>
                <strong>✓ Advanced research tools</strong><br>
                <strong>✓ AI-powered investment suggestions</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Open Brokerage Account", key="open_brokerage")
        
    with tab3:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">NovaWealth</div>
            <p>Comprehensive wealth management with personalized guidance.</p>
            <div class="feature-box">
                <strong>✓ Dedicated wealth advisor</strong><br>
                <strong>✓ Custom investment strategies</strong><br>
                <strong>✓ Tax optimization</strong><br>
                <strong>✓ Estate planning</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Schedule Consultation", key="schedule_consultation")

# Services Page
elif selected == "Services":
    st.header("Banking Services")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Online & Mobile Banking</div>
            <p>Manage your finances anytime, anywhere with our digital banking solutions.</p>
            <div class="feature-box">
                <strong>✓ 24/7 account access</strong><br>
                <strong>✓ Mobile check deposit</strong><br>
                <strong>✓ Bill pay & transfers</strong><br>
                <strong>✓ Financial insights dashboard</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">International Services</div>
            <p>Global banking solutions for travelers and international customers.</p>
            <div class="feature-box">
                <strong>✓ Multi-currency accounts</strong><br>
                <strong>✓ International wire transfers</strong><br>
                <strong>✓ No foreign transaction fees</strong><br>
                <strong>✓ Global ATM access</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Insurance Products</div>
            <p>Protect what matters most with our comprehensive insurance offerings.</p>
            <div class="feature-box">
                <strong>✓ Life insurance</strong><br>
                <strong>✓ Home & auto insurance</strong><br>
                <strong>✓ Health insurance</strong><br>
                <strong>✓ Business insurance</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Financial Planning</div>
            <p>Plan for your future with our AI-powered financial planning tools.</p>
            <div class="feature-box">
                <strong>✓ Retirement planning</strong><br>
                <strong>✓ College savings</strong><br>
                <strong>✓ Budget optimization</strong><br>
                <strong>✓ Goal-based planning</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# About Page
elif selected == "About":
    st.header("About NovaBank")
    
    st.markdown("""
    <div class="bank-card">
        <div class="card-title">Our Story</div>
        <p>NovaBank was founded in 2020 with a mission to revolutionize banking through artificial intelligence and cutting-edge technology. We believe that banking should be simple, transparent, and personalized to each customer's unique financial journey.</p>
        <p>As a 100% digital bank, we've eliminated the overhead costs of traditional brick-and-mortar branches, allowing us to offer better rates, lower fees, and innovative features that traditional banks simply can't match.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Our Mission</div>
            <p>To empower people to achieve financial wellness through AI-driven insights, personalized guidance, and innovative banking solutions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="bank-card">
            <div class="card-title">Our Vision</div>
            <p>To become the world's leading AI-powered financial institution, setting new standards for how people interact with their money.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="bank-card">
        <div class="card-title">NovaBot AI Assistant</div>
        <p>At the heart of our banking experience is NovaBot, our advanced AI assistant. NovaBot can answer questions about your accounts, help you make transactions, provide financial insights, and much more.</p>
        <p>Powered by natural language processing and machine learning, NovaBot understands your questions and provides personalized responses based on your financial situation and history.</p>
    </div>
    """, unsafe_allow_html=True)

# Chat toggle button (using Streamlit components)
chat_col = st.sidebar.container()

with chat_col:
    # Only show the chat toggle if chat is not visible
    if not st.session_state.chat_visible:
        if st.button("💬 Chat with NovaBot", key="chat_toggle"):
            st.session_state.chat_visible = True
            st.rerun()  # FIXED: Changed from st.experimental_rerun()

# Chat interface
if st.session_state.chat_visible:
    st.sidebar.markdown("### 🤖 NovaBot Assistant")
    
    # Display chat messages
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.sidebar.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    with st.sidebar.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message:", placeholder="Ask me anything about banking...")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get bot response
            bot_response = get_most_relevant_answer(user_input, questions, answers)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            
            # Force a rerun to update the chat display
            st.rerun()  # FIXED: Changed from st.experimental_rerun()
    
    # Close chat button
    if st.sidebar.button("Close Chat", key="close_chat"):
        st.session_state.chat_visible = False
        st.rerun()  # FIXED: Changed from st.experimental_rerun()

# Run the main function
if __name__ == "__main__":
    pass  # Main logic is now in the Streamlit app flow
