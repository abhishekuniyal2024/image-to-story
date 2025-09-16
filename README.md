## setup instructions

1. **Create a virtual enviroemnt**
python3.10 -m venv venv

2. **Activate the enviroment**
source venv/bin/activate

3. **Install dependency**
pip install -r requirements.txt

4. **Create a .env file** 
Add your GROQ API Key inside .env:
GROQ_API_KEY=your_groq_key_here


## Usage
1. Make sure your virtual environment is activated:
source venv/bin/activate

2. Run the Streamlit script
streamlit run app.py


## app.py: generate story from image.
## app2.py: generate story from image, will add text automatically
## app3.py: make change in the app2.py to get what client want
## app3a.py: fix audio in app3.py
## app4.py: this code is app7a.py from iamge to story local.