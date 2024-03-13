import os
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import streamlit as st

reports_list = []
folder_path = '2021_txt'
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        reports_list.append(text)

reports = ''

for text in reports_list: reports += '\n\n'+text

def main():

    if GOOGLE_API_KEY := st.text_input("GOOGLE API KEY:"):
        llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                temperature=0,google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=350)
        texts = text_splitter.split_text(reports)
        db = Chroma.from_texts(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        def generate_conclusion(observation):
            docs = retriever.get_relevant_documents(observation)

            prompt = f"""
        **Input:**

        * Radiology observation (text describing the case):\n {observation} \n
        * Similar reports retrieved using RAG: \n {docs}

        **Output:**

        * Conclusion of the radiology report indicating the medical problem that the patient has

        **Steps:**

        1. **Identify Key Findings:** Extract relevant anatomical structures, their appearance, and any abnormalities or variations from the observation text.
        2. **Compare with Similar Reports:** Examine the similar reports to identify any common findings or patterns that may support or contradict the current observation.
        3. **Establish Differential Diagnoses:** Consider possible medical conditions that could explain the key findings based on the observation and similar reports.
        4. **Rule Out Alternative Explanations:** Evaluate whether alternative causes, such as technical artifacts or patient positioning, could account for any unusual findings.
        5. **Formulate Conclusion:** Summarize the key findings, discuss the differential diagnoses, rule out alternative explanations, and provide a clear and detailed statement of the suspected medical problem based on the evidence in french.(You must give only the conclusion do not rewrite the input)

        **Example1:**

        **Input :**

        * **Observation radiologique (texte décrivant le cas) :** TECHNIQUE : Examen réalisé sur un échographe LOGIQ P9 GE, avec une sonde convexe
        fréquence de 1 à 6 HZ.
        CLINIQUE : Patiente adressée pour échographie abdomino-pelvienne.
        RESULTAT :
        Aérocolie importante.
        Foie : de taille normale, d’échostructure homogène, échogène aux contours
        réguliers, sans image nodulaire.
        lit vésiculaire libre
        Absence de dilatation des VBIH.
        VBP de calibre normal.
        Pancréas : vu, tête et corps, de taille normale, d’échostructure habituelle
        Rate : de volume habituel, d’échostructure et d’échogénicité homogène
        Reins : en position lombaire, de taille normale, aux contours réguliers.
        Bonne différentiation cortico-médullaire.
        Sans dilatation des cavités pyélocalicielles, ni lithiase.
        Vessie : en bonne réplétion, à plage transonore, à paroi épaissie de 08 mm.
        prostate: de volume augmentée mesurant 38 cm3, d'échostructure homogène, aux
        contours réguliers.
        Cul de sac de Douglas libre.
        Absence de liquide intra péritonéal.
        Absence d’adénopathie profonde.
        Absence d’épaississement digestif spontanément décelable.

        **Output :**

        "Conclusion : Examen échographique en faveur de:
        stéatose hépatique homogène, importante.
        hypertrophie prostatique de 38 cm3.
        une aérocolie"


        **Example2:**

        **Input :**

        * **Observation radiologique (texte décrivant le cas) :** CLINIQUE : Patiente adressée pour masse sous cutanée.
        TECHNIQUE : Examen réalisé sur un échographe GE logiq P9, avec une sonde linéaire, fréquence de
        6 à 15 HZ.
        RESULTAT :
        Mise en évidence d'une formation kystique, volumineuse, mesurant 40 x 26 x 18 mm,
        soit un volume de 10 cc, à contenu transonore , à paroi fine.
        Absence d’épanchement intra articulaire
        Respect de la graisse sou cutanée.

        **Output :**

        "Conclusion :Examen échographique en faveur d'un kyste poplité gauche simple de
        40 x 26 x 18 mm."

        **Example3:**

        **Input :**

        * **Observation radiologique (texte décrivant le cas) :** Technique : Examen réalisé sur un scanner GE 64 barrettes, selon le protocole crâne sans injection du PDCI,
        acquisition en coupes axiales, MRP et 3D.
        Motif : céphalée
        Résultats :
        • Absence d’anomalie de densité parenchymateuse cérébrale, cérébelleuse ou du tronc cérébral
        • Absence de dilatation du système ventriculaire.
        • Structures médianes en place.
        • Absence de collection péri cérébrale.
        • Absence de lésion osseuse.
        • Bonne pneumatisation des sinus.


        **Output :**

        "Conclusion :Examen TDM cérébral sans particularité."

        """

            result = llm.invoke(prompt)
            
            return result.content
        
        prompt_input = st.text_input("The observation:")
        if prompt_input != "":
            conclusion = generate_conclusion(prompt_input)

            with st.expander("Conclusion"):
                st.write(conclusion)

if __name__ == '__main__':
    main()