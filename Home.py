import streamlit as st
from PIL import Image
import pandas as pd
from rdkit import Chem
import streamlit.components.v1 as components

from pandas.io.formats.style import Styler
from utils import (smiles_to_mol, mol_file_to_mol, 
                   draw_molecule, mol_to_tensor_graph, get_model_predictions, get_model_predictions2, get_model_predictions3, get_model_predictions4, get_model_predictions5)

# ----------- General things
valid_molecule = True
loaded_molecule = None
selection = None
submit = None

# ----------- Sidebar
st.sidebar.image("img/GTransCYPs-logo.png")
page = st.sidebar.selectbox('Menu', ["Home", "Documentation", "Server"])

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return True
    except:
        return False

if page == "Home":
        st.subheader('Introduction')   
        st.write('In drug-drug interactions, the inhibition of cytochrome P450 (CYP450) enzymes plays a crucial role in drug efficacy, toxicity, and potential interactions. These enzymes are responsible for metabolizing numerous drugs in the body. If the activity of these enzymes is hindered by one drug, it can impact the metabolism of other drugs, potentially altering the drugs response and raising the risk of toxicity. Among the 57 commonly found CYP450 isoforms in the human liver, five of them – namely 1A2, 2C9, 2C19, 2D6, and 3A4 – play critical roles in most drug metabolism processes in the human body.')
        st.write('In-silico approach is appealing as it can be utilized at the early stages of drug discovery pathways, reducing the number of wet-lab experiment studies needed for selecting new drug candidates and thus minimizing costs. Here, we proposed a novel deep learning model, an improved graph transformer neural network with attention mechanism for predicting CYP450 inhibitors (GTransCYPs). The GTransCYPs is an effective tool to identify potential inhibitory compounds against CYP450 for further wet-lab experiment validation.')
        st.write('This web server is a practical and convenient service tool created based on the GTransCYPs model, which can help researchers predict molecules with inhibitory activities on five CYP450 isozymes through a friendly interface.')

elif page == "Documentation":
        st.subheader('How to use')
        st.write('1. Select menu "Server"') 
        st.write('2. The user must first select the predictor to be executed from the side bar. Predictor (CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4)') 
        st.write('This prediction option require a CSV file, and a sample of input file can be downloaded from the system for direction purposes')
        st.image('img/step1.png')
        st.write('3. Upload a CSV file containing the list molecules for virtual screening') 
        st.write('4. Click "Prediction"') 
        st.write('Displaying the Prediction Results. After the prediction is complete, the results will be displayed on the webpage.') 
        st.image('img/step3.png') 
        st.image('img/step5.png') 
        st.write("Materials code and datasets can be obtained from [here](https://github.com/zonwoo/GTransCYPs)")
     
elif page == "Server":
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"])      
        with tab1:
            st.subheader('CYP1A2')
            with open('example_molecules1a2.csv') as f:
                st.download_button('Download Example input file', f,'example_molecules1a2.csv')
            st.header('Upload CSV file')
            uploaded_file_1a2 = st.file_uploader("Upload your input file", type=['CSV'], key='fileupload')
            if st.button('Prediction', key='prediction'):
                if uploaded_file_1a2 is not None:
                    data = pd.read_csv(uploaded_file_1a2)
                    validity =[]
                    for index, row in data.iterrows():
                        validity.append(is_valid_smiles(row['Drug']))
                    if data.empty:
                        st.warning("File Empty Error: Uploaded file is empty. Please check and try again..")
                    elif any(x == False for x in validity):
                        st.warning("Please check the provided file for content verification; an invalid SMILES code has been detected")                    
                    else:
                        try:
                            with st.spinner('Prediction...'):
                                prediction_output = get_model_predictions(uploaded_file_1a2,data)
                                styler = Styler(prediction_output)                                
                                styler.set_table_styles(
                                    [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
                                )
                                table_title="<style> </style><div class='css-x9krnl';style='display:block;'><p style='font-family:times;font-weight:bold;padding-left:20%;'>CYP450 1A2 Inhibition prediction output</p></div>"
                                st.markdown(table_title, unsafe_allow_html=True)
                                st.dataframe(styler)
                                description = "<style> </style><div class='css-x9krnl ';style='display:block;'><p style='font-family:times;font-size:13px;margin-top:-20px'><b>PConfInh (%):</b> GTransCYPs model's Prediction confidence for molecule to be an inhibitor.</p></div>"                                
                                st.markdown(description, unsafe_allow_html=True)                                
                        except Exception as e:
                            st.warning(e)
                            st.warning("Unexpected Error: Please contact the developer of the tool for support.")
                else:
                    st.warning('Input Error: Upload a csv file containing valid SMILES code before pressing "Prediction" buttton.')

        with tab2:
            st.subheader('CYP2C9')
            with open('example_molecules1a2.csv') as f:
                st.download_button('Download Example input file', f,'example_molecules2c9.csv')
            # ----------- Inputs
            st.header('Upload CSV file')
            uploaded_file_2c9 = st.file_uploader("Upload your input file", type=['CSV'], key='fileupload2')
            if st.button('Prediction', key='prediction2'):
                if uploaded_file_2c9 is not None:
                    data = pd.read_csv(uploaded_file_2c9)
                    validity =[]
                    for index, row in data.iterrows():
                        validity.append(is_valid_smiles(row['Drug']))
                    if data.empty:
                        st.warning("Empty field Error: the file you uploaded was found to be empty, please verify and try again.")
                    elif any(x == False for x in validity):
                        st.warning("Please verify the content of the file provided, an invalid SMILES code is detected.")                    
                    else:
                        try:
                            with st.spinner('Prediction...'):
                                prediction_output = get_model_predictions2(uploaded_file_2c9,data)
                                styler = Styler(prediction_output)
                                styler.set_table_styles(
                                    [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
                                )
                                table_title="<style> </style><div class='css-x9krnl';style='display:block;'><p style='font-family:times;font-weight:bold;padding-left:20%;'>CYP450 2C9 Inhibition prediction output</p></div>"
                                st.markdown(table_title, unsafe_allow_html=True)
                                st.dataframe(styler)
                                description = "<style> </style><div class='css-x9krnl ';style='display:block;'><p style='font-family:times;font-size:13px;margin-top:-20px'><b>PConfInh (%):</b> GTransCYPs model's Prediction confidence for molecule to be an inhibitor.</p></div>"
                                st.markdown(description, unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(e)
                            st.warning("Unexpected Error: Please contact the developer of the tool for support.")
                else:
                    st.warning('Input Error: Upload a csv file containing valid SMILES code before pressing "Prediction" buttton.')

        with tab3:
            st.subheader('CYP2C19')
            with open('example_molecules1a2.csv') as f:
                st.download_button('Download Example input file', f,'example_molecules2c19.csv')
            st.header('Upload CSV file')
            uploaded_file_2c19= st.file_uploader("Upload your input file", type=['CSV'], key='fileupload3')
            if st.button('Prediction', key='prediction3'):
                if uploaded_file_2c19 is not None:
                    data = pd.read_csv(uploaded_file_2c19)
                    validity =[]
                    for index, row in data.iterrows():
                        validity.append(is_valid_smiles(row['Drug']))
                    if data.empty:
                        st.warning("File Empty Error: Uploaded file is empty. Please check and try again..")
                    elif any(x == False for x in validity):
                        st.warning("Please check the provided file for content verification; an invalid SMILES code has been detected")                    
                    else:
                        try:
                            with st.spinner('Prediction...'):
                                prediction_output = get_model_predictions3(uploaded_file_2c19,data)

                                styler = Styler(prediction_output)
                                
                                styler.set_table_styles(
                                    [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
                                )
                                table_title="<style> </style><div class='css-x9krnl';style='display:block;'><p style='font-family:times;font-weight:bold;padding-left:20%;'>CYP450 2C19 Inhibition prediction output</p></div>"
                                st.markdown(table_title, unsafe_allow_html=True)
                                st.dataframe(styler)
                                description = "<style> </style><div class='css-x9krnl ';style='display:block;'><p style='font-family:times;font-size:13px;margin-top:-20px'><b>PConfInh (%):</b> GTransCYPs model's Prediction confidence for molecule to be an inhibitor.</p></div>"                                
                                st.markdown(description, unsafe_allow_html=True)                                
                        except Exception as e:
                            st.warning(e)
                            st.warning("Unexpected Error: Please contact the developer of the tool for support.")
                else:
                    st.warning('Input Error: Upload a csv file containing valid SMILES code before pressing "Prediction" buttton.')
          
        with tab4:
            st.subheader('CYP2D6')
            with open('example_molecules1a2.csv') as f:
                st.download_button('Download Example input file', f,'example_molecules2d6.csv')
            st.header('Upload CSV file')
            uploaded_file_2d6= st.file_uploader("Upload your input file", type=['CSV'], key='fileupload4')
            if st.button('Prediction', key='prediction4'):
                if uploaded_file_2d6 is not None:
                    data = pd.read_csv(uploaded_file_2d6)
                    validity =[]
                    for index, row in data.iterrows():
                        validity.append(is_valid_smiles(row['Drug']))
                    if data.empty:
                        st.warning("File Empty Error: Uploaded file is empty. Please check and try again..")
                    elif any(x == False for x in validity):
                        st.warning("Please check the provided file for content verification; an invalid SMILES code has been detected")                   
                    else:
                        try:
                            with st.spinner('Prediction...'):
                                prediction_output = get_model_predictions4(uploaded_file_2d6,data)
                                styler = Styler(prediction_output)                                
                                styler.set_table_styles(
                                    [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
                                )
                                table_title="<style> </style><div class='css-x9krnl';style='display:block;'><p style='font-family:times;font-weight:bold;padding-left:20%;'>CYP450 2D6 Inhibition prediction output</p></div>"
                                st.markdown(table_title, unsafe_allow_html=True)
                                st.dataframe(styler)
                                description = "<style> </style><div class='css-x9krnl ';style='display:block;'><p style='font-family:times;font-size:13px;margin-top:-20px'><b>PConfInh (%):</b> GTransCYPs model's Prediction confidence for molecule to be an inhibitor.</p></div>"                                
                                st.markdown(description, unsafe_allow_html=True)                                
                        except Exception as e:
                            st.warning(e)
                            st.warning("Unexpected Error: Please contact the developer of the tool for support.")
                else:
                    st.warning('Input Error: Upload a csv file containing valid SMILES code before pressing "Prediction" buttton.')

        with tab5:
            st.subheader('CYP3A4')
            with open('example_molecules1a2.csv') as f:
                st.download_button('Download Example input file', f,'example_molecules3a4.csv')
            # ----------- Inputs
            st.header('Upload CSV file')
            uploaded_file_3a4= st.file_uploader("Upload your input file", type=['CSV'], key='fileupload5')
            if st.button('Prediction', key='prediction5'):
                if uploaded_file_3a4 is not None:
                    data = pd.read_csv(uploaded_file_3a4)
                    validity =[]
                    for index, row in data.iterrows():
                        validity.append(is_valid_smiles(row['Drug']))
                    if data.empty:
                        st.warning("File Empty Error: Uploaded file is empty. Please check and try again..")
                    elif any(x == False for x in validity):
                        st.warning("Please check the provided file for content verification; an invalid SMILES code has been detected")                    
                    else:
                        try:
                            with st.spinner('Prediction...'):
                                prediction_output = get_model_predictions5(uploaded_file_3a4,data)
                                styler = Styler(prediction_output)                                
                                styler.set_table_styles(
                                    [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
                                )
                                table_title="<style> </style><div class='css-x9krnl';style='display:block;'><p style='font-family:times;font-weight:bold;padding-left:20%;'>CYP450 3A4 Inhibition prediction output</p></div>"
                                st.markdown(table_title, unsafe_allow_html=True)
                                st.dataframe(styler)
                                description = "<style> </style><div class='css-x9krnl ';style='display:block;'><p style='font-family:times;font-size:13px;margin-top:-20px'><b>PConfInh (%):</b> GTransCYPs model's Prediction confidence for molecule to be an inhibitor.</p></div>"                                
                                st.markdown(description, unsafe_allow_html=True)                                
                        except Exception as e:
                            st.warning(e)
                            st.warning("Unexpected Error: Please contact the developer of the tool for support.")
                else:
                    st.warning('Input Error: Upload a csv file containing valid SMILES code before pressing "Prediction" buttton.')
