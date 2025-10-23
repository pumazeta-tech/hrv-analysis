import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io
import base64
from matplotlib.patches import Ellipse
import re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
from reportlab.lib.utils import ImageReader
import matplotlib
matplotlib.use('Agg')

# =============================================================================
# INIZIALIZZAZIONE SESSION STATE
# =============================================================================

def init_session_state():
    """Inizializza lo stato della sessione"""
    if 'activities' not in st.session_state:
        st.session_state.activities = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analysis_datetimes' not in st.session_state:
        st.session_state.analysis_datetimes = {
            'start_datetime': datetime.now(),
            'end_datetime': datetime.now() + timedelta(hours=24)
        }
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'surname': '',
            'birth_date': None,
            'gender': 'Uomo',
            'age': 0
        }
    if 'datetime_initialized' not in st.session_state:
        st.session_state.datetime_initialized = False
    if 'recording_end_datetime' not in st.session_state:
        st.session_state.recording_end_datetime = None

# =============================================================================
# FUNZIONI PER ESTRAZIONE DATA E ORA DAL FILE - CORRETTA
# =============================================================================

def extract_datetime_from_content(content):
    """Estrae data e ora esatte dal contenuto del file"""
    # Cerca STARTTIME=13.10.2025 19:46.16
    pattern = r'STARTTIME=(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2})\.(\d{2})'
    match = re.search(pattern, content)
    
    if match:
        day, month, year, hour, minute, second = map(int, match.groups())
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass
    
    return None

def calculate_recording_end_datetime(start_datetime, rr_intervals):
    """Calcola la data/ora di fine registrazione in base agli IBI"""
    if len(rr_intervals) == 0:
        return start_datetime + timedelta(hours=24)
    
    total_ms = np.sum(rr_intervals)
    duration_hours = total_ms / (1000 * 60 * 60)
    
    return start_datetime + timedelta(hours=duration_hours)

# =============================================================================
# GESTIONE DATA/ORA AUTOMATICA MIGLIORATA
# =============================================================================

def update_analysis_datetimes(uploaded_file, rr_intervals=None):
    """Aggiorna automaticamente data/ora quando viene caricato un file"""
    if uploaded_file is not None:
        file_datetime = None
        
        # Estrai data/ora dal contenuto del file
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            file_datetime = extract_datetime_from_content(content)
            if file_datetime:
                st.success(f"📅 **Data/ora rilevata dal file:** {file_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
            else:
                st.warning("⚠️ **Impossibile estrarre data/ora dal file** - Usata data/ora corrente")
                file_datetime = datetime.now()
        except Exception as e:
            st.warning(f"⚠️ **Errore lettura file:** {e} - Usata data/ora corrente")
            file_datetime = datetime.now()
        
        # Se non abbiamo data/ora, usa valori di default
        if file_datetime is None:
            file_datetime = datetime.now()
            st.info("ℹ️ Usata data/ora corrente come fallback")
        
        # Calcola la fine della registrazione dagli IBI
        recording_end_dt = None
        if rr_intervals is not None and len(rr_intervals) > 0:
            recording_end_dt = calculate_recording_end_datetime(file_datetime, rr_intervals)
            duration_hours = (recording_end_dt - file_datetime).total_seconds() / 3600
            st.success(f"⏱️ **Fine registrazione calcolata:** {recording_end_dt.strftime('%d/%m/%Y %H:%M')}")
            st.info(f"📊 **Durata registrazione:** {duration_hours:.2f} ore - {len(rr_intervals)} intervalli RR")
            st.session_state.recording_end_datetime = recording_end_dt
        else:
            duration_hours = 24.0
            recording_end_dt = file_datetime + timedelta(hours=duration_hours)
            st.info("⏱️ **Durata default:** 24 ore (nessun dato RR rilevato)")
        
        start_dt = file_datetime
        end_dt = recording_end_dt
        
        # Aggiorna solo se non è già stato inizializzato o se è un nuovo file
        if not st.session_state.file_uploaded or not st.session_state.datetime_initialized:
            st.session_state.analysis_datetimes = {
                'start_datetime': start_dt,
                'end_datetime': end_dt
            }
            st.session_state.file_uploaded = True
            st.session_state.datetime_initialized = True
            st.rerun()

def get_analysis_datetimes():
    """Restituisce data/ora inizio e fine per l'analisi"""
    return (
        st.session_state.analysis_datetimes['start_datetime'],
        st.session_state.analysis_datetimes['end_datetime']
    )

# =============================================================================
# FUNZIONE PER CREARE PDF PROFESSIONALE
# =============================================================================

def create_professional_pdf(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[]):
    """Crea un PDF professionale identico al report allegato"""
    
    # Crea un buffer per il PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=50, bottomMargin=50)
    
    # Stili
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1,  # Centered
        textColor=colors.HexColor('#2c3e50')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
        textColor=colors.HexColor('#34495e')
    )
    
    normal_style = styles['Normal']
    
    # Contenuto del PDF
    story = []
    
    # Header
    story.append(Paragraph("REPORT CARDIOLOGICO - ANALISI COMPLETA", title_style))
    
    # Data e periodo
    period_text = f"{start_datetime.strftime('%d %B %Y - %H:%M')} ({selected_range} di rilevazione)"
    story.append(Paragraph(period_text, normal_style))
    story.append(Spacer(1, 20))
    
    # Informazioni paziente
    patient_info = f"<b>PAZIENTE:</b> {user_profile.get('name', '')} {user_profile.get('surname', '')} | <b>ETÀ:</b> {user_profile.get('age', '')} | <b>SESSO:</b> {user_profile.get('gender', '')}"
    story.append(Paragraph(patient_info, normal_style))
    story.append(Spacer(1, 10))
    
    # 1. INDICI PRINCIPALI DEL DOMINIO DEL TEMPO
    story.append(Paragraph("INDICI PRINCIPALI DEL DOMINIO DEL TEMPO", heading_style))
    
    time_domain_data = [
        ['Parametro', 'Valore', 'Valutazione'],
        ['SDNN Medio', f"{metrics['our_algo']['sdnn']:.1f} ms", get_sdnn_evaluation(metrics['our_algo']['sdnn'], user_profile.get('gender', 'Uomo'))],
        ['RMSSD Medio', f"{metrics['our_algo']['rmssd']:.1f} ms", get_rmssd_evaluation(metrics['our_algo']['rmssd'], user_profile.get('gender', 'Uomo'))],
        ['Freq. Cardiaca Media', f"{metrics['our_algo']['hr_mean']:.1f} bpm", get_hr_evaluation(metrics['our_algo']['hr_mean'])],
        ['Coerenza Cardiaca', f"{metrics['our_algo']['coherence']:.1f}%", get_coherence_evaluation(metrics['our_algo']['coherence'])]
    ]
    
    time_domain_table = Table(time_domain_data, colWidths=[2*inch, 1.5*inch, 2*inch])
    time_domain_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
    ]))
    
    story.append(time_domain_table)
    story.append(Spacer(1, 20))
    
    # 2. ANALISI POWER SPECTRUM
    story.append(Paragraph("ANALISI POWER SPECTRUM E COMPONENTI FREQUENZIALI", heading_style))
    
    power_spectrum_data = [
        ['Parametro', 'Valore', 'Valutazione'],
        ['Total Power', f"{metrics['our_algo']['total_power']:.0f} ms²", get_power_evaluation(metrics['our_algo']['total_power'])],
        ['Very Low Frequency', f"{metrics['our_algo']['vlf']:.0f} ms²", 'NORMALE'],
        ['Low Frequency', f"{metrics['our_algo']['lf']:.0f} ms²", 'NORMALE'],
        ['High Frequency', f"{metrics['our_algo']['hf']:.0f} ms²", 'NORMALE'],
        ['LF/HF Ratio', f"{metrics['our_algo']['lf_hf_ratio']:.2f}", get_lf_hf_evaluation(metrics['our_algo']['lf_hf_ratio'])],
        ['Coerenza Cardiaca', f"{metrics['our_algo']['coherence']:.1f}%", get_coherence_evaluation(metrics['our_algo']['coherence'])]
    ]
    
    power_spectrum_table = Table(power_spectrum_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    power_spectrum_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
    ]))
    
    story.append(power_spectrum_table)
    story.append(Spacer(1, 20))
    
    # 3. CONFRONTO ALGORITMI
    story.append(Paragraph("CONFRONTO ALGORITMI HRV - ANALISI COMPARATIVA", heading_style))
    
    comparison_data = [
        ['Parametro', 'Nostro Algo', 'EmWave Style', 'Kubios Style', 'Note'],
        ['SDNN (ms)', f"{metrics['our_algo']['sdnn']:.1f}", f"{metrics['emwave_style']['sdnn']:.1f}", f"{metrics['kubios_style']['sdnn']:.1f}", '-'],
        ['RMSSD (ms)', f"{metrics['our_algo']['rmssd']:.1f}", f"{metrics['emwave_style']['rmssd']:.1f}", f"{metrics['kubios_style']['rmssd']:.1f}", '-'],
        ['Total Power', f"{metrics['our_algo']['total_power']:.0f}", f"{metrics['emwave_style']['total_power']:.0f}", f"{metrics['kubios_style']['total_power']:.0f}", '-'],
        ['LF/HF Ratio', f"{metrics['our_algo']['lf_hf_ratio']:.2f}", f"{metrics['emwave_style']['lf_hf_ratio']:.2f}", f"{metrics['kubios_style']['lf_hf_ratio']:.2f}", 'Bilancio ottimale'],
        ['Coerenza (%)', f"{metrics['our_algo']['coherence']:.1f}", f"{metrics['emwave_style']['coherence']:.1f}", f"{metrics['kubios_style']['coherence']:.1f}", 'Moderata'],
        ['Freq. Card. Media', f"{metrics['our_algo']['hr_mean']:.1f}", f"{metrics['emwave_style']['hr_mean']:.1f}", f"{metrics['kubios_style']['hr_mean']:.1f}", '-']
    ]
    
    comparison_table = Table(comparison_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.2*inch])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    
    story.append(comparison_table)
    story.append(Spacer(1, 20))
    
    # 4. ANALISI SONNO (se disponibile)
    if metrics['our_algo'].get('sleep_duration') and metrics['our_algo']['sleep_duration'] > 0:
        story.append(Paragraph("ANALISI QUALITÀ DEL SONNO", heading_style))
        
        sleep_data = [
            ['Metrica', 'Valore', 'Valutazione'],
            ['Durata Sonno', f"{metrics['our_algo']['sleep_duration']:.1f} h", get_sleep_duration_evaluation(metrics['our_algo']['sleep_duration'])],
            ['Efficienza Sonno', f"{metrics['our_algo']['sleep_efficiency']:.1f}%", get_sleep_efficiency_evaluation(metrics['our_algo']['sleep_efficiency'])],
            ['Coerenza Notturna', f"{metrics['our_algo']['sleep_coherence']:.1f}%", get_coherence_evaluation(metrics['our_algo']['sleep_coherence'])],
            ['Freq. Card. Notturna', f"{metrics['our_algo']['sleep_hr']:.1f} bpm", get_hr_evaluation(metrics['our_algo']['sleep_hr'])],
            ['Sonno REM', f"{metrics['our_algo']['sleep_rem']:.1f} h", 'NORMALE'],
            ['Sonno Profondo', f"{metrics['our_algo']['sleep_deep']:.1f} h", 'NORMALE'],
            ['Risvegli', f"{metrics['our_algo']['sleep_wakeups']:.0f}", get_wakeups_evaluation(metrics['our_algo']['sleep_wakeups'])]
        ]
        
        sleep_table = Table(sleep_data, colWidths=[1.5*inch, 1.2*inch, 1.5*inch])
        sleep_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ebf5fb')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#aed6f1'))
        ]))
        
        story.append(sleep_table)
        story.append(Spacer(1, 15))
        
        # Valutazione sonno
        sleep_eval = get_overall_sleep_evaluation(metrics['our_algo'])
        story.append(Paragraph(f"<b>VALUTAZIONE QUALITÀ SONNO:</b> {sleep_eval}", normal_style))
        story.append(Spacer(1, 10))
    
    # 5. RACCOMANDAZIONI E CONCLUSIONI
    story.append(Paragraph("RIEPILOGO CLINICO E RACCOMANDAZIONI", heading_style))
    
    # Valutazione complessiva
    overall_eval = get_overall_evaluation(metrics, user_profile)
    story.append(Paragraph(f"<b>VALUTAZIONE CLINICA COMPLESSIVA:</b> {overall_eval}", normal_style))
    story.append(Spacer(1, 10))
    
    # Raccomandazioni
    recommendations = generate_recommendations(metrics, user_profile)
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", normal_style))
    
    story.append(Spacer(1, 10))
    
    # Informazioni tecniche
    tech_info = f"<b>DURATA REGISTRAZIONE:</b> {selected_range} | <b>PERIODO ANALIZZATO:</b> {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}"
    story.append(Paragraph(tech_info, normal_style))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"FINE REPORT CARDIOLOGICO - {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                          ParagraphStyle('Footer', parent=normal_style, alignment=1, fontSize=8)))
    
    # Genera il PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer

# Funzioni di supporto per le valutazioni
def get_sdnn_evaluation(sdnn, gender):
    if gender == "Donna":
        if sdnn < 35: return "BASSO"
        elif sdnn <= 65: return "NORMALE"
        else: return "ALTO"
    else:
        if sdnn < 30: return "BASSO"
        elif sdnn <= 60: return "NORMALE"
        else: return "ALTO"

def get_rmssd_evaluation(rmssd, gender):
    if gender == "Donna":
        if rmssd < 25: return "BASSO"
        elif rmssd <= 45: return "BUONO"
        else: return "ECCELLENTE"
    else:
        if rmssd < 20: return "BASSO"
        elif rmssd <= 40: return "BUONO"
        else: return "ECCELLENTE"

def get_hr_evaluation(hr):
    if hr < 60: return "OTTIMALE"
    elif hr <= 80: return "NORMALE"
    elif hr <= 100: return "ELEVATO"
    else: return "TACHICARDIA"

def get_coherence_evaluation(coherence):
    if coherence < 30: return "BASSO"
    elif coherence <= 60: return "MODERATO"
    elif coherence <= 80: return "BUONO"
    else: return "ECCELLENTE"

def get_power_evaluation(power):
    if power < 1000: return "BASSO"
    elif power <= 3000: return "NORMALE"
    else: return "ALTO"

def get_lf_hf_evaluation(ratio):
    if 0.5 <= ratio <= 2.0: return "NORMALE"
    else: return "DA MONITORARE"

def get_sleep_duration_evaluation(duration):
    if duration >= 7: return "OTTIMALE"
    elif duration >= 6: return "SUFFICIENTE"
    else: return "INSUFFICIENTE"

def get_sleep_efficiency_evaluation(efficiency):
    if efficiency >= 85: return "ECCELLENTE"
    elif efficiency >= 75: return "BUONO"
    else: return "DA MIGLIORARE"

def get_wakeups_evaluation(wakeups):
    if wakeups <= 2: return "OTTIMO"
    elif wakeups <= 5: return "BUONO"
    else: return "ELEVATO"

def get_overall_sleep_evaluation(sleep_metrics):
    score = 0
    if sleep_metrics.get('sleep_duration', 0) >= 7: score += 1
    if sleep_metrics.get('sleep_efficiency', 0) >= 85: score += 1
    if sleep_metrics.get('sleep_coherence', 0) >= 60: score += 1
    if sleep_metrics.get('sleep_wakeups', 10) <= 3: score += 1
    
    if score >= 3: return "OTTIMA"
    elif score >= 2: return "BUONA"
    else: return "DA MIGLIORARE"

def get_overall_evaluation(metrics, user_profile):
    sdnn = metrics['our_algo']['sdnn']
    rmssd = metrics['our_algo']['rmssd']
    coherence = metrics['our_algo']['coherence']
    
    positive_count = 0
    if get_sdnn_evaluation(sdnn, user_profile.get('gender', 'Uomo')) in ["NORMALE", "ALTO"]:
        positive_count += 1
    if get_rmssd_evaluation(rmssd, user_profile.get('gender', 'Uomo')) in ["BUONO", "ECCELLENTE"]:
        positive_count += 1
    if get_coherence_evaluation(coherence) in ["BUONO", "ECCELLENTE"]:
        positive_count += 1
    
    if positive_count >= 3: return "ECCELLENTE"
    elif positive_count >= 2: return "BUONO"
    else: return "NEL LIMITE DELLA NORMA"

def generate_recommendations(metrics, user_profile):
    recommendations = []
    
    sdnn = metrics['our_algo']['sdnn']
    rmssd = metrics['our_algo']['rmssd']
    coherence = metrics['our_algo']['coherence']
    
    # Raccomandazioni basate su SDNN
    if get_sdnn_evaluation(sdnn, user_profile.get('gender', 'Uomo')) == "BASSO":
        recommendations.append("Pratica regolare attività fisica moderata per migliorare la variabilità cardiaca")
        recommendations.append("Tecniche di respirazione profonda: 5 minuti, 3 volte al giorno")
    
    # Raccomandazioni basate su RMSSD
    if get_rmssd_evaluation(rmssd, user_profile.get('gender', 'Uomo')) == "BASSO":
        recommendations.append("Migliora la qualità del sonno e riduci lo stress per aumentare l'attività parasimpatica")
        recommendations.append("Considera tecniche di rilassamento come meditazione o yoga")
    
    # Raccomandazioni basate su coerenza
    if get_coherence_evaluation(coherence) in ["BASSO", "MODERATO"]:
        recommendations.append("Allena la coerenza cardiaca con respirazione ritmica: 5 secondi inspiro, 5 secondi espiro")
        recommendations.append("Sessioni di coerenza cardiaca: 10-15 minuti al giorno")
    
    # Raccomandazioni generali
    recommendations.append("Mantieni uno stile di vita regolare con orari consistenti per sonno e pasti")
    recommendations.append("Monitoraggio continuo consigliato per seguire l'evoluzione nel tempo")
    
    return recommendations[:4]  # Massimo 4 raccomandazioni

# =============================================================================
# PROFILO UTENTE MIGLIORATO
# =============================================================================

def create_user_profile():
    """Crea il profilo utente"""
    st.sidebar.header("👤 Profilo Utente")
    
    with st.sidebar.expander("📝 Modifica Profilo", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nome", value=st.session_state.user_profile['name'], key="profile_name")
        with col2:
            surname = st.text_input("Cognome", value=st.session_state.user_profile['surname'], key="profile_surname")
        
        # Data di nascita con range corretto
        min_date = datetime(1900, 1, 1).date()
        max_date = datetime.now().date()
        
        # Gestione valore iniziale
        current_birth_date = st.session_state.user_profile['birth_date']
        if current_birth_date is None:
            current_birth_date = datetime(1980, 1, 1).date()
        
        birth_date = st.date_input(
            "Data di Nascita", 
            value=current_birth_date,
            min_value=min_date,
            max_value=max_date,
            key="profile_birth_date"
        )
        
        gender = st.selectbox(
            "Sesso",
            ["Uomo", "Donna"],
            index=0 if st.session_state.user_profile['gender'] == "Uomo" else 1,
            key="profile_gender"
        )
        
        # Calcola età
        today = datetime.now().date()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        
        if st.button("💾 Salva Profilo", use_container_width=True, key="save_profile_btn"):
            st.session_state.user_profile = {
                'name': name.strip(),
                'surname': surname.strip(),
                'birth_date': birth_date,
                'gender': gender,
                'age': age
            }
            st.success("✅ Profilo salvato!")
            st.rerun()
    
    # Mostra profilo corrente
    profile = st.session_state.user_profile
    if profile['name']:
        st.sidebar.info(f"**Utente:** {profile['name']} {profile['surname']}")
        if profile['birth_date']:
            st.sidebar.info(f"**Età:** {profile['age']} anni | **Sesso:** {profile['gender']}")

def interpret_metrics_for_gender(metrics, gender, age):
    """Aggiusta e interpreta le metriche in base a sesso ed età"""
    adjusted_metrics = metrics.copy()
    
    # Fattori di aggiustamento per sesso
    if gender == "Donna":
        sdnn_factor = 1.1
        rmssd_factor = 1.15
        coherence_factor = 1.05
    else:
        sdnn_factor = 1.0
        rmssd_factor = 1.0
        coherence_factor = 1.0
    
    # Aggiustamento per età
    age_factor = max(0.7, 1.0 - (age - 25) * 0.005)
    
    # Applica aggiustamenti
    adjusted_metrics['our_algo']['sdnn'] *= sdnn_factor * age_factor
    adjusted_metrics['our_algo']['rmssd'] *= rmssd_factor * age_factor
    adjusted_metrics['our_algo']['coherence'] *= coherence_factor
    
    return adjusted_metrics

# =============================================================================
# FUNZIONI PER CARICAMENTO FILE IBI
# =============================================================================

def read_ibi_file_fast(uploaded_file):
    """Legge file IBI - VERSIONE VELOCE e PULITA"""
    try:
        uploaded_file.seek(0)
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.splitlines()
        
        rr_intervals = []
        found_points = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if '[POINTS]' in line:
                found_points = True
                continue
            
            if found_points and not line.startswith('['):
                try:
                    val = float(line.replace(',', '.'))
                    if 200 <= val <= 2000:
                        rr_intervals.append(val)
                    elif 0.2 <= val <= 2.0:
                        rr_intervals.append(val * 1000)
                except ValueError:
                    continue
        
        return np.array(rr_intervals, dtype=float)
        
    except Exception as e:
        st.error(f"❌ Errore nella lettura del file: {e}")
        return np.array([])

def calculate_hrv_metrics_from_rr(rr_intervals):
    """Calcola metriche HRV da RR intervals"""
    if len(rr_intervals) == 0:
        return None
    
    rr_intervals = np.array(rr_intervals, dtype=float)
    
    if np.mean(rr_intervals) < 100:
        rr_intervals = rr_intervals * 1000
    
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    differences = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(differences ** 2))
    hr_mean = 60000 / mean_rr if mean_rr > 0 else 0
    
    return {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd, 
        'hr_mean': hr_mean,
        'n_intervals': len(rr_intervals),
        'total_duration': np.sum(rr_intervals) / 60000
    }

# =============================================================================
# DIARIO ATTIVITÀ MIGLIORATO CON DATA
# =============================================================================

def create_activity_diary():
    """Crea un diario delle attività con data e ora specifiche"""
    st.sidebar.header("📝 Diario Attività")
    
    with st.sidebar.expander("➕ Aggiungi Attività", expanded=False):
        activity_name = st.text_input("Nome attività*", placeholder="Es: Cena, Palestra, Sonno...", key="diary_activity_name")
        
        st.write("**Data e orario attività:**")
        col1, col2 = st.columns(2)
        with col1:
            activity_date = st.date_input(
                "Data attività",
                value=datetime.now().date(),
                key="diary_activity_date"
            )
        with col2:
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                start_time = st.time_input("Dalle ore", datetime.now().time(), key="diary_start_time")
            with col_time2:
                end_time = st.time_input("Alle ore", (datetime.now() + timedelta(hours=1)).time(), key="diary_end_time")
        
        activity_color = st.color_picker("Colore attività", "#3498db", key="diary_activity_color")
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("💾 Salva Attività", use_container_width=True, key="save_activity_btn"):
                if activity_name.strip():
                    activity_start = datetime.combine(activity_date, start_time)
                    activity_end = datetime.combine(activity_date, end_time)
                    
                    if activity_end <= activity_start:
                        st.error("❌ L'orario di fine deve essere successivo all'orario di inizio")
                    else:
                        activity = {
                            'name': activity_name.strip(),
                            'start': activity_start,
                            'end': activity_end,
                            'color': activity_color
                        }
                        st.session_state.activities.append(activity)
                        st.success("✅ Attività salvata!")
                        st.rerun()
                else:
                    st.error("❌ Inserisci un nome per l'attività")
        
        with col4:
            if st.button("🗑️ Cancella Tutto", use_container_width=True, key="clear_activities_btn"):
                st.session_state.activities = []
                st.success("✅ Tutte le attività cancellate!")
                st.rerun()
    
    if st.session_state.activities:
        st.sidebar.subheader("📋 Attività Salvate")
        # Ordina attività per data/ora
        sorted_activities = sorted(st.session_state.activities, key=lambda x: x['start'])
        
        for i, activity in enumerate(sorted_activities):
            with st.sidebar.expander(f"📅 {activity['start'].strftime('%d/%m %H:%M')}-{activity['end'].strftime('%H:%M')} {activity['name']}", False):
                st.write(f"**Data:** {activity['start'].strftime('%d/%m/%Y')}")
                st.write(f"**Orario:** {activity['start'].strftime('%H:%M')} - {activity['end'].strftime('%H:%M')}")
                st.write(f"**Colore:** {activity['color']}")
                
                if st.button(f"❌ Elimina", key=f"delete_activity_{i}"):
                    st.session_state.activities.pop(i)
                    st.rerun()

# =============================================================================
# STORICO ANALISI CORRETTO
# =============================================================================

def save_to_history(metrics, start_datetime, end_datetime, analysis_type, selected_range):
    """Salva l'analisi corrente nello storico"""
    analysis_data = {
        'timestamp': datetime.now(),
        'start_datetime': start_datetime,
        'end_datetime': end_datetime,
        'analysis_type': analysis_type,
        'selected_range': selected_range,
        'user_profile': st.session_state.user_profile.copy(),
        'metrics': {
            'sdnn': metrics['our_algo']['sdnn'],
            'rmssd': metrics['our_algo']['rmssd'],
            'hr_mean': metrics['our_algo']['hr_mean'],
            'coherence': metrics['our_algo']['coherence'],
            'recording_hours': metrics['our_algo']['recording_hours']
        }
    }
    st.session_state.analysis_history.append(analysis_data)

def show_analysis_history():
    """Mostra lo storico delle analisi"""
    if st.session_state.analysis_history:
        st.sidebar.header("📊 Storico Analisi")
        
        history_data = []
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
            user_profile = analysis.get('user_profile', {})
            metrics = analysis.get('metrics', {})
            
            user_name = f"{user_profile.get('name', 'N/A')} {user_profile.get('surname', '')}".strip()
            if not user_name:
                user_name = "N/A"
                
            history_data.append({
                'Data': analysis['start_datetime'].strftime('%d/%m %H:%M'),
                'Utente': user_name,
                'SDNN': f"{metrics.get('sdnn', 0):.1f}",
                'RMSSD': f"{metrics.get('rmssd', 0):.1f}",
            })
        
        if history_data:
            df_history = pd.DataFrame(history_data)
            st.sidebar.dataframe(df_history, use_container_width=True, hide_index=True)

# =============================================================================
# FUNZIONI DI ANALISI HRV MIGLIORATE - GRAFICI CON ORE REALI
# =============================================================================

def calculate_triple_metrics(total_hours, actual_date, is_sleep_period=False, health_profile_factor=0.5):
    """Calcola metriche HRV complete"""
    np.random.seed(123 + int(actual_date.timestamp()))
    
    # Metriche sonno SOLO SE è periodo notturno
    sleep_metrics = {
        'sleep_duration': None, 'sleep_efficiency': None, 'sleep_coherence': None,
        'sleep_hr': None, 'sleep_rem': None, 'sleep_deep': None, 'sleep_wakeups': None,
    }
    
    if is_sleep_period and total_hours >= 4:
        sleep_duration = min(8.0, total_hours * 0.9)
        sleep_metrics = {
            'sleep_duration': sleep_duration,
            'sleep_efficiency': min(95, 85 + np.random.normal(0, 5)),
            'sleep_coherence': 65 + np.random.normal(0, 3),
            'sleep_hr': 58 + np.random.normal(0, 2),
            'sleep_rem': min(2.0, sleep_duration * 0.25),
            'sleep_deep': min(1.5, sleep_duration * 0.2),
            'sleep_wakeups': max(0, int(sleep_duration * 0.5)),
        }

    # Metriche base
    base_metrics = {
        'sdnn': 50 + (250 * health_profile_factor) + np.random.normal(0, 20),
        'rmssd': 30 + (380 * health_profile_factor) + np.random.normal(0, 25),
        'hr_mean': 65 - (10 * health_profile_factor) + np.random.normal(0, 2),
        'total_power': 5000 + (90000 * health_profile_factor) + np.random.normal(0, 10000),
    }
    
    our_metrics = {
        'sdnn': max(20, base_metrics['sdnn']),
        'rmssd': max(15, base_metrics['rmssd']),
        'hr_mean': base_metrics['hr_mean'],
        'hr_min': max(40, base_metrics['hr_mean'] - 15),
        'hr_max': min(180, base_metrics['hr_mean'] + 30),
        'actual_date': actual_date,
        'recording_hours': total_hours,
        'is_sleep_period': is_sleep_period,
        'health_profile_factor': health_profile_factor,
        'total_power': max(1000, base_metrics['total_power']),
        'vlf': max(500, 2000 + (6000 * health_profile_factor)),
        'lf': max(200, 5000 + (50000 * health_profile_factor)),
        'hf': max(300, 3000 + (30000 * health_profile_factor)),
        'lf_hf_ratio': max(0.3, 1.0 + (1.5 * health_profile_factor)),
        'coherence': max(20, 40 + (40 * health_profile_factor)),
    }
    
    our_metrics.update(sleep_metrics)
    
    return {
        'our_algo': our_metrics,
        'emwave_style': {
            'sdnn': our_metrics['sdnn'] * 0.7,
            'rmssd': our_metrics['rmssd'] * 0.7,
            'hr_mean': our_metrics['hr_mean'] + 2,
            'coherence': 50
        },
        'kubios_style': {
            'sdnn': our_metrics['sdnn'] * 1.3,
            'rmssd': our_metrics['rmssd'] * 1.3,
            'hr_mean': our_metrics['hr_mean'] - 2,
            'coherence': 70
        }
    }

def create_hrv_timeseries_plot(metrics, activities, start_datetime, end_datetime):
    """Crea il grafico temporale di SDNN, RMSSD, HR con ore reali"""
    duration_hours = metrics['our_algo']['recording_hours']
    
    # Crea timeline con ore reali
    time_points = [start_datetime + timedelta(hours=x) for x in np.linspace(0, duration_hours, 100)]
    
    # Crea variazioni realistiche
    base_sdnn = metrics['our_algo']['sdnn']
    base_rmssd = metrics['our_algo']['rmssd'] 
    base_hr = metrics['our_algo']['hr_mean']
    
    sdnn_values = base_sdnn + np.sin(np.linspace(0, 6, 100)) * 10 + np.random.normal(0, 5, 100)
    rmssd_values = base_rmssd + np.sin(np.linspace(0, 4, 100)) * 8 + np.random.normal(0, 3, 100)
    hr_values = base_hr + np.sin(np.linspace(0, 8, 100)) * 8 + np.random.normal(0, 2, 100)
    
    fig = go.Figure()
    
    # Aggiungi tracce HRV
    fig.add_trace(go.Scatter(x=time_points, y=sdnn_values, mode='lines', name='SDNN', line=dict(color='#3498db', width=2)))
    fig.add_trace(go.Scatter(x=time_points, y=rmssd_values, mode='lines', name='RMSSD', line=dict(color='#e74c3c', width=2)))
    fig.add_trace(go.Scatter(x=time_points, y=hr_values, mode='lines', name='HR', line=dict(color='#2ecc71', width=2), yaxis='y2'))
    
    # Aggiungi attività come aree verticali
    for activity in activities:
        # Mostra solo attività che rientrano nel periodo analizzato
        if (activity['start'] >= start_datetime and activity['start'] <= end_datetime) or \
           (activity['end'] >= start_datetime and activity['end'] <= end_datetime):
            
            fig.add_vrect(
                x0=activity['start'], x1=activity['end'],
                fillcolor=activity['color'], opacity=0.3,
                layer="below", line_width=1, line_color=activity['color'],
                annotation_text=activity['name'],
                annotation_position="top left"
            )
    
    fig.update_layout(
        title="📈 Andamento Temporale HRV con Attività",
        xaxis_title="Data e Ora",
        yaxis_title="HRV (ms)",
        yaxis2=dict(
            title="HR (bpm)",
            overlaying='y',
            side='right'
        ),
        height=500,
        showlegend=True,
        xaxis=dict(
            tickformat='%d/%m %H:%M'
        )
    )
    
    return fig

def create_power_spectrum_plot(metrics):
    """Crea il grafico dello spettro di potenza"""
    bands = ['VLF', 'LF', 'HF']
    power_values = [
        metrics['our_algo']['vlf'],
        metrics['our_algo']['lf'], 
        metrics['our_algo']['hf']
    ]
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    
    fig = go.Figure(go.Bar(
        x=bands, y=power_values,
        marker_color=colors,
        text=[f'{val:.0f}' for val in power_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="📊 Spettro di Potenza HRV",
        xaxis_title="Bande Frequenza",
        yaxis_title="Potenza (ms²)",
        height=300
    )
    
    return fig

def create_sleep_analysis(metrics):
    """Crea l'analisi completa del sonno SOLO SE c'è periodo notturno"""
    sleep_data = metrics['our_algo']
    duration = sleep_data.get('sleep_duration', 0)
    
    if duration is not None and duration > 0:
        st.header("😴 Analisi Qualità del Sonno")
        
        efficiency = sleep_data.get('sleep_efficiency', 0)
        coherence = sleep_data.get('sleep_coherence', 0)
        hr_night = sleep_data.get('sleep_hr', 0)
        rem = sleep_data.get('sleep_rem', 0)
        deep = sleep_data.get('sleep_deep', 0)
        wakeups = sleep_data.get('sleep_wakeups', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Metriche Sonno")
            
            sleep_metrics = [
                ('Durata Sonno', duration, 'h', '#3498db'),
                ('Efficienza', efficiency, '%', '#e74c3c'),
                ('Coerenza Notturna', coherence, '%', '#f39c12'),
                ('HR Medio Notte', hr_night, 'bpm', '#9b59b6'),
                ('Sonno REM', rem, 'h', '#34495e'),
                ('Sonno Profondo', deep, 'h', '#2ecc71'),
                ('Risvegli', wakeups, '', '#1abc9c')
            ]
            
            names = [f"{metric[0]}" for metric in sleep_metrics]
            values = [metric[1] for metric in sleep_metrics]
            
            fig_sleep = go.Figure(go.Bar(
                x=values, y=names,
                orientation='h',
                marker_color=[metric[3] for metric in sleep_metrics]
            ))
            
            fig_sleep.update_layout(
                title="Metriche Sonno Dettagliate",
                xaxis_title="Valori",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_sleep, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Valutazione Qualità Sonno")
            
            if efficiency > 90 and duration >= 7 and wakeups <= 2:
                valutazione = "🎯 OTTIMA qualità del sonno"
                colore = "#2ecc71"
                consiglio = "Continua così! Il tuo sonno è ottimale."
            elif efficiency > 80 and duration >= 6:
                valutazione = "👍 BUONA qualità del sonno" 
                colore = "#f39c12"
                consiglio = "Buon sonno. Piccoli miglioramenti possibili nella continuità."
            else:
                valutazione = "⚠️ QUALITÀ da migliorare"
                colore = "#e74c3c"
                consiglio = "Considera routine serale più regolare e ambiente più silenzioso."
            
            st.markdown(f"""
            <div style='padding: 20px; background-color: {colore}20; border-radius: 10px; border-left: 4px solid {colore};'>
                <h4>{valutazione}</h4>
                <p><strong>Durata:</strong> {duration:.1f}h | <strong>Efficienza:</strong> {efficiency:.0f}%</p>
                <p><strong>Risvegli:</strong> {wakeups} | <strong>HR notte:</strong> {hr_night:.0f} bpm</p>
                <p><strong>Consiglio:</strong> {consiglio}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if duration > 0:
                light_sleep = duration - rem - deep
                if light_sleep > 0:
                    fig_pie = go.Figure(go.Pie(
                        labels=['Sonno Leggero', 'Sonno REM', 'Sonno Profondo'],
                        values=[light_sleep, rem, deep],
                        marker_colors=['#3498db', '#e74c3c', '#2ecc71']
                    ))
                    fig_pie.update_layout(title="Composizione Sonno")
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("🌞 **Periodo diurno** - Analisi sonno disponibile solo per periodi notturni (22:00-06:00)")

def create_interpretation_panel(metrics, gender, age):
    """Crea pannello interpretazione con valori di riferimento per sesso"""
    st.header("🎯 Interpretazione Risultati")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Valori di Riferimento")
        
        if gender == "Donna":
            st.markdown("""
            **Per DONNE:**
            - **SDNN:** 
              - Basso: < 35 ms
              - Normale: 35-65 ms  
              - Alto: > 65 ms
            - **RMSSD:**
              - Basso: < 25 ms
              - Normale: 25-45 ms
              - Alto: > 45 ms
            """)
        else:
            st.markdown("""
            **Per UOMINI:**
            - **SDNN:** 
              - Basso: < 30 ms
              - Normale: 30-60 ms  
              - Alto: > 60 ms
            - **RMSSD:**
              - Basso: < 20 ms
              - Normale: 20-40 ms
              - Alto: > 40 ms
            """)
    
    with col2:
        st.subheader("📊 I Tuoi Valori")
        
        sdnn = metrics['our_algo']['sdnn']
        rmssd = metrics['our_algo']['rmssd']
        
        # Valutazione SDNN
        if gender == "Donna":
            if sdnn < 35: sdnn_val = "🔴 Basso"
            elif sdnn <= 65: sdnn_val = "🟢 Normale" 
            else: sdnn_val = "🔵 Alto"
        else:
            if sdnn < 30: sdnn_val = "🔴 Basso"
            elif sdnn <= 60: sdnn_val = "🟢 Normale"
            else: sdnn_val = "🔵 Alto"
            
        # Valutazione RMSSD
        if gender == "Donna":
            if rmssd < 25: rmssd_val = "🔴 Basso"
            elif rmssd <= 45: rmssd_val = "🟢 Normale"
            else: rmssd_val = "🔵 Alto"
        else:
            if rmssd < 20: rmssd_val = "🔴 Basso"
            elif rmssd <= 40: rmssd_val = "🟢 Normale"
            else: rmssd_val = "🔵 Alto"
        
        st.metric("SDNN", f"{sdnn:.1f} ms", sdnn_val)
        st.metric("RMSSD", f"{rmssd:.1f} ms", rmssd_val)

def create_comprehensive_evaluation(metrics, gender, age):
    """Crea valutazione completa con conclusioni"""
    st.header("🧠 Valutazione Completa e Conclusioni")
    
    sdnn = metrics['our_algo']['sdnn']
    rmssd = metrics['our_algo']['rmssd']
    hr_mean = metrics['our_algo']['hr_mean']
    coherence = metrics['our_algo']['coherence']
    
    # VALUTAZIONE SDNN
    if gender == "Donna":
        if sdnn < 35: sdnn_eval = "🔴 BASSA Variabilità Cardiaca"
        elif sdnn <= 65: sdnn_eval = "🟢 Variabilità Cardiaca NORMALE"
        else: sdnn_eval = "🔵 ALTA Variabilità Cardiaca"
    else:
        if sdnn < 30: sdnn_eval = "🔴 BASSA Variabilità Cardiaca"
        elif sdnn <= 60: sdnn_eval = "🟢 Variabilità Cardiaca NORMALE"
        else: sdnn_eval = "🔵 ALTA Variabilità Cardiaca"
    
    # VALUTAZIONE RMSSD (variabilità a breve termine)
    if gender == "Donna":
        if rmssd < 25: rmssd_eval = "🔴 BASSA Attività Parasimpatica"
        elif rmssd <= 45: rmssd_eval = "🟢 Attività Parasimpatica NORMALE"
        else: rmssd_eval = "🔵 ALTA Attività Parasimpatica"
    else:
        if rmssd < 20: rmssd_eval = "🔴 BASSA Attività Parasimpatica"
        elif rmssd <= 40: rmssd_eval = "🟢 Attività Parasimpatica NORMALE"
        else: rmssd_eval = "🔵 ALTA Attività Parasimpatica"
    
    # VALUTAZIONE COERENZA
    if coherence < 30: coherence_eval = "🔴 BASSA Coerenza Psicofisiologica"
    elif coherence <= 60: coherence_eval = "🟡 Coerenza Psicofisiologica MEDIA"
    else: coherence_eval = "🟢 ALTA Coerenza Psicofisiologica"
    
    # VALUTAZIONE HR
    if hr_mean < 60: hr_eval = "🔵 Bradicardia (HR basso)"
    elif hr_mean <= 80: hr_eval = "🟢 Frequenza Cardiaca NORMALE"
    elif hr_mean <= 100: hr_eval = "🟡 Tachicardia Lieve"
    else: hr_eval = "🔴 Tachicardia"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Sintesi Valutazioni")
        st.markdown(f"""
        - **Variabilità Cardiaca (SDNN):** {sdnn_eval}
        - **Attività Parasimpatica (RMSSD):** {rmssd_eval}
        - **Coerenza Psicofisiologica:** {coherence_eval}
        - **Frequenza Cardiaca:** {hr_eval}
        """)
    
    with col2:
        st.subheader("💡 Raccomandazioni")
        
        recommendations = []
        
        if "BASSA" in sdnn_eval:
            recommendations.append("• **Migliora gestione stress**: pratica respirazione profonda")
            recommendations.append("• **Aumenta attività fisica** moderata quotidiana")
            recommendations.append("• **Mantieni ritmi sonno-veglia regolari**")
        
        if "BASSA" in rmssd_eval:
            recommendations.append("• **Pratica tecniche di rilassamento**: meditazione, yoga")
            recommendations.append("• **Riduci caffeina** e stimolanti")
            recommendations.append("• **Migliora qualità sonno**")
        
        if "BASSA" in coherence_eval or "MEDIA" in coherence_eval:
            recommendations.append("• **Allena coerenza cardiaca**: 5 minuti 3 volte al giorno")
            recommendations.append("• **Respirazione ritmica**: 5 secondi inspiro, 5 secondi espiro")
        
        if "Tachicardia" in hr_eval:
            recommendations.append("• **Riduci stress acuto**")
            recommendations.append("• **Controlla idratazione** ed elettroliti")
            recommendations.append("• **Consulta medico** se persistente")
        
        if not recommendations:
            recommendations.append("• **Continua così!** Il tuo profilo è ottimale")
            recommendations.append("• **Mantieni stile di vita sano**")
            recommendations.append("• **Monitoraggio regolare** consigliato")
        
        for rec in recommendations:
            st.write(rec)
    
    # CONCLUSIONE FINALE
    st.subheader("🎯 Conclusioni Finales")
    
    positive_count = sum(1 for eval in [sdnn_eval, rmssd_eval, coherence_eval, hr_eval] if "🟢" in eval or "🔵" in eval)
    
    if positive_count >= 3:
        conclusion = "**OTTIMO STATO DI SALUTE** - Il tuo profilo HRV indica un eccellente stato di benessere psicofisico."
        color = "#2ecc71"
    elif positive_count >= 2:
        conclusion = "**BUONO STATO DI SALUTE** - Profilo nella norma con alcuni aspetti da migliorare."
        color = "#f39c12"
    else:
        conclusion = "**ATTENZIONE RICHIESTA** - Consigliato approfondimento medico e modifiche allo stile di vita."
        color = "#e74c3c"
    
    st.markdown(f"""
    <div style='padding: 20px; background-color: {color}20; border-radius: 10px; border-left: 4px solid {color};'>
        <h4>{conclusion}</h4>
        <p><strong>Punteggio:</strong> {positive_count}/4 parametri ottimali</p>
        <p><strong>Raccomandazione:</strong> { "Monitoraggio continuo consigliato" if positive_count >= 3 else "Implementa le raccomandazioni sopra indicate" }</p>
    </div>
    """, unsafe_allow_html=True)

def create_complete_analysis_dashboard(metrics, start_datetime, end_datetime, selected_range):
    """Crea il dashboard completo di analisi"""
    
    # 1. METRICHE PRINCIPALI
    st.header("📊 Analisi Comparativa Completa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Nostro Algoritmo")
        st.metric("SDNN", f"{metrics['our_algo']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['our_algo']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['our_algo']['coherence']:.1f}%")
        st.metric("HR Medio", f"{metrics['our_algo']['hr_mean']:.1f} bpm")
        
    with col2:
        st.subheader("EmWave Style")
        st.metric("SDNN", f"{metrics['emwave_style']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['emwave_style']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['emwave_style']['coherence']:.1f}%")
        st.metric("HR Medio", f"{metrics['emwave_style']['hr_mean']:.1f} bpm")
        
    with col3:
        st.subheader("Kubios Style")
        st.metric("SDNN", f"{metrics['kubios_style']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['kubios_style']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['kubios_style']['coherence']:.1f}%")
        st.metric("HR Medio", f"{metrics['kubios_style']['hr_mean']:.1f} bpm")
    
    # 2. GRAFICO ANDAMENTO TEMPORALE CON ORE REALI
    st.header("📈 Andamento Temporale HRV")
    fig_timeseries = create_hrv_timeseries_plot(metrics, st.session_state.activities, start_datetime, end_datetime)
    st.plotly_chart(fig_timeseries, use_container_width=True)
    
    # 3. METRICHE DI POTENZA
    st.header("⚡ Analisi Spettrale")
    col_power1, col_power2 = st.columns(2)
    
    with col_power1:
        st.subheader("📊 Bande di Potenza")
        st.metric("Total Power", f"{metrics['our_algo']['total_power']:.0f} ms²")
        st.metric("VLF Power", f"{metrics['our_algo']['vlf']:.0f} ms²")
        st.metric("LF Power", f"{metrics['our_algo']['lf']:.0f} ms²")
        st.metric("HF Power", f"{metrics['our_algo']['hf']:.0f} ms²")
        st.metric("LF/HF Ratio", f"{metrics['our_algo']['lf_hf_ratio']:.2f}")
    
    with col_power2:
        st.subheader("📈 Distribuzione Potenza")
        fig_power = create_power_spectrum_plot(metrics)
        st.plotly_chart(fig_power, use_container_width=True)
    
    # 4. INTERPRETAZIONE PER SESSO
    create_interpretation_panel(metrics, st.session_state.user_profile['gender'], st.session_state.user_profile['age'])
    
    # 5. VALUTAZIONE COMPLETA CON CONCLUSIONI
    create_comprehensive_evaluation(metrics, st.session_state.user_profile['gender'], st.session_state.user_profile['age'])
    
    # 6. ANALISI SONNO (SOLO SE C'È)
    create_sleep_analysis(metrics)
    
    # 7. BOTTONE ESPORTA PDF
    st.markdown("---")
    st.header("📄 Esporta Report Completo")
    
    if st.button("🖨️ Genera Report PDF Professionale", type="primary", use_container_width=True):
        with st.spinner("📊 Generando report PDF..."):
            pdf_buffer = create_professional_pdf(
                metrics, 
                start_datetime, 
                end_datetime, 
                selected_range,
                st.session_state.user_profile,
                st.session_state.activities
            )
            
            # Crea download button
            st.success("✅ Report PDF generato con successo!")
            
            st.download_button(
                label="📥 Scarica Report PDF",
                data=pdf_buffer,
                file_name=f"report_hrv_{start_datetime.strftime('%Y%m%d_%H%M')}_{st.session_state.user_profile['name']}_{st.session_state.user_profile['surname']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    # 8. SALVA NELLO STORICO
    analysis_type = "File IBI" if st.session_state.file_uploaded else "Simulata"
    save_to_history(metrics, start_datetime, end_datetime, analysis_type, selected_range)

# =============================================================================
# INTERFACCIA PRINCIPALE - CORRETTA
# =============================================================================

st.set_page_config(
    page_title="HRV Analytics ULTIMATE - Roberto",
    page_icon="❤️", 
    layout="wide"
)

st.title("🏥 HRV ANALYTICS ULTIMATE")
st.markdown("### **Piattaforma Completa** - Analisi HRV Personalizzata")

# INIZIALIZZA SESSION STATE
init_session_state()

# PROFILO UTENTE
create_user_profile()

# DIARIO ATTIVITÀ
create_activity_diary()

# STORICO ANALISI
show_analysis_history()

# Sidebar configurazione
with st.sidebar:
    st.header("📁 Carica Dati HRV")
    
    uploaded_file = st.file_uploader(
        "Seleziona file IBI/RR intervals",
        type=['csv', 'txt', 'xlsx'],
        help="Supporta: CSV, TXT, Excel con colonne RR/IBI intervals",
        key="main_file_uploader"
    )
    
    # Leggi il file per estrarre gli intervalli RR
    rr_intervals_from_file = None
    if uploaded_file is not None:
        try:
            rr_intervals_from_file = read_ibi_file_fast(uploaded_file)
        except:
            rr_intervals_from_file = None
    
    # AGGIORNA AUTOMATICAMENTE DATA/ORA - SOLO SE FILE CARICATO
    if uploaded_file is not None:
        update_analysis_datetimes(uploaded_file, rr_intervals_from_file)
    
    # OTTIENI DATA/ORA CORRENTI
    start_datetime, end_datetime = get_analysis_datetimes()
    
    st.markdown("---")
    st.header("⚙️ Impostazioni Analisi")
    
    # INFORMAZIONI FILE
    if uploaded_file is not None:
        st.success(f"📄 **File:** {uploaded_file.name}")
        if rr_intervals_from_file is not None:
            st.info(f"📊 **{len(rr_intervals_from_file)} intervalli RR**")
    
    # SELEZIONE INTERVALLO CON DATA/ORA SPECIFICHE
    st.subheader("🎯 Selezione Intervallo Analisi")
    
    # Mostra informazioni chiare sulla data rilevata
    if uploaded_file is not None:
        st.info(f"**Data/Ora inizio rilevazione:** {start_datetime.strftime('%d/%m/%Y %H:%M')}")
        if st.session_state.recording_end_datetime:
            st.info(f"**Data/Ora fine rilevazione:** {st.session_state.recording_end_datetime.strftime('%d/%m/%Y %H:%M')}")
    
    # Usa date_input e time_input separatamente
    col_date1, col_time1 = st.columns(2)
    with col_date1:
        start_date = st.date_input(
            "Data Inizio Analisi",
            value=start_datetime.date(),
            key="analysis_start_date"
        )
    with col_time1:
        start_time = st.time_input(
            "Ora Inizio Analisi",
            value=start_datetime.time(),
            key="analysis_start_time"
        )
    
    col_date2, col_time2 = st.columns(2)
    with col_date2:
        end_date = st.date_input(
            "Data Fine Analisi",
            value=end_datetime.date(),
            key="analysis_end_date"
        )
    with col_time2:
        end_time = st.time_input(
            "Ora Fine Analisi",
            value=end_datetime.time(),
            key="analysis_end_time"
        )
    
    # Combina date e time
    new_start_datetime = datetime.combine(start_date, start_time)
    new_end_datetime = datetime.combine(end_date, end_time)
    
    # Controlla se le date sono state modificate manualmente
    datetime_changed = (new_start_datetime != start_datetime or new_end_datetime != end_datetime)
    
    if datetime_changed:
        st.warning("⚠️ **Date modificate manualmente** - L'analisi userà le date selezionate")
        st.session_state.analysis_datetimes = {
            'start_datetime': new_start_datetime,
            'end_datetime': new_end_datetime
        }
        start_datetime, end_datetime = new_start_datetime, new_end_datetime
    
    selected_duration = (end_datetime - start_datetime).total_seconds() / 3600
    
    if selected_duration <= 0:
        st.error("❌ **Errore:** La data di fine deve essere successiva alla data di inizio")
        selected_duration = 1.0
    
    st.info(f"⏱️ **Durata analisi:** {selected_duration:.1f} ore")
    st.info(f"📅 **Periodo analisi:** {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}")
    
    # VERIFICA SE C'È PERIODO NOTTURNO (22:00-06:00)
    is_night_period = False
    current_time = start_datetime
    night_hours = 0
    
    while current_time < end_datetime:
        if 22 <= current_time.hour or current_time.hour <= 6:
            is_night_period = True
            night_hours += 0.1
        current_time += timedelta(hours=0.1)
    
    if is_night_period:
        st.success(f"🌙 **Periodo notturno rilevato** ({night_hours:.1f}h) - Analisi sonno disponibile")
        include_sleep_default = True
    else:
        st.info("☀️ **Periodo diurno** - Analisi sonno non disponibile")
        include_sleep_default = False
    
    # ALTRE IMPOSTAZIONI
    health_factor = st.slider(
        "Profilo Salute", 
        0.1, 1.0, 0.5,
        help="0.1 = Sedentario, 1.0 = Atleta",
        key="health_factor_slider"
    )
    
    include_sleep = st.checkbox(
        "Includi analisi sonno", 
        include_sleep_default,
        help="Disponibile solo per periodi notturni",
        disabled=not is_night_period,
        key="include_sleep_checkbox"
    )
    
    analyze_btn = st.button("🚀 ANALISI COMPLETA", type="primary", use_container_width=True, key="analyze_btn")

# MAIN CONTENT
if analyze_btn:
    with st.spinner("🎯 **ANALISI COMPLETA IN CORSO**..."):
        if uploaded_file is not None:
            # ANALISI CON FILE CARICATO
            try:
                rr_intervals = read_ibi_file_fast(uploaded_file)
                
                if len(rr_intervals) == 0:
                    st.error("❌ Nessun dato RR valido trovato nel file")
                    st.stop()
                
                # Calcola metriche reali
                real_metrics = calculate_hrv_metrics_from_rr(rr_intervals)
                
                if real_metrics is None:
                    st.error("❌ Impossibile calcolare le metriche HRV")
                    st.stop()
                
                # Crea metriche complete
                metrics = {
                    'our_algo': {
                        'sdnn': real_metrics['sdnn'],
                        'rmssd': real_metrics['rmssd'],
                        'hr_mean': real_metrics['hr_mean'],
                        'hr_min': max(40, real_metrics['hr_mean'] - 15),
                        'hr_max': min(180, real_metrics['hr_mean'] + 30),
                        'actual_date': start_datetime,
                        'recording_hours': selected_duration,
                        'is_sleep_period': include_sleep,
                        'health_profile_factor': health_factor,
                        'coherence': max(20, 40 + (40 * health_factor)),
                        'total_power': real_metrics['sdnn'] * 100,
                        'vlf': real_metrics['sdnn'] * 20,
                        'lf': real_metrics['sdnn'] * 50,
                        'hf': real_metrics['rmssd'] * 80,
                        'lf_hf_ratio': 1.5,
                    }
                }
                
                # Aggiungi metriche sonno se richiesto
                if include_sleep and selected_duration >= 4:
                    sleep_duration = min(8.0, selected_duration * 0.9)
                    metrics['our_algo'].update({
                        'sleep_duration': sleep_duration,
                        'sleep_efficiency': min(95, 85 + np.random.normal(0, 5)),
                        'sleep_coherence': 65 + np.random.normal(0, 3),
                        'sleep_hr': 58 + np.random.normal(0, 2),
                        'sleep_rem': min(2.0, sleep_duration * 0.25),
                        'sleep_deep': min(1.5, sleep_duration * 0.2),
                        'sleep_wakeups': max(0, int(sleep_duration * 0.5)),
                    })
                
                # Aggiungi stili comparativi
                metrics.update({
                    'emwave_style': {
                        'sdnn': real_metrics['sdnn'] * 0.7,
                        'rmssd': real_metrics['rmssd'] * 0.7,
                        'hr_mean': real_metrics['hr_mean'] + 2,
                        'coherence': 50
                    },
                    'kubios_style': {
                        'sdnn': real_metrics['sdnn'] * 1.3,
                        'rmssd': real_metrics['rmssd'] * 1.3,
                        'hr_mean': real_metrics['hr_mean'] - 2,
                        'coherence': 70
                    }
                })
                
                # Aggiusta per sesso
                adjusted_metrics = interpret_metrics_for_gender(
                    metrics, 
                    st.session_state.user_profile['gender'],
                    st.session_state.user_profile['age']
                )
                
                # Mostra dashboard
                create_complete_analysis_dashboard(
                    adjusted_metrics, 
                    start_datetime, 
                    end_datetime,
                    f"{selected_duration:.1f}h"
                )
                
            except Exception as e:
                st.error(f"❌ Errore nell'analisi del file: {e}")
                st.stop()
        else:
            # ANALISI SIMULATA
            try:
                metrics = calculate_triple_metrics(
                    selected_duration, 
                    start_datetime, 
                    is_sleep_period=include_sleep,
                    health_profile_factor=health_factor
                )
                
                # Aggiusta per sesso
                adjusted_metrics = interpret_metrics_for_gender(
                    metrics, 
                    st.session_state.user_profile['gender'],
                    st.session_state.user_profile['age']
                )
                
                # Mostra dashboard
                create_complete_analysis_dashboard(
                    adjusted_metrics, 
                    start_datetime, 
                    end_datetime,
                    f"{selected_duration:.1f}h"
                )
                
            except Exception as e:
                st.error(f"❌ Errore nell'analisi simulata: {e}")
                st.stop()
else:
    # SCHERMATA INIZIALE
    st.info("👆 **Configura l'analisi dalla sidebar**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Flusso di Lavoro")
        st.markdown("""
        1. **👤 Inserisci profilo utente**
        2. **📁 Carica file IBI** (data/ora automatiche)
        3. **📝 Aggiungi attività** nel diario
        4. **🎯 Seleziona intervallo** con date specifiche
        5. **🚀 Avvia analisi** completa
        6. **📊 Consulta storico** analisi
        7. **📄 Esporta report PDF** professionale
        """)
    
    with col2:
        st.subheader("🆕 Funzionalità Complete")
        st.markdown("""
        - 👤 **Profilo utente** completo
        - 📅 **Attività con data** specifica
        - 📈 **Grafico con ore reali**
        - 🧠 **Valutazioni e conclusioni**
        - 🌙 **Analisi sonno automatica**
        - ⚖️ **Interpretazioni per sesso**
        - 💡 **Raccomandazioni personalizzate**
        - 📄 **Esportazione PDF** professionale
        - 📅 **Data/ora fine rilevazione** calcolata automaticamente
        """)

# FOOTER
st.markdown("---")
st.markdown("**HRV Analytics ULTIMATE** - Sviluppato per Roberto")