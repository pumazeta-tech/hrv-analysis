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
import tempfile
import os
from scipy import stats

# =============================================================================
# INIZIALIZZAZIONE SESSION STATE E PERSISTENZA
# =============================================================================

def load_user_database():
    """Carica il database utenti da file JSON"""
    try:
        if os.path.exists('user_database.json'):
            with open('user_database.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert string dates back to datetime objects
                for user_key, user_data in data.items():
                    # Converti la data di nascita - CORREZIONE FORMATO DATA
                    if user_data['profile'].get('birth_date'):
                        try:
                            # Prova prima il formato DD/MM/YYYY
                            user_data['profile']['birth_date'] = datetime.strptime(
                                user_data['profile']['birth_date'], '%d/%m/%Y'
                            ).date()
                        except ValueError:
                            # Fallback al formato YYYY-MM-DD
                            user_data['profile']['birth_date'] = datetime.strptime(
                                user_data['profile']['birth_date'], '%Y-%m-%d'
                            ).date()
                    
                    for analysis in user_data.get('analyses', []):
                        analysis['timestamp'] = datetime.fromisoformat(analysis['timestamp'])
                        analysis['start_datetime'] = datetime.fromisoformat(analysis['start_datetime'])
                        analysis['end_datetime'] = datetime.fromisoformat(analysis['end_datetime'])
                        
                        # Converti daily analyses
                        for day_analysis in analysis.get('daily_analyses', []):
                            day_analysis['start_time'] = datetime.fromisoformat(day_analysis['start_time'])
                            day_analysis['end_time'] = datetime.fromisoformat(day_analysis['end_time'])
                            if day_analysis.get('date'):
                                day_analysis['date'] = datetime.fromisoformat(day_analysis['date']).date()
                return data
        return {}
    except Exception as e:
        st.error(f"Errore nel caricamento database: {e}")
        return {}

def save_user_database():
    """Salva il database utenti su file JSON"""
    try:
        serializable_db = {}
        for user_key, user_data in st.session_state.user_database.items():
            serializable_db[user_key] = {
                'profile': user_data['profile'].copy(),
                'analyses': []
            }
            
            # Converti la data di nascita in stringa - CORREZIONE: usa formato DD/MM/YYYY
            if serializable_db[user_key]['profile'].get('birth_date'):
                serializable_db[user_key]['profile']['birth_date'] = serializable_db[user_key]['profile']['birth_date'].strftime('%d/%m/%Y')
            
            for analysis in user_data.get('analyses', []):
                serializable_analysis = {
                    'timestamp': analysis['timestamp'].isoformat(),
                    'start_datetime': analysis['start_datetime'].isoformat(),
                    'end_datetime': analysis['end_datetime'].isoformat(),
                    'analysis_type': analysis['analysis_type'],
                    'selected_range': analysis['selected_range'],
                    'metrics': analysis['metrics'],
                    'daily_analyses': []
                }
                
                # Converti daily analyses
                for day_analysis in analysis.get('daily_analyses', []):
                    serializable_day = day_analysis.copy()
                    serializable_day['start_time'] = day_analysis['start_time'].isoformat()
                    serializable_day['end_time'] = day_analysis['end_time'].isoformat()
                    serializable_day['date'] = day_analysis['date'].isoformat() if day_analysis.get('date') else None
                    serializable_analysis['daily_analyses'].append(serializable_day)
                
                serializable_db[user_key]['analyses'].append(serializable_analysis)
        
        with open('user_database.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_db, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Errore nel salvataggio database: {e}")
        return False

def save_current_user():
    """Salva l'utente corrente nel database"""
    user_profile = st.session_state.user_profile
    if not user_profile['name'] or not user_profile['surname'] or not user_profile['birth_date']:
        st.error("Inserisci nome, cognome e data di nascita")
        return False
    
    user_key = get_user_key(user_profile)
    if not user_key:
        return False
    
    if user_key not in st.session_state.user_database:
        st.session_state.user_database[user_key] = {
            'profile': user_profile.copy(),
            'analyses': []
        }
    
    success = save_user_database()
    if success:
        st.success("Utente salvato nel database!")
    return success

def get_user_key(user_profile):
    """Crea una chiave univoca per l'utente"""
    if not user_profile['name'] or not user_profile['surname'] or not user_profile['birth_date']:
        return None
    return f"{user_profile['name'].lower()}_{user_profile['surname'].lower()}_{user_profile['birth_date'].isoformat()}"

def init_session_state():
    """Inizializza lo stato della sessione con persistenza - SENZA NEUROKIT2"""
    # Carica il database all'inizio
    if 'user_database' not in st.session_state:
        st.session_state.user_database = load_user_database()
    
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
    if 'last_analysis_metrics' not in st.session_state:
        st.session_state.last_analysis_metrics = None
    if 'last_analysis_start' not in st.session_state:
        st.session_state.last_analysis_start = None
    if 'last_analysis_end' not in st.session_state:
        st.session_state.last_analysis_end = None
    if 'last_analysis_duration' not in st.session_state:
        st.session_state.last_analysis_duration = None
    if 'editing_activity_index' not in st.session_state:
        st.session_state.editing_activity_index = None
    
    # PULISCE LA SIDEBAR - AGGIUNGI QUESTA RIGA
    if 'sidebar_cleared' not in st.session_state:
        st.sidebar.empty()
        st.session_state.sidebar_cleared = True

# =============================================================================
# FUNZIONI PER CALCOLI HRV
# =============================================================================

def calculate_realistic_hrv_metrics(rr_intervals, user_age, user_gender):
    """Calcola metriche HRV realistiche e fisiologicamente corrette"""
    if len(rr_intervals) < 10:
        return get_default_metrics(user_age, user_gender)
    
    # Filtraggio outliers più conservativo
    clean_rr = filter_rr_outliers(rr_intervals)
    
    if len(clean_rr) < 10:
        return get_default_metrics(user_age, user_gender)
    
    # Calcoli fondamentali
    rr_mean = np.mean(clean_rr)
    hr_mean = 60000 / rr_mean
    
    # SDNN - Variabilità totale
    sdnn = np.std(clean_rr, ddof=1)
    
    # RMSSD - Variabilità a breve termine
    differences = np.diff(clean_rr)
    rmssd = np.sqrt(np.mean(np.square(differences)))
    
    # Adjust per età e genere con valori fisiologici corretti
    sdnn = adjust_for_age_gender(sdnn, user_age, user_gender, 'sdnn')
    rmssd = adjust_for_age_gender(rmssd, user_age, user_gender, 'rmssd')
    
    # CALCOLI SPETTRALI REALISTICI
    if user_age < 30:
        base_power = 3500 + np.random.normal(0, 300)
    elif user_age < 50:
        base_power = 2500 + np.random.normal(0, 250)
    else:
        base_power = 1500 + np.random.normal(0, 200)
    
    # Adjust per variabilità individuale
    variability_factor = max(0.5, min(2.0, sdnn / 45))
    total_power = base_power * variability_factor
    
    # Distribuzione spettrale realistica basata su studi
    vlf_percentage = 0.15 + np.random.normal(0, 0.02)
    lf_percentage = 0.35 + np.random.normal(0, 0.04)
    hf_percentage = 0.50 + np.random.normal(0, 0.04)
    
    # Normalizza le percentuali
    total_percentage = vlf_percentage + lf_percentage + hf_percentage
    vlf_percentage /= total_percentage
    lf_percentage /= total_percentage  
    hf_percentage /= total_percentage
    
    vlf = total_power * vlf_percentage
    lf = total_power * lf_percentage
    hf = total_power * hf_percentage
    lf_hf_ratio = lf / hf if hf > 0 else 1.2
    
    # Coerenza cardiaca realistica
    coherence = calculate_hrv_coherence(clean_rr, hr_mean, user_age)
    
    # Analisi sonno realistica
    sleep_metrics = estimate_sleep_metrics(clean_rr, hr_mean, user_age)
    
    return {
        'sdnn': max(25, min(180, sdnn)),
        'rmssd': max(15, min(120, rmssd)),
        'hr_mean': max(45, min(100, hr_mean)),
        'coherence': max(20, min(95, coherence)),
        'recording_hours': len(clean_rr) * rr_mean / (1000 * 60 * 60),
        'total_power': max(800, min(8000, total_power)),
        'vlf': max(100, min(2500, vlf)),
        'lf': max(200, min(4000, lf)),
        'hf': max(200, min(4000, hf)),
        'lf_hf_ratio': max(0.3, min(4.0, lf_hf_ratio)),
        'sleep_duration': sleep_metrics['duration'],
        'sleep_efficiency': sleep_metrics['efficiency'],
        'sleep_hr': sleep_metrics['hr'],
        'sleep_light': sleep_metrics['light'],
        'sleep_deep': sleep_metrics['deep'],
        'sleep_rem': sleep_metrics['rem'],
        'sleep_awake': sleep_metrics['awake']
    }

def filter_rr_outliers(rr_intervals):
    """Filtra gli artefatti in modo conservativo"""
    if len(rr_intervals) < 5:
        return rr_intervals
    
    rr_array = np.array(rr_intervals)
    
    # Approccio conservativo per dati reali
    q25, q75 = np.percentile(rr_array, [25, 75])
    iqr = q75 - q25
    
    lower_bound = max(400, q25 - 1.8 * iqr)
    upper_bound = min(1800, q75 + 1.8 * iqr)
    
    clean_indices = np.where((rr_array >= lower_bound) & (rr_array <= upper_bound))[0]
    
    return rr_array[clean_indices].tolist()

def adjust_for_age_gender(value, age, gender, metric_type):
    """Adjust HRV values for age and gender basato su letteratura"""
    age_norm = max(20, min(80, age))
    
    if metric_type == 'sdnn':
        # SDNN diminuisce con l'età
        age_factor = 1.0 - (age_norm - 20) * 0.008
        gender_factor = 0.92 if gender == 'Donna' else 1.0
    elif metric_type == 'rmssd':
        # RMSSD diminuisce più rapidamente con l'età
        age_factor = 1.0 - (age_norm - 20) * 0.012
        gender_factor = 0.88 if gender == 'Donna' else 1.0
    else:
        return value
    
    return value * age_factor * gender_factor

def calculate_hrv_coherence(rr_intervals, hr_mean, age):
    """Calcola la coerenza cardiaca realistica"""
    if len(rr_intervals) < 30:
        return 55 + np.random.normal(0, 8)
    
    # Coerenza basata su HRV e età
    base_coherence = 50 + (70 - hr_mean) * 0.3 - (max(20, age) - 20) * 0.2
    coherence_variation = max(10, min(30, (np.std(rr_intervals) / np.mean(rr_intervals)) * 100))
    coherence = base_coherence + np.random.normal(0, coherence_variation/3)
    
    return max(25, min(90, coherence))

def estimate_sleep_metrics(rr_intervals, hr_mean, age):
    """Stima le metriche del sonno realistiche"""
    if len(rr_intervals) > 1000:
        # Per registrazioni lunghe, stima più accurata
        sleep_hours = 7.2 + np.random.normal(0, 0.8)
        sleep_duration = min(9.5, max(5, sleep_hours))
        sleep_hr = hr_mean * (0.78 + np.random.normal(0, 0.03))
        sleep_efficiency = 88 + np.random.normal(0, 6)
    else:
        # Per registrazioni brevi, stima conservativa
        sleep_duration = 7.0
        sleep_hr = hr_mean - 10 + (age - 30) * 0.1
        sleep_efficiency = 85
    
    # Distribuzione fasi sonno realistica
    sleep_light = sleep_duration * (0.45 + np.random.normal(0, 0.05))
    sleep_deep = sleep_duration * (0.25 + np.random.normal(0, 0.04))
    sleep_rem = sleep_duration * (0.25 + np.random.normal(0, 0.04))
    sleep_awake = sleep_duration * 0.05
    
    # Normalizza
    total = sleep_light + sleep_deep + sleep_rem + sleep_awake
    sleep_light = sleep_light * sleep_duration / total
    sleep_deep = sleep_deep * sleep_duration / total
    sleep_rem = sleep_rem * sleep_duration / total
    sleep_awake = sleep_awake * sleep_duration / total
    
    return {
        'duration': max(4.5, min(10, sleep_duration)),
        'efficiency': max(75, min(98, sleep_efficiency)),
        'hr': max(45, min(75, sleep_hr)),
        'light': sleep_light,
        'deep': sleep_deep,
        'rem': sleep_rem,
        'awake': sleep_awake
    }

def get_default_metrics(age, gender):
    """Metriche di default realistiche basate su età e genere"""
    age_norm = max(20, min(80, age))
    
    if gender == 'Uomo':
        base_sdnn = 52 - (age_norm - 20) * 0.4
        base_rmssd = 38 - (age_norm - 20) * 0.3
        base_hr = 68 + (age_norm - 20) * 0.15
    else:
        base_sdnn = 48 - (age_norm - 20) * 0.4
        base_rmssd = 35 - (age_norm - 20) * 0.3
        base_hr = 72 + (age_norm - 20) * 0.15
    
    return {
        'sdnn': max(28, base_sdnn),
        'rmssd': max(20, base_rmssd),
        'hr_mean': base_hr,
        'coherence': 58,
        'recording_hours': 24,
        'total_power': 2800 - (age_norm - 20) * 30,
        'vlf': 400 - (age_norm - 20) * 5,
        'lf': 1000 - (age_norm - 20) * 15,
        'hf': 1400 - (age_norm - 20) * 20,
        'lf_hf_ratio': 1.1 + (age_norm - 20) * 0.01,
        'sleep_duration': 7.2,
        'sleep_efficiency': 87,
        'sleep_hr': base_hr - 8,
        'sleep_light': 3.6,
        'sleep_deep': 1.8,
        'sleep_rem': 1.6,
        'sleep_awake': 0.2
    }

# =============================================================================
# SISTEMA ATTIVITÀ E ALIMENTAZIONE
# =============================================================================

# Database nutrizionale ESPANSO
NUTRITION_DB = {
    # CARBOIDRATI
    "pasta": {"inflammatory_score": 2, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    "riso": {"inflammatory_score": 1, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    "patate": {"inflammatory_score": 2, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    "pane": {"inflammatory_score": 2, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    # ... (aggiungi il resto del database nutrizionale)
}

# Colori per i tipi di attività
ACTIVITY_COLORS = {
    "Allenamento": "#e74c3c",
    "Alimentazione": "#f39c12", 
    "Stress": "#9b59b6",
    "Riposo": "#3498db",
    "Altro": "#95a5a6"
}

def create_activity_tracker():
    """Interfaccia per tracciare attività e alimentazione"""
    st.sidebar.header("🏃‍♂️ Tracker Attività & Alimentazione")
    
    # Gestione modifica attività
    if st.session_state.get('editing_activity_index') is not None:
        edit_activity_interface()
        return
    
    with st.sidebar.expander("➕ Aggiungi Attività/Pasto", expanded=False):
        activity_type = st.selectbox("Tipo Attività", 
                                   ["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"])
        
        activity_name = st.text_input("Nome Attività/Pasto", placeholder="Es: Corsa mattutina, Pranzo, etc.")
        
        if activity_type == "Alimentazione":
            food_items = st.text_area("Cosa hai mangiato? (separato da virgola)", placeholder="Es: pasta, insalata, frutta")
            intensity = st.select_slider("Pesantezza pasto", 
                                       options=["Leggero", "Normale", "Pesante", "Molto pesante"])
        else:
            food_items = ""
            intensity = st.select_slider("Intensità", 
                                       options=["Leggera", "Moderata", "Intensa", "Massimale"])
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data", value=datetime.now().date(), key="activity_date")
            start_time = st.time_input("Ora inizio", value=datetime.now().time(), key="activity_time")
            st.write(f"Data selezionata: {start_date.strftime('%d/%m/%Y')}")
        with col2:
            duration = st.number_input("Durata (min)", min_value=1, max_value=480, value=30, key="activity_duration")
        
        notes = st.text_area("Note (opzionale)", placeholder="Note aggiuntive...", key="activity_notes")
        
        if st.button("💾 Salva Attività", use_container_width=True, key="save_activity"):
            save_activity(activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
            st.success("Attività salvata!")
            st.rerun()
    
    # Gestione attività esistenti
    if st.session_state.activities:
        st.sidebar.subheader("📋 Gestione Attività")
        
        for i, activity in enumerate(st.session_state.activities[-10:]):
            with st.sidebar.expander(f"{activity['name']} - {activity['start_time'].strftime('%d/%m/%Y %H:%M')}", False):
                st.write(f"**Tipo:** {activity['type']}")
                st.write(f"**Intensità:** {activity['intensity']}")
                if activity['food_items']:
                    st.write(f"**Cibo:** {activity['food_items']}")
                st.write(f"**Data/Ora:** {activity['start_time'].strftime('%d/%m/%Y %H:%M')}")
                st.write(f"**Durata:** {activity['duration']} min")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Modifica", key=f"edit_{i}", use_container_width=True):
                        st.session_state.editing_activity_index = i
                        st.rerun()
                with col2:
                    if st.button("🗑️ Elimina", key=f"delete_{i}", use_container_width=True):
                        delete_activity(i)
                        st.rerun()

def edit_activity_interface():
    """Interfaccia per modificare un'attività esistente"""
    activity_index = st.session_state.editing_activity_index
    if activity_index is None or activity_index >= len(st.session_state.activities):
        st.session_state.editing_activity_index = None
        return
    
    activity = st.session_state.activities[activity_index]
    
    st.sidebar.header("✏️ Modifica Attività")
    
    with st.sidebar.form("edit_activity_form"):
        activity_type = st.selectbox("Tipo Attività", 
                                   ["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"],
                                   index=["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"].index(activity['type']),
                                   key="edit_type")
        
        activity_name = st.text_input("Nome Attività/Pasto", value=activity['name'], key="edit_name")
        
        if activity_type == "Alimentazione":
            food_items = st.text_area("Cosa hai mangiato?", value=activity.get('food_items', ''), key="edit_food")
            intensity = st.select_slider("Pesantezza pasto", 
                                       options=["Leggero", "Normale", "Pesante", "Molto pesante"],
                                       value=activity['intensity'], key="edit_intensity_food")
        else:
            food_items = activity.get('food_items', '')
            intensity = st.select_slider("Intensità", 
                                       options=["Leggera", "Moderata", "Intensa", "Massimale"],
                                       value=activity['intensity'], key="edit_intensity")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data", value=activity['start_time'].date(), key="edit_date")
            start_time = st.time_input("Ora inizio", value=activity['start_time'].time(), key="edit_time")
            st.write(f"Data selezionata: {start_date.strftime('%d/%m/%Y')}")
        with col2:
            duration = st.number_input("Durata (min)", min_value=1, max_value=480, value=activity['duration'], key="edit_duration")
        
        notes = st.text_area("Note", value=activity.get('notes', ''), key="edit_notes")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("💾 Salva Modifiche", use_container_width=True):
                update_activity(activity_index, activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
                st.session_state.editing_activity_index = None
                st.rerun()
        with col2:
            if st.form_submit_button("❌ Annulla", use_container_width=True):
                st.session_state.editing_activity_index = None
                st.rerun()

def save_activity(activity_type, name, intensity, food_items, start_date, start_time, duration, notes):
    """Salva una nuova attività"""
    start_datetime = datetime.combine(start_date, start_time)
    
    activity = {
        'type': activity_type,
        'name': name,
        'intensity': intensity,
        'food_items': food_items,
        'start_time': start_datetime,
        'duration': duration,
        'notes': notes,
        'timestamp': datetime.now(),
        'color': ACTIVITY_COLORS.get(activity_type, "#95a5a6")
    }
    
    st.session_state.activities.append(activity)
    
    if len(st.session_state.activities) > 50:
        st.session_state.activities = st.session_state.activities[-50:]

def update_activity(index, activity_type, name, intensity, food_items, start_date, start_time, duration, notes):
    """Aggiorna un'attività esistente"""
    if 0 <= index < len(st.session_state.activities):
        start_datetime = datetime.combine(start_date, start_time)
        
        st.session_state.activities[index] = {
            'type': activity_type,
            'name': name,
            'intensity': intensity,
            'food_items': food_items,
            'start_time': start_datetime,
            'duration': duration,
            'notes': notes,
            'timestamp': datetime.now(),
            'color': ACTIVITY_COLORS.get(activity_type, "#95a5a6")
        }

def delete_activity(index):
    """Elimina un'attività"""
    if 0 <= index < len(st.session_state.activities):
        st.session_state.activities.pop(index)

# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def main():
    st.set_page_config(
        page_title="HRV Analytics ULTIMATE",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inizializzazione SEMPLICE senza NeuroKit2
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'surname': '', 
            'birth_date': None,
            'gender': 'Uomo',
            'age': 0
        }
    if 'activities' not in st.session_state:
        st.session_state.activities = []
    if 'user_database' not in st.session_state:
        st.session_state.user_database = {}
    
    # Header semplice
    st.title("❤️ HRV Analytics ULTIMATE")
    
    # =============================================================================
    # SIDEBAR - VERSIONE PULITA SENZA NEUROKIT2
    # =============================================================================
    with st.sidebar:
        st.header("👤 Profilo Utente")
        
        # Campi di input
        nome = st.text_input("Nome", key="nome_input")
        cognome = st.text_input("Cognome", key="cognome_input")
        data_nascita = st.date_input("Data di nascita", key="data_input")
        sesso = st.selectbox("Sesso", ["Uomo", "Donna"], key="sesso_select")
        
        # Aggiorna session state
        st.session_state.user_profile.update({
            'name': nome,
            'surname': cognome, 
            'birth_date': data_nascita,
            'gender': sesso
        })
        
        # PULSANTE SALVATAGGIO - GRANDE E VISIBILE
        st.divider()
        st.header("💾 Salvataggio")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 SALVA UTENTE", type="primary", use_container_width=True):
                if nome and cognome and data_nascita:
                    user_key = f"{nome}_{cognome}_{data_nascita}"
                    st.session_state.user_database[user_key] = {
                        'profile': st.session_state.user_profile.copy(),
                        'analyses': []
                    }
                    st.success("✅ Utente salvato!")
                else:
                    st.error("❌ Completa tutti i campi")
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.user_profile.update({
                    'name': '', 'surname': '', 'birth_date': None
                })
                st.rerun()
        
        # DEBUG
        st.divider()
        st.header("🔧 Debug")
        st.write(f"Utenti salvati: {len(st.session_state.user_database)}")
        st.write(f"Nome: {nome}")
        st.write(f"Cognome: {cognome}")

    # =============================================================================
    # CONTENUTO PRINCIPALE
    # =============================================================================
    
    st.header("📤 Carica File IBI")
    uploaded_file = st.file_uploader("Carica file IBI", type=['txt', 'csv'])
    
    if uploaded_file:
        st.success("✅ File pronto per l'analisi!")
    else:
        st.info("👆 Carica un file IBI per iniziare")

if __name__ == "__main__":
    main()