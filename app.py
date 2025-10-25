import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io
import base64
import os

# =============================================================================
# INIZIALIZZAZIONE SEMPLICE
# =============================================================================

def init_session_state():
    if 'activities' not in st.session_state:
        st.session_state.activities = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {'name': '', 'age': 30, 'gender': 'Uomo'}
    if 'rr_data' not in st.session_state:
        st.session_state.rr_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

# =============================================================================
# CALCOLI HRV MOLTO SEMPLICI
# =============================================================================

def simple_hrv_analysis(rr_intervals):
    if len(rr_intervals) < 10:
        return None
    
    rr_array = np.array(rr_intervals)
    
    # Calcoli base
    sdnn = np.std(rr_array)
    rmssd = np.sqrt(np.mean(np.diff(rr_array) ** 2))
    hr_mean = 60000 / np.mean(rr_array)
    
    # Calcoli semplificati per altre metriche
    total_power = sdnn * 50
    lf = total_power * 0.4
    hf = total_power * 0.3
    lf_hf = lf / hf if hf > 0 else 1.0
    
    return {
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'hr_mean': float(hr_mean),
        'total_power': float(total_power),
        'lf': float(lf),
        'hf': float(hf),
        'lf_hf_ratio': float(lf_hf),
        'recording_hours': len(rr_intervals) * np.mean(rr_intervals) / 3600000,
        'data_points': len(rr_intervals)
    }

# =============================================================================
# ANALISI GIORNALIERA SUPER SEMPLICE
# =============================================================================

def analyze_by_days(rr_intervals, start_date):
    if not rr_intervals:
        return []
    
    daily_analyses = []
    current_index = 0
    day_number = 1
    
    while current_index < len(rr_intervals):
        # Prendi circa 24 ore di dati (stimando 60 bpm = 86400 battiti/giorno)
        end_index = min(current_index + 86400, len(rr_intervals))
        day_rr = rr_intervals[current_index:end_index]
        
        if len(day_rr) > 100:  # Almeno 100 battiti per analisi
            metrics = simple_hrv_analysis(day_rr)
            if metrics:
                daily_analyses.append({
                    'day_number': day_number,
                    'date': (start_date + timedelta(days=day_number-1)).date(),
                    'metrics': metrics,
                    'data_points': len(day_rr)
                })
        
        current_index = end_index
        day_number += 1
        
        # Massimo 7 giorni per semplicità
        if day_number > 7:
            break
    
    return daily_analyses

# =============================================================================
# GESTIONE ATTIVITÀ
# =============================================================================

def save_activity(name, activity_type, date, time, duration):
    activity_datetime = datetime.combine(date, time)
    
    activity = {
        'name': name,
        'type': activity_type,
        'datetime': activity_datetime,
        'duration': duration,
        'timestamp': datetime.now()
    }
    
    st.session_state.activities.append(activity)
    st.success(f"Attività '{name}' salvata!")

# =============================================================================
# INTERFACCIA ATTIVITÀ
# =============================================================================

def activity_section():
    st.sidebar.header("📝 Attività")
    
    with st.sidebar.expander("Aggiungi Attività"):
        name = st.text_input("Nome attività")
        activity_type = st.selectbox("Tipo", ["Allenamento", "Alimentazione", "Riposo", "Altro"])
        date = st.date_input("Data", datetime.now())
        time = st.time_input("Ora", datetime.now().time())
        duration = st.number_input("Durata (min)", 1, 480, 30)
        
        if st.button("Salva Attività") and name:
            save_activity(name, activity_type, date, time, duration)
    
    # Mostra attività recenti
    if st.session_state.activities:
        st.sidebar.subheader("Ultime Attività")
        for i, activity in enumerate(st.session_state.activities[-5:]):
            st.sidebar.write(f"• {activity['name']} ({activity['type']})")
            st.sidebar.write(f"  {activity['datetime'].strftime('%d/%m %H:%M')}")

# =============================================================================
# VISUALIZZAZIONE RISULTATI
# =============================================================================

def show_results(metrics, daily_analyses):
    st.header("📊 Risultati Analisi")
    
    # Metriche principali
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SDNN", f"{metrics['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['rmssd']:.1f} ms")
    with col2:
        st.metric("FC Media", f"{metrics['hr_mean']:.1f} bpm")
        st.metric("LF/HF", f"{metrics['lf_hf_ratio']:.2f}")
    with col3:
        st.metric("Durata", f"{metrics['recording_hours']:.1f} h")
        st.metric("Battiti", f"{metrics['data_points']:,}")
    
    # Analisi giornaliera
    if daily_analyses:
        st.header("📅 Analisi Giornaliera")
        
        # Grafico
        days = [f"Giorno {day['day_number']}" for day in daily_analyses]
        sdnn_values = [day['metrics']['sdnn'] for day in daily_analyses]
        rmssd_values = [day['metrics']['rmssd'] for day in daily_analyses]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=sdnn_values, name='SDNN', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=days, y=rmssd_values, name='RMSSD', line=dict(color='red')))
        fig.update_layout(title="Andamento SDNN e RMSSD", height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Dettaglio giorni
        for day in daily_analyses:
            with st.expander(f"Giorno {day['day_number']} - {day['date']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"SDNN: {day['metrics']['sdnn']:.1f} ms")
                    st.write(f"RMSSD: {day['metrics']['rmssd']:.1f} ms")
                    st.write(f"FC: {day['metrics']['hr_mean']:.1f} bpm")
                with col2:
                    st.write(f"LF/HF: {day['metrics']['lf_hf_ratio']:.2f}")
                    st.write(f"Battiti: {day['data_points']:,}")
                    st.write(f"Ore: {day['metrics']['recording_hours']:.1f} h")

# =============================================================================
# GENERAZIONE PDF SEMPLICE
# =============================================================================

def create_simple_pdf(metrics, daily_analyses):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        
        # Contenuto semplice
        c.drawString(100, 800, "REPORT HRV ANALYTICS")
        c.drawString(100, 780, f"Generato il: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        c.drawString(100, 750, "METRICHE PRINCIPALI:")
        c.drawString(100, 730, f"SDNN: {metrics['sdnn']:.1f} ms")
        c.drawString(100, 710, f"RMSSD: {metrics['rmssd']:.1f} ms")
        c.drawString(100, 690, f"FC Media: {metrics['hr_mean']:.1f} bpm")
        c.drawString(100, 670, f"LF/HF: {metrics['lf_hf_ratio']:.2f}")
        
        if daily_analyses:
            c.drawString(100, 640, "ANALISI GIORNALIERA:")
            y = 620
            for day in daily_analyses:
                c.drawString(100, y, f"Giorno {day['day_number']}: SDNN={day['metrics']['sdnn']:.1f}, RMSSD={day['metrics']['rmssd']:.1f}")
                y -= 20
                if y < 100:
                    c.showPage()
                    y = 800
        
        c.save()
        buffer.seek(0)
        return buffer
    except:
        return None

# =============================================================================
# INTERFACCIA PRINCIPALE
# =============================================================================

def main():
    st.set_page_config(
        page_title="HRV Analytics",
        page_icon="❤️",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("❤️ HRV Analytics")
    st.write("Analisi semplice della Variabilità della Frequenza Cardiaca")
    
    # Sidebar
    activity_section()
    
    # Upload file
    st.header("📤 Carica Dati")
    uploaded_file = st.file_uploader("Carica file con intervalli RR (un valore per riga)", type=['txt', 'csv'])
    
    if uploaded_file:
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            rr_intervals = []
            for line in lines:
                try:
                    val = float(line)
                    if 300 < val < 2000:  # Range fisiologico
                        rr_intervals.append(val)
                except:
                    continue
            
            if rr_intervals:
                st.success(f"✅ Caricati {len(rr_intervals)} intervalli RR")
                st.session_state.rr_data = rr_intervals
                
                # Analisi immediata
                if st.button("🔍 Analizza Dati", type="primary"):
                    with st.spinner("Analisi in corso..."):
                        # Analisi complessiva
                        metrics = simple_hrv_analysis(rr_intervals)
                        
                        # Analisi giornaliera
                        start_date = datetime.now().date()
                        daily_analyses = analyze_by_days(rr_intervals, start_date)
                        
                        st.session_state.analysis_results = {
                            'metrics': metrics,
                            'daily_analyses': daily_analyses
                        }
                        
                        st.rerun()
            
            else:
                st.error("Nessun dato RR valido trovato")
                
        except Exception as e:
            st.error(f"Errore: {str(e)}")
    
    # Mostra risultati
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        show_results(results['metrics'], results['daily_analyses'])
        
        # Bottone PDF
        if st.button("📄 Genera Report PDF"):
            pdf_buffer = create_simple_pdf(results['metrics'], results['daily_analyses'])
            if pdf_buffer:
                b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="hrv_report.pdf" style="background:#4CAF50; color:white; padding:10px; border-radius:5px; text-decoration:none;">Scarica PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Errore nella generazione PDF")

if __name__ == "__main__":
    main()