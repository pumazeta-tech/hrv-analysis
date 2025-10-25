import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64

# Inizializzazione session state
if 'activities' not in st.session_state:
    st.session_state.activities = []
if 'rr_data' not in st.session_state:
    st.session_state.rr_data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Configurazione pagina
st.set_page_config(page_title="HRV Analytics", page_icon="❤️", layout="wide")

st.title("❤️ HRV Analytics - Versione Semplice")
st.write("Analisi della Variabilità della Frequenza Cardiaca")

# =============================================================================
# FUNZIONI HRV SEMPLICI
# =============================================================================

def calculate_hrv(rr_intervals):
    """Calcola metriche HRV di base"""
    if len(rr_intervals) < 10:
        return None
    
    rr = np.array(rr_intervals)
    
    # Calcoli base
    sdnn = np.std(rr)
    differences = np.diff(rr)
    rmssd = np.sqrt(np.mean(differences ** 2))
    hr_mean = 60000 / np.mean(rr)
    
    return {
        'sdnn': float(sdnn),
        'rmssd': float(rmssd), 
        'hr_mean': float(hr_mean),
        'total_power': float(sdnn * 60),
        'lf': float(sdnn * 25),
        'hf': float(rmssd * 20),
        'lf_hf_ratio': float((sdnn * 25) / (rmssd * 20)) if rmssd > 0 else 1.0,
        'data_points': len(rr_intervals),
        'recording_hours': len(rr_intervals) * np.mean(rr) / 3600000
    }

def analyze_daily(rr_intervals, start_date):
    """Divide l'analisi per giorni"""
    if not rr_intervals:
        return []
    
    daily_results = []
    current_index = 0
    day_num = 1
    
    # Stima: circa 1000 battiti/ora = 24000 battiti/giorno
    while current_index < len(rr_intervals):
        end_index = min(current_index + 24000, len(rr_intervals))
        day_data = rr_intervals[current_index:end_index]
        
        if len(day_data) > 500:  # Almeno 500 battiti
            metrics = calculate_hrv(day_data)
            if metrics:
                daily_results.append({
                    'day_number': day_num,
                    'date': (start_date + timedelta(days=day_num-1)).strftime('%d/%m/%Y'),
                    'metrics': metrics,
                    'data_points': len(day_data)
                })
        
        current_index = end_index
        day_num += 1
        
        if day_num > 10:  # Massimo 10 giorni
            break
    
    return daily_results

# =============================================================================
# GESTIONE ATTIVITÀ
# =============================================================================

def activity_manager():
    """Gestione semplice delle attività"""
    st.sidebar.header("📝 Gestione Attività")
    
    # Aggiungi attività
    with st.sidebar.expander("➕ Nuova Attività", expanded=True):
        name = st.text_input("Nome attività")
        act_type = st.selectbox("Tipo", ["Allenamento", "Alimentazione", "Riposo", "Lavoro", "Altro"])
        date = st.date_input("Data", datetime.now())
        time = st.time_input("Ora", datetime.now().time())
        duration = st.slider("Durata (minuti)", 1, 240, 30)
        
        if st.button("💾 Salva Attività", use_container_width=True) and name.strip():
            activity = {
                'name': name,
                'type': act_type,
                'datetime': datetime.combine(date, time),
                'duration': duration,
                'added': datetime.now()
            }
            st.session_state.activities.append(activity)
            st.sidebar.success("Attività salvata!")
    
    # Lista attività
    if st.session_state.activities:
        st.sidebar.subheader("📋 Attività Salvate")
        for i, act in enumerate(st.session_state.activities[-5:]):  # Ultime 5
            st.sidebar.write(f"**{act['name']}**")
            st.sidebar.write(f"{act['type']} - {act['datetime'].strftime('%d/%m %H:%M')}")
            st.sidebar.write("---")

# =============================================================================
# VISUALIZZAZIONE
# =============================================================================

def show_daily_analysis(daily_data):
    """Mostra analisi giornaliera"""
    st.header("📅 Analisi Giornaliera")
    
    if not daily_data:
        st.info("Nessuna analisi giornaliera disponibile")
        return
    
    # Grafico andamento
    days = [f"Giorno {day['day_number']}" for day in daily_data]
    sdnn_vals = [day['metrics']['sdnn'] for day in daily_data]
    rmssd_vals = [day['metrics']['rmssd'] for day in daily_data]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=sdnn_vals, name='SDNN', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=days, y=rmssd_vals, name='RMSSD', line=dict(color='red', width=3)))
    
    fig.update_layout(
        title='Andamento SDNN e RMSSD',
        xaxis_title='Giorno',
        yaxis_title='Valore (ms)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Dettaglio per giorno
    st.subheader("Dettaglio per Giorno")
    for day in daily_data:
        with st.expander(f"🗓️ {day['date']} - Giorno {day['day_number']}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("SDNN", f"{day['metrics']['sdnn']:.1f} ms")
                st.metric("RMSSD", f"{day['metrics']['rmssd']:.1f} ms")
            with col2:
                st.metric("FC Media", f"{day['metrics']['hr_mean']:.1f} bpm")
                st.metric("LF/HF", f"{day['metrics']['lf_hf_ratio']:.2f}")
            with col3:
                st.metric("Battiti", f"{day['data_points']:,}")
                st.metric("Ore", f"{day['metrics']['recording_hours']:.1f} h")
            with col4:
                st.metric("Potenza", f"{day['metrics']['total_power']:.0f}")
            
            # Attività per questo giorno
            day_date = datetime.strptime(day['date'], '%d/%m/%Y').date()
            day_activities = [
                act for act in st.session_state.activities 
                if act['datetime'].date() == day_date
            ]
            
            if day_activities:
                st.write("**Attività del giorno:**")
                for act in day_activities:
                    st.write(f"• {act['name']} ({act['type']}) - {act['datetime'].strftime('%H:%M')}")

# =============================================================================
# GENERAZIONE PDF
# =============================================================================

def create_pdf_report(overall_metrics, daily_data):
    """Crea un PDF semplice"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        
        # Titolo
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 800, "REPORT HRV ANALYTICS")
        c.setFont("Helvetica", 10)
        c.drawString(100, 780, f"Generato il: {datetime.now().strftime('%d/%m/%Y alle %H:%M')}")
        
        # Metriche principali
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, 750, "METRICHE PRINCIPALI:")
        c.setFont("Helvetica", 10)
        
        y_pos = 730
        metrics_text = [
            f"SDNN: {overall_metrics['sdnn']:.1f} ms",
            f"RMSSD: {overall_metrics['rmssd']:.1f} ms", 
            f"Frequenza Cardiaca Media: {overall_metrics['hr_mean']:.1f} bpm",
            f"LF/HF Ratio: {overall_metrics['lf_hf_ratio']:.2f}",
            f"Battiti analizzati: {overall_metrics['data_points']:,}",
            f"Durata registrazione: {overall_metrics['recording_hours']:.1f} ore"
        ]
        
        for text in metrics_text:
            c.drawString(100, y_pos, text)
            y_pos -= 20
        
        # Analisi giornaliera
        if daily_data:
            y_pos -= 30
            c.setFont("Helvetica-Bold", 12)
            c.drawString(100, y_pos, "ANALISI GIORNALIERA:")
            y_pos -= 20
            c.setFont("Helvetica", 9)
            
            for day in daily_data:
                day_text = f"{day['date']}: SDNN={day['metrics']['sdnn']:.1f}, RMSSD={day['metrics']['rmssd']:.1f}, FC={day['metrics']['hr_mean']:.1f}"
                c.drawString(100, y_pos, day_text)
                y_pos -= 15
                
                if y_pos < 50:  # Nuova pagina se necessario
                    c.showPage()
                    y_pos = 800
                    c.setFont("Helvetica", 9)
        
        c.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nella creazione del PDF: {e}")
        return None

# =============================================================================
# INTERFACCIA PRINCIPALE
# =============================================================================

def main():
    # Sidebar con attività
    activity_manager()
    
    # Sezione principale
    st.header("📤 Carica Dati HRV")
    
    uploaded_file = st.file_uploader(
        "Seleziona il file con gli intervalli RR (formato: un numero per riga)",
        type=['txt', 'csv'],
        help="Il file deve contenere gli intervalli RR in millisecondi, uno per riga"
    )
    
    if uploaded_file is not None:
        try:
            # Leggi il file
            content = uploaded_file.getvalue().decode('utf-8').strip()
            lines = content.split('\n')
            
            # Processa i dati RR
            rr_intervals = []
            valid_count = 0
            
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        value = float(line)
                        # Filtra valori non fisiologici
                        if 300 <= value <= 2000:
                            rr_intervals.append(value)
                            valid_count += 1
                    except ValueError:
                        continue
            
            if valid_count == 0:
                st.error("❌ Nessun dato RR valido trovato nel file.")
                return
            
            st.success(f"✅ File elaborato con successo! Trovati {valid_count} intervalli RR validi.")
            
            # Salva i dati
            st.session_state.rr_data = rr_intervals
            
            # Pulsante analisi
            st.header("🔍 Analisi Dati")
            
            if st.button("🚀 AVVIA ANALISI COMPLETA", type="primary", use_container_width=True):
                with st.spinner("Analisi in corso..."):
                    # Analisi complessiva
                    overall_metrics = calculate_hrv(rr_intervals)
                    
                    if overall_metrics is None:
                        st.error("Dati insufficienti per l'analisi")
                        return
                    
                    # Analisi giornaliera  
                    start_date = datetime.now().date()
                    daily_analysis = analyze_daily(rr_intervals, start_date)
                    
                    # Salva risultati
                    st.session_state.analysis_results = {
                        'overall': overall_metrics,
                        'daily': daily_analysis
                    }
                    st.session_state.analysis_done = True
                    
                    st.success("🎉 Analisi completata!")
            
            # Mostra risultati se disponibili
            if st.session_state.analysis_done and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                
                # Metriche principali
                st.header("📊 Risultati Principali")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("SDNN", f"{results['overall']['sdnn']:.1f} ms")
                    st.metric("RMSSD", f"{results['overall']['rmssd']:.1f} ms")
                with col2:
                    st.metric("FC Media", f"{results['overall']['hr_mean']:.1f} bpm")
                    st.metric("LF/HF Ratio", f"{results['overall']['lf_hf_ratio']:.2f}")
                with col3:
                    st.metric("Battiti Totali", f"{results['overall']['data_points']:,}")
                    st.metric("Durata", f"{results['overall']['recording_hours']:.1f} h")
                with col4:
                    st.metric("Potenza Totale", f"{results['overall']['total_power']:.0f}")
                
                # Analisi giornaliera
                show_daily_analysis(results['daily'])
                
                # Genera PDF
                st.header("📄 Report PDF")
                if st.button("🖨️ Genera Report PDF", use_container_width=True):
                    pdf_buffer = create_pdf_report(results['overall'], results['daily'])
                    if pdf_buffer:
                        # Crea link download
                        b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                        pdf_display = f'''
                        <a href="data:application/pdf;base64,{b64_pdf}" download="report_hrv.pdf" 
                           style="background-color:#4CAF50; color:white; padding:12px 20px; text-decoration:none; border-radius:5px; display:inline-block; text-align:center;">
                           📥 Scarica Report PDF
                        </a>
                        '''
                        st.markdown(pdf_display, unsafe_allow_html=True)
                        st.success("PDF generato con successo! Clicca sul link per scaricare.")
                    else:
                        st.error("Errore nella generazione del PDF")
        
        except Exception as e:
            st.error(f"❌ Errore durante l'elaborazione: {str(e)}")
    
    else:
        # Schermata iniziale
        st.info("""
        ### 👋 Benvenuto in HRV Analytics
        
        **Per iniziare:**
        1. **Carica un file** con gli intervalli RR (in millisecondi)
        2. **Aggiungi attività** dalla sidebar (opzionale)
        3. **Clicca 'Avvia Analisi Completa'**
        4. **Esplora i risultati** e genera il report PDF
        
        **Formato file supportato:** File di testo con un valore numerico per riga
        **Esempio:**
        ```
        800
        810
        795
        820
        ...
        ```
        """)

# Esegui l'app
if __name__ == "__main__":
    main()