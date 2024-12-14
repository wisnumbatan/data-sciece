import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Must be the first Streamlit command
st.set_page_config(
    page_title="Jelajah Probolinggo", 
    page_icon="🌄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced styling
# Update just the CSS styling section in app.py
st.markdown("""
<style>
    /* Dark theme with white text */
    :root {
        --primary-color: #2E8B57;
        --background-color: #0E1117;
        --text-color: #FFFFFF;
    }
    
    /* Apply dark background */
    .stApp {
        background-color: var(--background-color) !important;
    }
    
    /* Make all text white */
    h1, h2, h3, p, li, label, .stSelectbox, .stMarkdown, .stText, span {
        color: var(--text-color) !important;
    }
    
    /* Style headings */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem !important;
        font-weight: 700;
        color: var(--text-color) !important;
    }
    
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: var(--text-color) !important;
    }
    
    /* Style cards/expanders */
    .stExpander {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        color: var(--text-color) !important;
    }
    
    /* Style buttons */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), #45B08C);
        color: white !important;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border-radius: 8px;
    }
    
    /* Style selectbox and other inputs */
    .stSelectbox > div > div, 
    .stTextInput > div > div {
        background-color: rgba(255,255,255,0.1) !important;
        color: var(--text-color) !important;
    }
    
    /* Style metric containers */
    .css-1xarl3l, .css-1r6slb0 {
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Style metric labels and values */
    .metric-label, .metric-value {
        color: var(--text-color) !important;
    }
    
    /* Style dataframe */
    .dataframe {
        color: var(--text-color) !important;
    }
    
    /* Style tooltips and popups */
    div[data-baseweb="tooltip"], 
    div[data-baseweb="popover"] {
        background-color: #1E2530 !important;
        color: var(--text-color) !important;
    }
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
col1, col2 = st.columns([1, 4])
with col1:
    st.image("probolinggo.png", width=120)
with col2:
    st.title("✨ Jelajah Probolinggo")
    st.markdown("*Temukan Keindahan Tersembunyi Kota Probolinggo*")

# Load data
@st.cache_data
def load_data():
    # Load existing tourism data
    info_tourism = pd.read_csv('data sciece/data/tourism_with_id.csv')
    tourism_rating = pd.read_csv('data sciece/data/tourism_rating.csv')
    
    # Extract coordinates
    info_tourism[['latitude', 'longitude']] = info_tourism['Coordinate'].str.split(',', expand=True).astype(float)
    
    # Generate sample visit data for trending analysis
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    visit_data = []
    
    for place_id in info_tourism['Place_Id'].unique():
        base_visits = np.random.randint(50, 200)
        for date in dates:
            # Add some seasonality and randomness
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            daily_visits = int(base_visits * seasonal_factor * (1 + 0.2 * np.random.randn()))
            visit_data.append({
                'Place_Id': place_id,
                'Date': date,
                'Visits': max(0, daily_visits)
            })
    
    visits_df = pd.DataFrame(visit_data)
    
    # Continue with existing merging logic
    all_tourism = pd.merge(
        tourism_rating, 
        info_tourism[["Place_Id","Place_Name","Description","City","Category","latitude","longitude"]], 
        on='Place_Id', 
        how='inner'
    )
    
    # Create Description_category column
    all_tourism['Description_category'] = all_tourism[['Category','Description']].agg(' '.join, axis=1)
    
    # Drop duplicates to get unique places
    preparation = all_tourism.drop_duplicates('Place_Id')
    
    # Prepare final dataset
    tourism_new = pd.DataFrame({
        "id": preparation.Place_Id.tolist(),
        "name": preparation.Place_Name.tolist(),
        "category": preparation.Category.tolist(),
        "description": preparation.Description.tolist(),
        "city": preparation.City.tolist(),
        "Description_category": preparation.Description_category.tolist()
    })
    
    users = pd.read_csv('data sciece/data/user.csv')
    
    return tourism_new, info_tourism, visits_df, users

data, locations_data, visits_data, users_data = load_data()

# Create content-based recommendation system
@st.cache_resource
def create_recommendation_system(data):
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(data['Description_category'])
    cosine_sim = cosine_similarity(cv_matrix)
    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=data['name'],
        columns=data['name']
    )
    return cosine_sim_df

similarity_df = create_recommendation_system(data)

def tourism_recommendations(place_name, similarity_data=similarity_df, items=data[['name','category','description','city']], k=5):
    index = similarity_data.loc[:,place_name].to_numpy().argpartition(range(-1,-k,-1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Rekomendasi Wisata",
    "🗺️ Peta Interaktif",
    "📊 Statistik & Tren",
    "📅 Rencana Perjalanan",
    "💫 Preferensi Pribadi"
])

with tab1:
    col1, col2 = st.columns([1,2])

    with col1:
        st.header("Temukan Tempat Wisata")
        st.markdown("""
            <p style='font-size: 1.1em; color: #666;'>
            Pilih tempat wisata favorit Anda dan temukan rekomendasi serupa
            </p>
        """, unsafe_allow_html=True)
        selected_place = st.selectbox(
            "Pilih tempat wisata sebagai referensi:",
            options=data['name'].tolist()
        )
        
        if st.button("Cari Rekomendasi"):
            with col2:
                st.subheader("Rekomendasi Tempat Wisata Serupa")
                recommendations = tourism_recommendations(selected_place)
                
                for idx, row in recommendations.iterrows():
                    with st.expander(f"🏝️ {row['name']}"):
                        st.write(f"**Kategori:** {row['category']}")
                        st.write(f"**Kota:** {row['city']}")
                        st.write(f"**Deskripsi:** {row['description']}")

with tab2:
    st.subheader("Peta Lokasi Wisata")
    
    # Create base map
    m = folium.Map(location=[-7.9129, 112.9117], zoom_start=9)
    
    # Add markers for each location
    for idx, row in locations_data.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']], # Fixed the syntax error here
            popup=f"<b>{row['Place_Name']}</b><br>{row['Category']}<br>Rating: {row['Rating']}/5",
            tooltip=row['Place_Name']
        ).add_to(m)
    
    # Display map
    st_folium(m, width=800, height=500)

with tab3:
    st.subheader("Dashboard Analytics")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_places = len(locations_data)
        st.metric("Total Tempat Wisata", total_places)
    
    with col2:
        avg_rating = locations_data['Rating'].mean()
        st.metric("Rating Rata-rata", f"{avg_rating:.2f}")
        
    with col3:
        total_reviews = len(visits_data)
        st.metric("Total Kunjungan", f"{total_reviews:,}")
        
    with col4:
        avg_price = locations_data['Price'].mean()
        st.metric("Rata-rata Harga", f"Rp {avg_price:,.0f}")

    # Trending Places Chart
    st.subheader("Tren Kunjungan Wisata")
    
    # Aggregate visits by date
    daily_visits = visits_data.groupby('Date')['Visits'].sum().reset_index()
    
    fig = px.line(daily_visits, x='Date', y='Visits',
                  title='Tren Kunjungan Harian')
    st.plotly_chart(fig)
    
    # Popular Categories
    col1, col2 = st.columns(2)
    
    with col1:
        category_visits = locations_data.groupby('Category').size()
        fig = px.pie(values=category_visits.values, 
                     names=category_visits.index,
                     title='Distribusi Kategori Wisata')
        st.plotly_chart(fig)
        
    with col2:
        price_range = pd.cut(locations_data['Price'], 
                           bins=[0, 10000, 50000, 100000, float('inf')],
                           labels=['0-10k', '10k-50k', '50k-100k', '>100k'])
        price_dist = price_range.value_counts()
        fig = px.bar(x=price_dist.index, y=price_dist.values,
                    title='Distribusi Harga Tiket',
                    labels={'x': 'Range Harga', 'y': 'Jumlah Tempat'})
        st.plotly_chart(fig)

    # Top Rated Places
    st.subheader("Tempat Wisata dengan Rating Tertinggi")
    top_places = locations_data.nlargest(10, 'Rating')[['Place_Name', 'Category', 'Rating', 'City']]
    st.dataframe(top_places)

with tab4:
    st.subheader("🗺️ Rencana Perjalanan Wisata")
    
    # Create session state to persist the plan
    if 'show_plan' not in st.session_state:
        st.session_state.show_plan = False
    
    # Input for trip planning
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Tanggal Mulai", min_value=datetime.today())
        duration = st.number_input("Durasi (hari)", min_value=1, max_value=7, value=1)
        budget = st.number_input("Budget (Rp)", min_value=0, value=100000, step=50000)
    
    with col2:
        preferred_categories = st.multiselect(
            "Kategori Wisata Yang Diminati",
            options=locations_data['Category'].unique()
        )
        max_places_per_day = st.slider("Jumlah Tempat Per Hari", 1, 5, 3)

    if st.button("Buat Rencana"):
        st.session_state.show_plan = True
        st.session_state.plan_data = {
            'start_date': start_date,
            'duration': duration,
            'budget': budget,
            'categories': preferred_categories,
            'max_places': max_places_per_day
        }

    # Show the plan if it exists in session state
    if st.session_state.get('show_plan'):
        # Filter places based on preferences
        filtered_places = locations_data.copy()
        
        if st.session_state.plan_data['categories']:
            filtered_places = filtered_places[filtered_places['Category'].isin(st.session_state.plan_data['categories'])]
        
        # Filter by budget
        filtered_places = filtered_places[filtered_places['Price'] <= st.session_state.plan_data['budget']]
        
        # Sort by rating
        filtered_places = filtered_places.sort_values('Rating', ascending=False)
        
        # Create itinerary
        st.subheader("📋 Itinerary Perjalanan")
        
        total_budget = 0
        for day in range(st.session_state.plan_data['duration']):
            current_date = st.session_state.plan_data['start_date'] + timedelta(days=day)
            
            # Select places for the day
            day_places = filtered_places.head(st.session_state.plan_data['max_places'])
            filtered_places = filtered_places.iloc[st.session_state.plan_data['max_places']:]
            
            if len(day_places) > 0:
                st.markdown(f"### Hari {day + 1} - {current_date.strftime('%A, %d %B %Y')}")
                for _, place in day_places.iterrows():
                    total_budget += place['Price']
                    
                    st.markdown(f"""
                    * **{place['Place_Name']}**
                        * 🏷️ Kategori: {place['Category']}
                        * 💰 Harga: Rp {place['Price']:,}
                        * ⭐ Rating: {place['Rating']}/5
                        * 🕒 Estimasi waktu: {place['Time_Minutes']}
                        * 📍 Koordinat: {place['Coordinate']}
                    """)
            else:
                st.warning(f"Tidak ada tempat wisata yang tersedia untuk hari {day + 1}")

        # Rest of the code (summary, map, etc) remains the same
        st.subheader("💰 Ringkasan Biaya")
        st.info(f"""
        * Total Biaya Tiket: Rp {total_budget:,}
        * Sisa Budget: Rp {budget - total_budget:,}
        * Jumlah Tempat Wisata: {len(day_places) * duration}
        """)
        
        # Show route on map
        st.subheader("🗺️ Rute Perjalanan")
        trip_map = folium.Map(location=[filtered_places['latitude'].mean(), 
                                      filtered_places['longitude'].mean()], 
                            zoom_start=10)
        
        # Add markers and connect them with lines
        coordinates = []
        for _, place in day_places.iterrows():
            coordinates.append([place['latitude'], place['longitude']])
            folium.Marker(
                [place['latitude'], place['longitude']],
                popup=f"<b>{place['Place_Name']}</b><br>{place['Category']}<br>Rp {place['Price']:,}",
                tooltip=place['Place_Name']
            ).add_to(trip_map)
        
        # Add line connecting all places
        if len(coordinates) > 1:
            folium.PolyLine(
                coordinates,
                weight=2,
                color='red',
                opacity=0.8
            ).add_to(trip_map)
        
        st_folium(trip_map, width=800, height=500)

    # Add a clear button to reset the plan
    if st.session_state.get('show_plan'):
        if st.button("Buat Rencana Baru"):
            st.session_state.show_plan = False
            st.experimental_rerun()

with tab5:
    st.subheader("🎯 Rekomendasi Berdasarkan Preferensi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User preferences input
        user_location = st.selectbox(
            "Lokasi Anda:",
            options=sorted(users_data['Location'].unique())
        )
        
        user_age = st.slider(
            "Usia:",
            min_value=int(users_data['Age'].min()),
            max_value=int(users_data['Age'].max()),
            value=25
        )
        
        price_range = st.select_slider(
            "Range Harga (Rp):",
            options=[0, 10000, 25000, 50000, 100000, 250000, 300000],
            value=(0, 100000)
        )
        
    with col2:
        selected_categories = st.multiselect(
            "Kategori Wisata yang Diminati:",
            options=sorted(locations_data['Category'].unique()),
            default=locations_data['Category'].unique()[0]
        )
    
    if st.button("Dapatkan Rekomendasi Personal"):
        # Filter similar users
        similar_users = users_data[
            (users_data['Location'] == user_location) &
            (users_data['Age'].between(user_age - 5, user_age + 5))
        ]
        
        # Get place recommendations
        recommended_places = locations_data[
            (locations_data['Price'].between(price_range[0], price_range[1])) &
            (locations_data['Category'].isin(selected_categories))
        ].copy()
        
        # Calculate relevance score based on multiple factors
        recommended_places['relevance_score'] = (
            recommended_places['Rating'] * 0.4 +  # Rating weight
            (1 - (recommended_places['Price'] - price_range[0]) / 
             (price_range[1] - price_range[0])) * 0.3 +  # Price weight
            (recommended_places['Category'].isin(selected_categories)) * 0.3  # Category weight
        )
        
        # Sort and display top recommendations
        top_recommendations = recommended_places.nlargest(5, 'relevance_score')
        
        if len(top_recommendations) > 0:
            st.subheader("✨ Rekomendasi Tempat Wisata Untuk Anda")
            
            for idx, place in top_recommendations.iterrows():
                with st.expander(f"🏖️ {place['Place_Name']} - {place['Category']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Rating:** {'⭐' * int(place['Rating'])}")
                        st.write(f"**Harga:** Rp {place['Price']:,}")
                        st.write(f"**Kategori:** {place['Category']}")
                        
                    with col2:
                        st.write(f"**Lokasi:** {place['City']}")
                        st.write(f"**Durasi:** {place['Time_Minutes']}")
                        
                    st.write("**Deskripsi:**")
                    st.write(place['Description'])
                    
            # Show recommendation stats
            st.subheader("📊 Statistik Rekomendasi")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_price = top_recommendations['Price'].mean()
                st.metric("Rata-rata Harga", f"Rp {avg_price:,.0f}")
                
            with col2:
                avg_rating = top_recommendations['Rating'].mean()
                st.metric("Rata-rata Rating", f"{avg_rating:.1f} ⭐")
                
            with col3:
                categories = len(top_recommendations['Category'].unique())
                st.metric("Variasi Kategori", categories)
                
        else:
            st.warning("Tidak ditemukan tempat wisata yang sesuai dengan preferensi Anda. Coba sesuaikan filter Anda.")

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Dibuat dengan Streamlit oleh Kelompok 10</p>
    <p style='font-size: 0.8em;'>Sistem Rekomendasi Wisata Probolinggo © 2024</p>
</div>
""", unsafe_allow_html=True)